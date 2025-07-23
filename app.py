from flask import Flask, request, jsonify, send_from_directory
import bcrypt
import jwt
import os
import json
from datetime import datetime, timedelta
import requests
from functools import wraps
from dotenv import load_dotenv
from flask_cors import CORS
import time

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow all origins for testing

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    response.headers.add('Access-Control-Max-Age', '86400')
    print(f"Added CORS headers to {request.method} {request.path}")
    return response

# Secret key for JWT
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', 'YOUR_DEEPSEEK_API_KEY')

# File paths
USERS_FILE = 'users.json'

# ----------------- Admin credentials -----------------
ADMIN_CRED_FILE = 'admin_credentials.txt'

def load_admin_credentials():
    if os.path.exists(ADMIN_CRED_FILE):
        try:
            with open(ADMIN_CRED_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('email'), data.get('password_hash')
        except Exception as e:
            print(f"Error loading admin credentials: {e}")
    # Fallback to None values if file missing
    return None, None

ADMIN_EMAIL, ADMIN_PW_HASH = load_admin_credentials()

# In-memory trackers
conversation_context = {}  # {username: [history]}

# Persistent chat history -----------------
CHAT_HISTORY_FILE = 'chat_history.json'

def load_chat_records():
    if os.path.exists(CHAT_HISTORY_FILE):
        with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_chat_records():
    with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(chat_records, f, ensure_ascii=False, indent=2)

chat_records = load_chat_records()  # list of message dicts

# ----------------- Patient history -----------------
PATIENT_HISTORY_FILE = 'patient_history.json'

def load_patient_histories():
    if os.path.exists(PATIENT_HISTORY_FILE):
        with open(PATIENT_HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_patient_histories(data):
    with open(PATIENT_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

patient_histories = load_patient_histories()  # {username: [ {user, assistant, timestamp} ]}

online_users = {}          # {username: last_seen_datetime}

MED_DB_FILE = 'medical_db.json'

# ----------------- Medical DB helpers -----------------

def load_med_db():
    if os.path.exists(MED_DB_FILE):
        with open(MED_DB_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {"diseases": []}
    return {"diseases": []}

def save_med_db(db):
    with open(MED_DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

# --------------- Dynamic prompt builder ---------------

def build_med_db_text(db):
    """Convert JSON DB into prompt-friendly markdown."""
    lines = []
    systems = {}
    for d in db.get('diseases', []):
        systems.setdefault(d['system'], []).append(d)
    # Sort systems alphabetically
    for sys_code in sorted(systems.keys()):
        sys_list = systems[sys_code]
        # assume all items share system_name
        lines.append(f"## {sys_code}. {sys_list[0]['system_name']}")
        for item in sorted(sys_list, key=lambda x: x['id']):
            lines.append(f"\n### {item['id']}. {item['name']}")
            if item.get('judgement_factors'):
                lines.append("\n#### 判断因子")
                for lvl, factors in item['judgement_factors'].items():
                    lvl_num = {'level1':'一级','level2':'二级','level3':'三级'}.get(lvl, lvl)
                    lines.append(f"* **{lvl_num}**：")
                    for fct in factors:
                        lines.append(f"  - {fct}")
            if item.get('decision_paths'):
                lines.append("\n#### 决策路径")
                for idx, path in enumerate(item['decision_paths'], 1):
                    lines.append(f"{idx}. {path}")
    return "\n".join(lines)

# Override get_initial_prompt

def get_initial_prompt():
    with open('prompt_react_cn.txt', 'r', encoding='utf-8') as f:
        template = f.read()

    db_markdown = build_med_db_text(load_med_db())

    # 首选占位符替换
    if '[MEDICAL_DATABASE]' in template:
        return template.replace('[MEDICAL_DATABASE]', db_markdown)

    # 其次在 ## 医疗知识库 标记后插入
    marker = '## 医疗知识库'
    if marker in template:
        before, after = template.split(marker, 1)
        return before + marker + '\n\n' + db_markdown + '\n' + after

    # 默认返回原模板
    return template

# Load users from file
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

# Save users to file
def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

# Initialize users
users = load_users()

def generate_token(username: str, is_admin: bool = False):
    """Generate a JWT token valid for 7 days."""
    return jwt.encode({
        'username': username,
        'is_admin': is_admin,
        'exp': datetime.utcnow() + timedelta(days=7)
    }, SECRET_KEY, algorithm='HS256')

def token_required(f):
    """Decorator to ensure the request carries a valid JWT token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        # Allow OPTIONS pre-flight requests without token
        if request.method == 'OPTIONS':
            return f(None, *args, **kwargs)

        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'message': 'Token is missing'}), 401

        token = auth_header.split(' ')[1]
        try:
            data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            username = data['username']
            current_user = users.get(username)
            if not current_user:
                return jsonify({'message': 'User not found'}), 401
        except jwt.ExpiredSignatureError:
            return jsonify({'message': 'Token has expired'}), 401
        except Exception:
            return jsonify({'message': 'Token is invalid'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            token = token.split(' ')[1]
            data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            if not data.get('is_admin'):
                return jsonify({'message': 'Admin access required'}), 403
        except:
            return jsonify({'message': 'Token is invalid'}), 401
        return f(*args, **kwargs)
    return decorated

# ----------------- Simple rate limiting -----------------
RATE_LIMIT_WINDOW_SEC = 12 * 60 * 60  # 12 hours
RATE_LIMIT_MAX = 100
ip_rate_table = {}  # {ip: {'start': epoch, 'count': int}}

def check_rate_limit(ip: str):
    """Return True if within limit, False if exceeded."""
    now = time.time()
    rec = ip_rate_table.get(ip)
    if not rec or now - rec['start'] > RATE_LIMIT_WINDOW_SEC:
        # New window
        ip_rate_table[ip] = {'start': now, 'count': 1}
        return True
    if rec['count'] >= RATE_LIMIT_MAX:
        return False
    rec['count'] += 1
    return True

@app.route('/')
def index():
    # Serve login.html by default; chat.html is loaded after successful login from frontend
    return send_from_directory('.', 'login.html')

@app.route('/<path:filename>')
def static_files(filename):
    # Serve other static files (like styles.css) from the current directory
    return send_from_directory('.', filename)

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    name = data.get('name', username)

    if not username or not password:
        return jsonify({"success": False, "message": "Username and password are required"}), 400

    if username in users:
        return jsonify({"success": False, "message": "Username already exists"}), 400

    # Hash the password for secure storage
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    users[username] = {
        'username': username,
        'password': hashed_pw,
        'name': name
    }

    save_users(users)

    token = generate_token(username)

    return jsonify({
        "success": True,
        "message": "Registration successful",
        "token": token,
        "name": name
    }), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not all([username, password]):
        return jsonify({"success": False, "message": "Username and password are required"}), 400

    user = users.get(username)
    if not user:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    if not bcrypt.checkpw(password.encode('utf-8'), user['password'].encode('utf-8')):
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    token = generate_token(username)

    return jsonify({
        "success": True,
        "message": "Login successful",
        "name": user['name'],
        "token": token
    })

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    # Use stored hash comparison
    if email == ADMIN_EMAIL and ADMIN_PW_HASH and bcrypt.checkpw(password.encode('utf-8'), ADMIN_PW_HASH.encode('utf-8')):
        token = generate_token("admin", is_admin=True)
        return jsonify({"success": True, "message": "Login successful", "token": token})
    else:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/chat', methods=['OPTIONS'])
def chat_preflight():
    """Handle CORS pre-flight requests without auth."""
    return '', 200

@app.route('/api/chat', methods=['POST'])
@token_required
def chat(current_user):
    # Use authenticated username as the conversation id
    user_id = current_user['username']

    # Mark user as online / update last_seen
    online_users[user_id] = datetime.utcnow()

    print(f"Received {request.method} request to /api/chat")
    print(f"Origin: {request.headers.get('Origin', 'None')}")

    if request.method == 'OPTIONS':
        print("Handling OPTIONS preflight request")
        return '', 200

    # ---- Rate limiting per IP ----
    client_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if not check_rate_limit(client_ip):
        return jsonify({'response': 'Rate limit reached, try again later.'}), 429

    data = request.get_json()
    message = data.get('message', '')
    print(f"Message received: {message}")

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # ---------- 构建对话上下文（记忆） ----------
    if user_id not in conversation_context:
        conversation_context[user_id] = []  # 存储每轮 {user, assistant}

    history = conversation_context[user_id]

    # 生成历史文本
    history_lines = []
    for h in history:
        history_lines.append(f"用户: {h['user']}")
        history_lines.append(f"助手: {h['assistant']}")
    history_text = "\n".join(history_lines)
    concise_rule = "医生口吻，回复≤60字；信息不足时仅问一个新的关键问题(单句，无序号无客套)，禁止总结解释；收集完必要信息后再给出诊断与用药建议，当收集信息有矛盾时，向用户确认或者考虑可能性。"

    # 注入跨会话就诊历史
    pat_hist = patient_histories.get(user_id, [])
    pat_lines = []
    for itm in pat_hist:
        pat_lines.append(f"患者: {itm['user']}")
        pat_lines.append(f"医生: {itm['assistant']}")
    pat_text = "\n".join(pat_lines) if pat_lines else '无'

    patient_block = f"该用户此前的就诊历史：{pat_text}"

    system_history = f"{concise_rule}\n\n{patient_block}\n\n以下是此前对话历史：\n{history_text}"

    messages = [
        {"role": "system", "content": get_initial_prompt()},
        {"role": "system", "content": system_history},
        {"role": "user", "content": message}
    ]

    url = 'https://api.deepseek.com/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': 'deepseek-chat',
        'messages': messages,
        'temperature': 0.5,
        'max_tokens': 200,
        'top_p': 0.9
    }

    # 调试打印
    import json as _j
    print("===== Payload 发往 DeepSeek =====")
    print(_j.dumps(payload, ensure_ascii=False, indent=2))
    print("===== End Payload =====")

    try:
        print("Calling DeepSeek API...")
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        print(f"DeepSeek API Status Code: {response.status_code}")
        print(f"DeepSeek API Response Headers: {dict(response.headers)}")

        response.raise_for_status()

        response_data = response.json()
        print(f"DeepSeek API Response: {response_data}")

        ai_response = response_data['choices'][0]['message']['content']
        print("DeepSeek responded successfully")

        # 保存本轮对话到历史
        conversation_context[user_id].append({
            'user': message,
            'assistant': ai_response
        })
        
        # 如历史过长，可截断至最近20轮
        if len(conversation_context[user_id]) > 40:
            conversation_context[user_id] = conversation_context[user_id][-40:]

        # 保存全局聊天记录供 Admin
        chat_record = {
            'userId': user_id,
            'userMessage': message,
            'assistantMessage': ai_response,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        chat_records.append(chat_record)
        save_chat_records()

        # ----------- 保存就诊历史 -----------
        patient_histories.setdefault(user_id, []).append({
            'user': message,
            'assistant': ai_response,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
        save_patient_histories(patient_histories)

        # Update last_seen again after processing
        online_users[user_id] = datetime.utcnow()

        return jsonify({'response': ai_response})
    except requests.exceptions.HTTPError as e:
        print(f'DeepSeek API HTTP Error: {e}')
        print(f'Response content: {response.text}')
        return jsonify({'response': "Sorry, I'm having trouble connecting to the AI right now."}), 500
    except KeyError as e:
        print(f'DeepSeek API Response Parsing Error: {e}')
        print(f'Full response: {response.json()}')
        return jsonify({'response': "Sorry, I received an unexpected response format from the AI."}), 500
    except Exception as e:
        print(f'DeepSeek API General Error: {e}')
        print(f'Error type: {type(e).__name__}')
        return jsonify({'response': "Sorry, I'm having trouble connecting to the AI right now."}), 500

@token_required
def chat_history(current_user):
    history = conversation_context.get(current_user['username'], [])
    return jsonify(history)

@app.route('/api/history', methods=['GET'])
@token_required
def get_history(current_user):
    return jsonify(patient_histories.get(current_user['username'], []))

@app.route('/api/history/clear', methods=['POST'])
@token_required
def clear_history_endpoint(current_user):
    username = current_user['username']
    
    # ================= COMPLETE HISTORY CLEARANCE =================
    # 1. Clear persistent history
    patient_histories[username] = []
    save_patient_histories(patient_histories)
    
    # 2. Clear current session context
    if username in conversation_context:
        del conversation_context[username]
    
    # 3. Clear chat records for admin view
    global chat_records
    chat_records = [r for r in chat_records if r['userId'] != username]
    save_chat_records()
    # ================= END FIX =================
    
    return jsonify({'success': True})

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    cleanup_online_users()
    total_users = len(users)
    active_users = len(online_users)
    active_chats = len(conversation_context)
    total_messages = len(chat_records)
    
    return jsonify({
        'totalUsers': total_users,
        'activeUsers': active_users,
        'activeChats': active_chats,
        'totalMessages': total_messages
    })

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def admin_users():
    cleanup_online_users()
    user_list = []
    for username, info in users.items():
        status = 'active' if username in online_users else 'offline'
        user_list.append({
            'username': username,
            'name': info.get('name', username),
            'status': status
        })
    return jsonify(user_list)

@app.route('/api/admin/chats', methods=['GET'])
@admin_required
def admin_get_chats():
    return jsonify(chat_records)

@app.route('/api/admin/chats/clear', methods=['POST'])
@admin_required
def admin_clear_chats():
    chat_records.clear()
    save_chat_records()
    return jsonify({'success': True})

@app.route('/api/admin/settings', methods=['POST'])
@admin_required
def admin_settings():
    data = request.get_json()
    global DEEPSEEK_API_KEY
    DEEPSEEK_API_KEY = data.get('apiKey', DEEPSEEK_API_KEY)
    return jsonify({'message': 'Settings updated'})

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'status': 'Flask server is running with CORS!'})

@app.route('/api/chat/clear', methods=['POST'])
@token_required
def clear_chat(current_user):
    user_id = current_user['username']
    if user_id in conversation_context:
        del conversation_context[user_id]
        print(f"Conversation history for {user_id} cleared")
    return jsonify({'success': True})

# ----------------- Admin endpoints for medical db -----------------

@app.route('/api/admin/medical_db', methods=['GET', 'POST'])
@admin_required
def admin_medical_db():
    if request.method == 'GET':
        return jsonify(load_med_db())
    data = request.get_json()
    if not data or 'diseases' not in data:
        return jsonify({'message': 'Invalid payload'}), 400
    save_med_db(data)
    return jsonify({'success': True})

# ----------------- User session helpers -----------------

ONLINE_THRESHOLD_SEC = 300  # 5 minutes

def cleanup_online_users():
    """Remove users who have been inactive for longer than threshold."""
    stale = [u for u, ts in online_users.items() if (datetime.utcnow() - ts).total_seconds() > ONLINE_THRESHOLD_SEC]
    for u in stale:
        del online_users[u]

@app.route('/api/logout', methods=['POST'])
@token_required
def logout(current_user):
    username = current_user['username']
    # Remove from online list
    online_users.pop(username, None)
    # 持久化就诊历史
    save_patient_histories(patient_histories)
    return jsonify({'success': True})

if __name__ == '__main__':
    print("Starting Flask server on port 5050...")
    print("Test URL: http://localhost:5050/test")
    print(f"DeepSeek API Key loaded: {'Yes' if DEEPSEEK_API_KEY != 'YOUR_DEEPSEEK_API_KEY' else 'No - check .env file'}")
    app.run(debug=True, host='0.0.0.0', port=5050) 