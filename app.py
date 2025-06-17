from flask import Flask, request, jsonify
import bcrypt
import jwt
import os
import json
from datetime import datetime, timedelta
import requests
from functools import wraps
from dotenv import load_dotenv
from flask_cors import CORS

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

# Define admin credentials
ADMIN_EMAIL = "admin@medicalassistant.com"
ADMIN_PASSWORD = "MedAssist@nyu1"

# In-memory storage for chats
chats = {}

# In-memory storage for conversation context
conversation_context = {}
chat_records = []

def get_initial_prompt():
    return '''1. 你是一位医疗助手，请通过与患者的对话提问，判断其感冒类型，并在必要时推荐常用药物与护理建议。你需要根据患者的描述逐步缩小范围，直到能够明确诊断。请避免重复提问，避免给出过早或错误的诊断。

---

2. 不同感冒的症状分别是：

a) 鼻部与呼吸道症状：

- 风寒感冒：清水样鼻涕，鼻塞较轻，喉咙发痒，咳嗽偏干，有时伴少量白痰；打喷嚏频繁。  
- 风热感冒：黄稠鼻涕，咽喉红肿疼痛，干咳或痰黄，咳嗽明显。  
- 暑湿感冒：鼻塞黏滞，咽喉不适但不剧烈，胸闷口黏，痰黏稠。  
- 虚弱性感冒：鼻塞轻微，症状不剧烈，常反复发作或拖延不愈，伴咽干。  
- 过敏性鼻炎：清涕如水、阵发性打喷嚏、鼻痒、眼痒，通常无咽痛和发热。  
- COVID/流感：喉咙剧痛、持续干咳、咽痒、咽干，有时伴味觉嗅觉丧失。

b) 全身症状：

- 风寒感冒：畏寒重于发热，无汗或少汗，四肢酸痛，头痛。  
- 风热感冒：发热明显，微恶风或恶寒，出汗、口渴、头胀痛。  
- 暑湿感冒：体温不高或微热，身重、乏力、出汗多、无口渴或口腻。  
- 虚弱性感冒：轻度发热或无热，容易疲劳，自汗，精神不振。  
- 过敏性鼻炎：无发热，不伴全身症状，晨起或过敏源接触时发作明显。  
- COVID/流感：高热、肌肉酸痛、乏力明显、寒战、头痛、虚脱感。

c) 发作诱因：

- 风寒感冒：着凉、吹风、淋雨、寒冷天气。  
- 风热感冒：气候干热、感染他人、空气污染等。  
- 暑湿感冒：高温湿热天气、吹空调后受凉、饮食生冷。  
- 虚弱性感冒：体质差、频繁感冒、劳累后复发。  
- 过敏性鼻炎：灰尘、花粉、冷空气等过敏原。  
- COVID/流感：接触病人、密闭空间、公共传播。

d) 舌象与口腔表现（如已观察）：

- 风寒感冒：舌苔薄白、口不渴。  
- 风热感冒：舌红苔黄、口干渴、口苦。  
- 暑湿感冒：舌苔腻或厚白、口腻。  
- 虚弱性感冒：舌淡、苔薄，口干或无特殊表现。  
- 过敏性鼻炎：舌苔正常或略白。  
- COVID/流感：口干、舌苔略黄，口腔异味可能。


---

3. 判断逻辑：

如果你已经给出诊断，跳过本条内容。  
否则请按以下逻辑执行：

- 若患者信息足够判断感冒类型，请输出明确判断；
- 若信息不够，请追加提问，聚焦未覆盖的重要维度（如诱因、痰色、舌苔、是否出汗、疲倦程度等）；
- 若患者症状表现出矛盾信息（如同时出现风寒与风热特征），请考虑以下三种情况之一：
  1. 属于混合型感冒（提示可能存在风寒夹热等）；
  2. 为其他类型疾病（如鼻窦炎、支气管炎、急性扁桃体炎）；
  3. 用户回答不一致，建议确认答案或引导其仔细观察症状。

---

5. 药物推荐与治疗建议：

在明确诊断后，继续询问患者是否需要药物建议。

如需要，请根据感冒类型、患者年龄、体质、慢病情况推荐以下药物（中西结合）：

- 风寒感冒：正柴胡饮颗粒、荆防颗粒、通宣理肺丸；配合对乙酰氨基酚  
- 风热感冒：银翘解毒片、连花清瘟、桑菊感冒颗粒；配合布洛芬退烧  
- 暑湿感冒：藿香正气水、香砂六君丸，注意补水  
- 虚弱性感冒：参苏丸、益气感冒颗粒、玉屏风散，避免寒凉药物  
- COVID/流感：连花清瘟、对乙酰氨基酚、右美沙芬、布洛芬（高烧）  
- 过敏性鼻炎：氯雷他定、左西替利嗪、鼻炎通窍颗粒、布地奈德喷雾

如患者为孕妇、婴幼儿或高龄慢病者，请额外提示用药禁忌与就医建议。

最后，提供简洁康复建议，如：

- 多饮温水，适当休息  
- 保持室内通风，避免着凉  
- 三日内症状无改善或加重请就医  
'''

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

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'message': 'Token is missing'}), 401
        try:
            token = token.split(' ')[1]
            data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            current_user = users.get(data['email'])
            if not current_user:
                return jsonify({'message': 'User not found'}), 401
        except:
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

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    name = data.get('name')

    if not all([email, password, name]):
        return jsonify({"success": False, "message": "All fields are required"}), 400

    if email in users:
        return jsonify({"success": False, "message": "Email already registered"}), 400

    # Store user
    users[email] = {
        'email': email,
        'password': password,  # In production, hash the password
        'name': name
    }

    return jsonify({"success": True, "message": "Registration successful"}), 201

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({"success": False, "message": "Email and password are required"}), 400

    user = users.get(email)
    if not user or user['password'] != password:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

    return jsonify({
        "success": True,
        "message": "Login successful",
        "name": user['name']
    })

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
        return jsonify({"success": True, "message": "Login successful"})
    else:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    print(f"Received {request.method} request to /api/chat")
    print(f"Origin: {request.headers.get('Origin', 'None')}")

    if request.method == 'OPTIONS':
        print("Handling OPTIONS preflight request")
        return '', 200

    data = request.get_json()
    message = data.get('message', '')
    print(f"Message received: {message}")

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    # ---------- 构建对话上下文（记忆） ----------
    user_id = data.get('user_id') or request.remote_addr
    if user_id not in conversation_context:
        conversation_context[user_id] = []  # 存储每轮 {user, assistant}

    history = conversation_context[user_id]

    # 生成历史文本
    history_lines = [f"用户: {h['user']}" for h in history]
    history_text = "\n".join(history_lines)
    concise_rule = "医生口吻，回复≤60字；信息不足时仅问一个新的关键问题(单句，无序号无客套)，禁止总结解释；收集完必要信息后再给出诊断与用药建议，当收集信息有矛盾时，向用户确认或者考虑可能性。"

    system_history = f"{concise_rule}\n\n以下是此前对话历史：\n{history_text}"

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
        chat_records.append({
            'userId': user_id,
            'userMessage': message,
            'assistantMessage': ai_response,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })

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

@app.route('/api/chat/history', methods=['GET'])
@token_required
def chat_history(current_user):
    user_chats = [chats[chat_id] for chat_id in current_user['chats']]
    return jsonify(user_chats)

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def admin_stats():
    total_users = len(users)
    active_chats = len([c for c in chats.values() if (datetime.utcnow() - datetime.fromisoformat(c['timestamp'])).days < 1])
    total_messages = len(chats)
    
    return jsonify({
        'totalUsers': total_users,
        'activeChats': active_chats,
        'totalMessages': total_messages
    })

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def admin_users():
    user_list = [{
        'id': email,
        'name': user['name'],
        'email': email,
        'status': 'active'
    } for email, user in users.items()]
    return jsonify(user_list)

@app.route('/api/admin/chats', methods=['GET'])
@admin_required
def admin_get_chats():
    return jsonify(chat_records)

@app.route('/api/admin/chats/clear', methods=['POST'])
@admin_required
def admin_clear_chats():
    chat_records.clear()
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
def clear_chat():
    data = request.get_json() or {}
    user_id = data.get('user_id') or request.remote_addr
    if user_id in conversation_context:
        del conversation_context[user_id]
        print(f"Conversation history for {user_id} cleared")
    return jsonify({'success': True})

if __name__ == '__main__':
    print("Starting Flask server on port 5050...")
    print("Test URL: http://localhost:5050/test")
    print(f"DeepSeek API Key loaded: {'Yes' if DEEPSEEK_API_KEY != 'YOUR_DEEPSEEK_API_KEY' else 'No - check .env file'}")
    app.run(debug=True, host='0.0.0.0', port=5050) 