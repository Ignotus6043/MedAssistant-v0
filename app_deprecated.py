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
import re
from collections import defaultdict, Counter

# =========================
# CONFIGURABLE PARAMETERS
# =========================
"""
All configurable parameters grouped at top. Changing values below does not change logic.
"""
# Server
SERVER_HOST = '0.0.0.0'
SERVER_PORT = 5050
SERVER_DEBUG = True

# Files
PROMPT_FILE = 'prompt_react_cn.txt'
USERS_FILE = 'users.json'
ADMIN_CRED_FILE = 'admin_credentials.txt'
CHAT_HISTORY_FILE = 'chat_history.json'
PATIENT_HISTORY_FILE = 'patient_history.json'
MED_DB_FILE = 'medical_db.json'
# Use disease taxonomy v2 as the live checklist/taxonomy source
CHECKLIST_FILE = 'disease_taxonomy_v2.json'

# Auth & Sessions
ONLINE_THRESHOLD_SEC = 1800  # 30 minutes inactive -> offline

# Rate limiting
RATE_LIMIT_WINDOW_SEC = 12 * 60 * 60  # 12 hours
RATE_LIMIT_MAX = 100

# DeepSeek API
DEEPSEEK_URL = 'https://api.deepseek.com/v1/chat/completions'
DEEPSEEK_MODEL = 'deepseek-chat'
DS_TEMPERATURE = 0.3
DS_MAX_TOKENS = 300
DS_TOP_P = 0.9
DS_TIMEOUT = 120

# Prompt injection budgets
CHAR_BUDGET = 1500  # max characters injected for checklist snippet
TOPK_STEP1_CATEGORIES = 6
CATEGORY_CC_MAX = 8
TOPK_STEP2_SYSTEMS = 2
TOPK_STEP2_DISEASES_PER_SYSTEM = 3
TOPK_STEP3_DISEASES = 3
KEYPOINTS_TOPK = 4
STEP_GE4_TOPN = 2   # steps 4-5 top diseases
STEP_GT5_TOPN = 1   # steps 6-7 top diseases
MIN_OVERLAP_FOR_DB = 1  # 至少有1个关键词重合才认为数据库可用

# History budgets
CONVO_HISTORY_LIMIT = 40
USER_STATE_HISTORY_LIMIT = 50
STATE_HISTORY_RETURN_LAST_N = 100

# NLP helpers
COMMON_PREFIXES = ["是否", "有无", "是否出现", "是否存在", "是否为", "是否合并", "是否伴", "当前是否", "近期是否", "夜间是否", "清晨是否"]
SPLIT_DELIMS = r"[、/或和,，;；\\s]+"
KEYWORD_STOP_WORDS = set(["我", "有", "了", "感觉", "最近", "一直", "可能", "需要", "想", "是否", "出现", "存在", "伴", "症状", "疼", "痛", "请", "帮", "一下", "今天", "昨天"]) 
RED_FLAG_TOKENS = ['警示', '红旗', '呼吸困难', '气促', '意识', '抽搐', '高热', '昏迷', '出血', '胸痛', '紫绀', '休克', '低血压', '剧烈', '窒息']

# Utility: parse disease key like '1_疾病名'

def parse_disease_key(name: str) -> dict:
    m = re.match(r"^(\d+)_([\s\S]+)$", name)
    if m:
        return {"id": int(m.group(1)), "name": m.group(2)}
    return {"id": None, "name": name}

# Lightweight keyword extraction (fallback only when LLM未提供)

def _preprocess_text_for_keywords(text: str) -> str:
    """Heuristically split常见并列症状（无分隔符），提高匹配度。"""
    if not text:
        return ""
    replacements = {
        '头晕头痛': '头晕 头痛',
        '头痛头晕': '头痛 头晕',
        '恶心呕吐': '恶心 呕吐',
        '鼻塞流涕': '鼻塞 流涕',
        '咽痛咳嗽': '咽痛 咳嗽',
        '胸闷胸痛': '胸闷 胸痛',
        '腹痛腹泻': '腹痛 腹泻',
        '肌肉酸痛乏力': '肌肉酸痛 乏力'
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def extract_keywords(text: str) -> list:
    if not text:
        return []
    text = _preprocess_text_for_keywords(text)
    text = re.sub(r"[。！？!?,，;；\n]", " ", text)
    text = re.sub(r"（.*?）", " ", text)
    words = re.split(r"\s+|/|、|,|，|;|；|或|和", text)
    words = [w.strip() for w in words if 1 < len(w.strip()) <= 8]
    res = []
    seen = set()
    for w in words:
        if w in KEYWORD_STOP_WORDS:
            continue
        if w not in seen:
            seen.add(w)
            res.append(w)
    return res[:12]

# ------------- Choice normalization (parse A./B./C. options) -------------

OPTION_PREFIXES = ["A.", "B.", "C.", "D.", "E.", "F.", "G.", "H."]

def _parse_option_mapping(assistant_text: str) -> dict:
    mapping = {}
    if not assistant_text:
        return mapping
    for line in str(assistant_text).split("\n"):
        line = line.strip()
        for i, p in enumerate(OPTION_PREFIXES):
            if line.startswith(p):
                key = chr(ord('A') + i)
                mapping[key] = line[len(p):].strip()
                break
    return mapping

def _normalize_user_choice(user_text: str) -> str:
    if not isinstance(user_text, str):
        return ""
    s = user_text.strip()
    if not s:
        return ""
    # Accept formats: 'A' / 'A.' / 'a' / '1' etc
    first = s[0].upper()
    if first in [chr(ord('A') + i) for i in range(len(OPTION_PREFIXES))]:
        return first
    if first.isdigit():
        idx = int(first) - 1
        if 0 <= idx < len(OPTION_PREFIXES):
            return chr(ord('A') + idx)
    return ""

# Extract a normalized question stem to suppress duplicates
def _extract_question_stem(assistant_text: str) -> str:
    if not assistant_text:
        return ""
    lines = [ln.strip() for ln in str(assistant_text).split("\n") if ln.strip()]
    for ln in lines:
        if not any(ln.startswith(p) for p in OPTION_PREFIXES):
            # remove trailing punctuation
            s = re.sub(r"[？?。!！]$", "", ln)
            s = re.sub(r"\s+", " ", s)
            return s[:120]
    return ""

def _extract_question_stems(assistant_text: str) -> list:
    if not assistant_text:
        return []
    lines = [ln.strip() for ln in str(assistant_text).split("\n") if ln.strip()]
    stems = []
    for ln in lines:
        if any(ln.startswith(p) for p in OPTION_PREFIXES):
            continue
        # treat as a question headline
        s = re.sub(r"[？?。!！]$", "", ln)
        s = re.sub(r"\s+", " ", s)
        # 更严格的问题提取：移除常见前缀词
        for prefix in COMMON_PREFIXES:
            if s.startswith(prefix):
                s = s[len(prefix):].strip()
                break
        if s and (not stems or s != stems[-1]):
            stems.append(s[:120])
    return stems

def _parse_response_question_and_options(response_text: str) -> tuple[str, list]:
    """Parse the assistant 'Response' field into (question_stem, options_list).
    Options are the lines starting with 'A.'...'H.'
    """
    if not isinstance(response_text, str):
        return "", []
    lines = [ln.rstrip() for ln in response_text.split("\n")]
    question = ""
    options = []
    for ln in lines:
        if any(ln.strip().startswith(p) for p in OPTION_PREFIXES):
            options.append(ln.strip())
        elif not question and ln.strip():
            q = re.sub(r"[？?。!！]$", "", ln.strip())
            question = q[:120]
    return question, options

# =========================
# APPLICATION SETUP
# =========================

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

Med_DB_FILE_UNUSED = None  # legacy placeholder to avoid accidental reuse

# ----------------- Medical DB & Taxonomy helpers -----------------

# Cache for taxonomy file
_CHECKLIST_CACHE = None
_CHECKLIST_MTIME = None

def load_checklist() -> dict:
    """Load and cache `disease_taxonomy_v2.json`. Returns {} on failure."""
    global _CHECKLIST_CACHE, _CHECKLIST_MTIME
    try:
        if not os.path.exists(CHECKLIST_FILE):
            return {}
        mtime = os.path.getmtime(CHECKLIST_FILE)
        if _CHECKLIST_CACHE is None or _CHECKLIST_MTIME != mtime:
            with open(CHECKLIST_FILE, 'r', encoding='utf-8') as f:
                _CHECKLIST_CACHE = json.load(f)
            _CHECKLIST_MTIME = mtime
        return _CHECKLIST_CACHE or {}
    except Exception as e:
        print(f"Failed to load checklist: {e}")
        return {}

# === NEW: Question duplication check ===

def _is_semantically_similar_question(q1: str, q2: str) -> bool:
    """检查两个问题是否语义相似。
    移除常见前缀词和标点符号后进行比较。
    """
    def normalize(q: str) -> str:
        q = re.sub(r"[？?。!！]$", "", q.strip())
        q = re.sub(r"\s+", " ", q)
        for prefix in COMMON_PREFIXES:
            if q.startswith(prefix):
                q = q[len(prefix):].strip()
        return q.lower()
    
    return normalize(q1) == normalize(q2)

def _check_question_duplication(new_question: str, history: list, asked_questions: list = None) -> tuple[bool, list]:
    """检查新问题是否与历史问题重复。
    返回 (is_duplicate, existing_questions)
    """
    if not new_question:
        return False, []
    
    # 首先从历史记录中收集问题和选项
    existing_qa = []
    if history:
        for h in history:
            question, options = _parse_response_question_and_options(h.get('assistant', ''))
            if question:
                existing_qa.append({"question": question, "options": options})
    
    # 如果提供了已问问题集合，添加到现有问题中
    if asked_questions:
        existing_qa.extend(asked_questions)
        print(f"Current asked_questions: {asked_questions}")
    
    # 使用更准确的问题解析
    new_question_stem, new_options = _parse_response_question_and_options(new_question)
    if new_question_stem:
        for qa in existing_qa:
            # 检查问题是否语义相似
            if _is_semantically_similar_question(new_question_stem, qa["question"]):
                return True, existing_qa
            # 检查选项是否相似
            if new_options and qa["options"]:
                # 计算选项的重叠度
                common_options = set(new_options) & set(qa["options"])
                if len(common_options) >= min(len(new_options), len(qa["options"])) * 0.7:  # 如果70%以上的选项重叠
                    return True, existing_qa
    
    return False, existing_qa

# === NEW: auto-derive chief complaints ===

def _derive_chief_complaints(dis: dict, max_items: int = 4):
    """Select up to `max_items` 主诉采集 symptoms names ordered by priority."""
    if not dis.get('symptoms'):
        return []
    main = [s for s in dis['symptoms'] if s.get('type') == '主诉采集']
    main_sorted = sorted(main, key=lambda s: s.get('priority', 99))
    return [s['name'] for s in main_sorted[:max_items]]

# Update load_med_db so every disease has chief_complaints

def load_med_db():
    """Load DB and ensure chief_complaints field exists."""
    if os.path.exists(MED_DB_FILE):
        with open(MED_DB_FILE, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    # Wrap list into dict
    if isinstance(data, list):
        data = {"diseases": data}
    # 不再自动推断 chief_complaints，要求在 medical_db.json 中显式给出
    return data

# ---------- persist helper ----------

def save_med_db(db):
    """Save the in-memory DB back to MED_DB_FILE."""
    with open(MED_DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

# === NEW: overview builder ===

def build_med_db_overview_text(db):
    """Return markdown: system -> id name | 主诉 A,B."""
    diseases_list = db.get('diseases', []) if isinstance(db, dict) else db
    lines = []
    systems = {}
    for d in diseases_list:
        systems.setdefault(d['system'], []).append(d)
    for sys_code in sorted(systems.keys()):
        sys_list = systems[sys_code]
        lines.append(f"## {sys_code}. {sys_list[0]['system_name']}")
        for item in sorted(sys_list, key=lambda x: x['id']):
            cc_list=_chief_list(item)
            if cc_list:
                cc="、".join(cc_list[:4])
                lines.append(f"{item['id']}. {item['name']} | 主诉: {cc}")
            else:
                lines.append(f"{item['id']}. {item['name']}")
    return "\n".join(lines)

# Keep old detailed builder but rename

def build_med_db_full_text(db):
    """Full detailed text (keeps existing build_med_db_text logic)."""
    if isinstance(db, dict):
        diseases_list = db.get('diseases', [])
    else:
        diseases_list = db
    lines = []
    systems = {}
    for d in diseases_list:
        systems.setdefault(d['system'], []).append(d)
    for sys_code in sorted(systems.keys()):
        sys_list = systems[sys_code]
        lines.append(f"## {sys_code}. {sys_list[0]['system_name']}")
        for item in sorted(sys_list, key=lambda x: x['id']):
            lines.append(f"\n### {item['id']}. {item['name']}")
            ccl=_chief_list(item)
            if ccl:
                lines.append(f"* 主诉关键词: {'、'.join(ccl)}")
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
            if item.get('symptoms'):
                lines.append("\n#### 详细症状")
                for s in item['symptoms']:
                    opts = "、".join(s.get('options', []))
                    lines.append(f"- {s['name']}（{s['type']}，优先级{s['priority']}）可选: {opts}")
    return "\n".join(lines)

# (Old get_initial_prompt removed; unified version defined later.)

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

# ============== New user state management ==============
user_states = defaultdict(lambda: {
    "current_step": -1, 
    "demographics_known": False, 
    "possible_diseases_paths": [], 
    "age": None, 
    "sex": None, 
    "chief_complaints": [], 
    "collected_answers": {}, 
    "red_flags_checked": False, 
    "history": [],
    "asked_questions": []  # 新增：用于跟踪所有已问过的问题，每个元素是 {question: str, options: list}
})


def update_demographics_known(user_id: str):
    state = user_states[user_id]
    if state.get('age') is not None and state.get('sex') in ('男', '女'):
        state['demographics_known'] = True
    elif state.get('current_step', -1) >= 1 and state.get('demographics_known'):
        pass

# Robust JSON parsing for model outputs

def parse_llm_json(text: str) -> dict:
    if not text:
        return {}
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        if text.endswith('```'):
            text = text[:-3]
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(text)
    except Exception:
        return {}

# =========================
# RATE LIMITING
# =========================

ip_rate_table = {}  # {ip: {'start': epoch, 'count': int}}

def check_rate_limit(ip: str):
    now = time.time()
    rec = ip_rate_table.get(ip)
    if not rec or now - rec['start'] > RATE_LIMIT_WINDOW_SEC:
        ip_rate_table[ip] = {'start': now, 'count': 1}
        return True
    if rec['count'] >= RATE_LIMIT_MAX:
        return False
    rec['count'] += 1
    return True

# =========================
# CHECKLIST SEGMENTATION & MATCHING
# =========================

def parse_summary(summary_obj, state: dict):
    if not isinstance(summary_obj, dict):
        return
    age = summary_obj.get('age') or summary_obj.get('年龄')
    sex = summary_obj.get('sex') or summary_obj.get('性别')
    try:
        if age is not None:
            state['age'] = int(age)
    except Exception:
        pass
    if isinstance(sex, str):
        if '女' in sex:
            state['sex'] = '女'
        elif '男' in sex:
            state['sex'] = '男'
    cc = summary_obj.get('chief_complaints') or summary_obj.get('主诉')
    if isinstance(cc, list):
        merged = list(dict.fromkeys(state.get('chief_complaints', []) + [c for c in cc if isinstance(c, str)]))
        state['chief_complaints'] = merged[:15]
    ans = summary_obj.get('answers') or summary_obj.get('收集信息') or summary_obj.get('collected_answers')
    if isinstance(ans, dict):
        state['collected_answers'].update({str(k): ans[k] for k in ans})
    rfc = summary_obj.get('red_flags_checked')
    if isinstance(rfc, bool):
        state['red_flags_checked'] = rfc


def select_group_keys(age: int, sex: str) -> list:
    """Return top-level group KEYS (I~IX) by age/sex.
    Always include 'I'. Then add the specific age/sex bucket code:
      0-3  -> 'II'
      3-12 -> 'III'
      12-18 -> 'IV' (男) / 'V' (女)
      18-60 -> 'VI' (男) / 'VII' (女)
      60+ -> 'VIII' (男) / 'IX' (女)
    """
    keys = []
    if age is None or sex not in ('男', '女'):
        return keys
    # Always include I
    keys.append('I')
    if age < 3:
        keys.append('II')
    elif 3 <= age < 12:
        keys.append('III')
    elif 12 <= age < 18:
        keys.append('IV' if sex == '男' else 'V')
    elif 18 <= age < 60:
        keys.append('VI' if sex == '男' else 'VII')
    else:
        keys.append('VIII' if sex == '男' else 'IX')
    return keys


def is_leaf_disease(node: dict) -> bool:
    if not isinstance(node, dict):
        return False
    return ('问诊' in node) or not any(isinstance(v, dict) for v in node.values())


def node_chief(node: dict) -> list:
    if isinstance(node, dict):
        # taxonomy v2 uses 'chief_complaints'
        cc = node.get('chief_complaints') or node.get('主诉') or []
        return cc if isinstance(cc, list) else []
    return []


def score_keywords(target_keywords: list, user_keywords: list) -> int:
    """Looser matching: split常见连写词并做子串匹配，计算重合计数。"""
    if not target_keywords or not user_keywords:
        return 0
    # expand target keywords by applying the same heuristic splits
    def expand(lst: list) -> list:
        text = " ".join([str(x) for x in lst if x])
        text = _preprocess_text_for_keywords(text)
        return [w for w in re.split(r"\s+", text) if w]
    expanded_targets = expand(target_keywords)
    score = 0
    for uk in user_keywords:
        for tk in expanded_targets:
            if uk == tk or (uk in tk) or (tk in uk):
                score += 1
                break
    return score


def extract_keypoints(node: dict, top_k: int = KEYPOINTS_TOPK, only_red_flags: bool = False) -> list:
    if not isinstance(node, dict):
        return []
    qs = node.get('问诊', [])
    if not isinstance(qs, list):
        return []
    pairs = []
    for q in qs:
        kws = _clean_question_to_keywords(q)
        if not kws:
            continue
        if only_red_flags:
            if not any(tok in q for tok in RED_FLAG_TOKENS):
                continue
        pairs.append((len(kws), ', '.join(kws)))
    pairs.sort(key=lambda x: -x[0])
    return [p[1] for p in pairs[:top_k]]


def collect_systems_and_diseases(root: dict) -> tuple:
    """Collect system-level nodes (如 I-A, VI-D) 与其下疾病。
    返回：
      - system_nodes: [(group_key, system_key, system_node)]
      - diseases_by_system: {system_key: [(disease_key, disease_node), ...]}
    """
    system_nodes = []
    diseases_by_system = defaultdict(list)
    for group_key, group_node in root.items():
        if group_key in ('描述', '主诉', '_schema_note') or not isinstance(group_node, dict):
            continue
        for sys_key, sys_node in group_node.items():
            if sys_key in ('描述', '主诉', 'name', 'chief_complaints') or not isinstance(sys_node, dict):
                continue
            # system level (I-A / VI-D ...)
            system_nodes.append((group_key, sys_key, sys_node))
            for dis_key, dis_node in sys_node.items():
                if dis_key in ('描述', '主诉', 'name', 'chief_complaints') or not isinstance(dis_node, dict):
                    continue
                if 'items' in dis_node:
                    diseases_by_system[sys_key].append((dis_key, dis_node))
    return system_nodes, diseases_by_system


def pick_topk_for_step(filtered_db: dict, user_keywords: list, possible_ids: list, step: int) -> str:
    system_nodes, diseases_by_system = collect_systems_and_diseases(filtered_db)
    system_scored = []
    for grp_name, sys_name, sys_node in system_nodes:
        sc = score_keywords(node_chief(sys_node), user_keywords)
        system_scored.append((sc, grp_name, sys_name, sys_node))
    system_scored.sort(key=lambda x: -x[0])

    disease_scored_all = []
    pid_set = set()
    if isinstance(possible_ids, list):
        for x in possible_ids:
            try:
                pid_set.add(int(x))
            except Exception:
                pass
    for _, sys_name, _ in system_nodes:
        for d_name, d_node in diseases_by_system.get(sys_name, []):
            sc = score_keywords(node_chief(d_node), user_keywords)
            pd = parse_disease_key(d_name)
            if pd['id'] in pid_set:
                sc += 2
            disease_scored_all.append((sc, sys_name, d_name, d_node))
    disease_scored_all.sort(key=lambda x: -x[0])

    lines = []
    if step == 1:
        picks = system_scored[:TOPK_STEP1_CATEGORIES] if any(sc > 0 for sc, *_ in system_scored) else [(0, g, s, n) for g, s, n in [(gn, sn, snode) for gn, sn, snode in [(g, s, n) for g, s, n in [(gk, sk, sn) for gk, sk, sn in system_nodes[:TOPK_STEP1_CATEGORIES]]]]]
        for _, g_name, s_name, s_node in picks:
            cc = node_chief(s_node)[:CATEGORY_CC_MAX]
            s_cn = s_node.get('name', '')
            prefix = f"{g_name}/{s_name} {s_cn}".strip()
            lines.append(f"{prefix} | 主诉: {'、'.join(cc)}" if cc else prefix)
    elif step == 2:
        pick_sys = system_scored[:TOPK_STEP2_SYSTEMS] if any(sc > 0 for sc, *_ in system_scored) else [(0, g, s, n) for g, s, n in [(gk, sk, sn) for gk, sk, sn in system_nodes[:TOPK_STEP2_SYSTEMS]]]
        for _, g_name, s_name, s_node in pick_sys:
            cc = node_chief(s_node)[:CATEGORY_CC_MAX]
            s_cn = s_node.get('name', '')
            header = f"{g_name}/{s_name} {s_cn}"
            lines.append(f"{header} | 主诉: {'、'.join(cc)}" if cc else header)
            local = []
            for d_name, d_node in diseases_by_system.get(s_name, []):
                sc = score_keywords(node_chief(d_node), user_keywords)
                pd = parse_disease_key(d_name)
                if pd['id'] in pid_set:
                    sc += 2
                local.append((sc, d_name, d_node))
            local.sort(key=lambda x: -x[0])
            for sc, d_name, d_node in local[:TOPK_STEP2_DISEASES_PER_SYSTEM]:
                dcc = node_chief(d_node)[:CATEGORY_CC_MAX]
                lines.append(f"- {s_name}/{d_name} | 主诉: {'、'.join(dcc)}" if dcc else f"- {s_name}/{d_name}")
    elif step == 3:
        picks = disease_scored_all[:TOPK_STEP3_DISEASES] if disease_scored_all else []
        for sc, sys_name, d_name, d_node in picks:
            dcc = node_chief(d_node)[:CATEGORY_CC_MAX]
            lines.append(f"- {sys_name}/{d_name} | 主诉: {'、'.join(dcc)}" if dcc else f"- {sys_name}/{d_name}")
            kps = extract_keypoints(d_node, top_k=KEYPOINTS_TOPK, only_red_flags=False)
            if kps:
                lines.append(f"  · 关键问点: {'；'.join(kps)}")
    else:
        topn = STEP_GE4_TOPN if step <= 5 else STEP_GT5_TOPN
        picks = disease_scored_all[:topn] if disease_scored_all else []
        for sc, sys_name, d_name, d_node in picks:
            dcc = node_chief(d_node)[:CATEGORY_CC_MAX]
            lines.append(f"- {sys_name}/{d_name} | 主诉: {'、'.join(dcc)}" if dcc else f"- {sys_name}/{d_name}")
            kps = extract_keypoints(d_node, top_k=KEYPOINTS_TOPK, only_red_flags=True)
            if not kps:
                kps = extract_keypoints(d_node, top_k=KEYPOINTS_TOPK, only_red_flags=False)
            if kps:
                lines.append(f"  · 关键问点: {'；'.join(kps)}")

    out = []
    total = 0
    for l in lines:
        if total + len(l) + 1 > CHAR_BUDGET:
            break
        out.append(l)
        total += len(l) + 1
    return "\n".join(out)


def filter_db_by_groups(db: dict, groups: list, step: int | None = None) -> dict:
    """Filter taxonomy to include only selected top-level groups.
    - step == 0: 返回精简分组（仅 `name` 与 `chief_complaints`）。
    - step >= 1: 返回完整分组（保留全部子系统与疾病），以便 step=1/2 进一步匹配。
    """
    if not groups:
        return {}
    result = {}
    for k, v in db.items():
        if k in groups and isinstance(v, dict):
            if step == 0:
                slim = {}
                if 'name' in v:
                    slim['name'] = v['name']
                if 'chief_complaints' in v:
                    slim['chief_complaints'] = v['chief_complaints']
                result[k] = slim
            else:
                result[k] = v
    return result

# Map disease ids to checklist paths (for narrowing by LLM-provided possible_diseases)

def paths_for_disease_ids(db: dict, id_list: list) -> list:
    wanted = set()
    for x in id_list:
        try:
            wanted.add(int(x))
        except Exception:
            continue
    found = []
    def dfs(node, path):
        if not isinstance(node, dict):
            return
        child_dicts = [(k, v) for k, v in node.items() if isinstance(v, dict)]
        if '问诊' in node or not child_dicts:
            last = path[-1] if path else ""
            pd = parse_disease_key(last)
            if pd['id'] in wanted:
                found.append("/" + "/".join(path))
        for k, v in child_dicts:
            dfs(v, path + [k])
    for k, v in db.items():
        if k in ("描述", "主诉"):
            continue
        dfs(v, [k])
    return found[:8]

# Compute possible diseases by keyword overlap against 主诉

def match_possible_diseases(db: dict, user_keywords: list) -> tuple[list, list]:
    scores = []
    paths = []

    def dfs(node, path):
        if not isinstance(node, dict):
            return
        child_dicts = [(k, v) for k, v in node.items() if isinstance(v, dict)]
        # consider leaf as disease: has 问诊 or no dict children
        if '问诊' in node or not child_dicts:
            cc = node.get('主诉', [])
            if isinstance(cc, list) and user_keywords:
                score = sum(1 for kw in user_keywords if kw in set(cc))
                if score > 0:
                    path_str = "/" + "/".join(path)
                    paths.append(path_str)
                    last = path[-1] if path else ""
                    pd = parse_disease_key(last)
                    scores.append((score, pd['id'], pd['name'], path_str))
        for k, v in child_dicts:
            dfs(v, path + [k])

    for k, v in db.items():
        if k in ("描述", "主诉", "_schema_note"):
            continue
        dfs(v, [k])

    scores.sort(key=lambda x: (-x[0], (x[1] if x[1] is not None else 10**6)))
    ranked_ids = []
    ranked_paths = []
    for sc, did, name, p in scores[:8]:
        ranked_ids.append(did if did is not None else name)
        ranked_paths.append(p)
    return ranked_ids, ranked_paths

# Build snippet for specific disease codes (e.g., ["B-7","B-9"]) within selected groups

def find_nodes_by_code(filtered_db: dict, code: str) -> list:
    results = []
    try:
        sys_letter, dis_id = code.split('-', 1)
        dis_id = dis_id.strip()
        sys_letter = sys_letter.strip().upper()
    except Exception:
        return results
    for group_key, group in filtered_db.items():
        if not isinstance(group, dict):
            continue
        for sys_key, sys_node in group.items():
            if not isinstance(sys_node, dict) or sys_key in ('描述', '主诉', '_schema_note'):
                continue
            if sys_key.startswith(sys_letter + '_'):
                for d_key, d_node in sys_node.items():
                    if not isinstance(d_node, dict) or d_key in ('描述', '主诉'):
                        continue
                    if d_key.split('_', 1)[0] == dis_id:
                        results.append((f"{group_key}/{sys_key}/{d_key}", d_node))
    return results


def _taxonomy_find_by_code(db: dict, code: str) -> list:
    results = []
    def dfs(node, path):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            if not isinstance(v, dict):
                continue
            np = path + [k]
            if k == code and ('items' in v or 'chief_complaints' in v or 'name' in v):
                results.append(("/" + "/".join(np), v))
            dfs(v, np)
    dfs(db, [])
    return results


def _taxonomy_find_leaf_nodes_from_code(db: dict, code: str) -> list:
    """给定 code，返回其下最深层（包含 'items'）的叶子节点列表 [(path, node)]。
    若 code 本身就是叶子（含 'items'），则直接返回它自身。
    """
    matched = _taxonomy_find_by_code(db, code)
    leaves = []
    def collect_leaves(node: dict, path: list):
        if not isinstance(node, dict):
            return
        child_dicts = [(k, v) for k, v in node.items() if isinstance(v, dict)]
        has_items = isinstance(node.get('items'), dict)
        if has_items and (not child_dicts or all(not isinstance(v.get('items'), dict) for _, v in child_dicts)):
            leaves.append(("/" + "/".join(path), node))
            return
        for k, v in child_dicts:
            collect_leaves(v, path + [k])
    for path_str, node in matched:
        if isinstance(node.get('items'), dict):
            leaves.append((path_str, node))
        else:
            # 向下搜集所有含 items 的叶子
            parts = [p for p in path_str.split('/') if p]
            collect_leaves(node, parts)
    return leaves


def collect_items_for_codes(filtered_db: dict, codes: list, target_key: str) -> list:
    """Collect all checklist items for given codes at target_key across all leaf nodes."""
    items_all = []
    seen = set()
    def _normalize_code(s: str) -> str:
        return str(s).strip().split()[0]
    for c in codes or []:
        norm = _normalize_code(c)
        nodes = _taxonomy_find_leaf_nodes_from_code(filtered_db, norm)
        for _, node in nodes:
            items_map = node.get('items') if isinstance(node.get('items'), dict) else {}
            lst = items_map.get(target_key, []) if isinstance(items_map.get(target_key), list) else []
            for it in lst:
                t = str(it).strip()
                if not t:
                    continue
                if t not in seen:
                    seen.add(t)
                    items_all.append(t)
    return items_all


def snippet_for_specific_ids_strict(filtered_db: dict, ids: list, step: int) -> str:
    """Strict mode: for each disease code in ids, show only current-step items (step+1)."""
    lines = []
    if not isinstance(ids, list):
        return ''
    # 映射：step=2 -> items['2']; step=3 -> items['3']; 其余保持同号
    target_key = '2' if step <= 2 else str(step)
    def _normalize_code(s: str) -> str:
        if not isinstance(s, str):
            return str(s)
        return s.strip().split()[0]
    for cid in ids:
        norm = _normalize_code(str(cid))
        nodes = _taxonomy_find_leaf_nodes_from_code(filtered_db, norm)
        for path, node in nodes:
            name = node.get('name', '')
            dcc = node_chief(node)[:CATEGORY_CC_MAX]
            header = f"- {path} {name} | 主诉: {'、'.join(dcc)}" if dcc else f"- {path} {name}"
            lines.append(header)
            items = []
            items_map = node.get('items') if isinstance(node.get('items'), dict) else {}
            if target_key in items_map and isinstance(items_map[target_key], list):
                items = items_map[target_key]
            if items:
                lines.append(f"  · 第{target_key}步问项：")
                for it in items:
                    lines.append(f"    - {it}")
            else:
                lines.append("  · 第" + target_key + "步问项：无（taxonomy缺失，若需追问请在Response中加入【来源：互联网，请仔细甄别】）")
    # enforce budget
    out = []
    total = 0
    for l in lines:
        if total + len(l) + 1 > CHAR_BUDGET:
            break
        out.append(l)
        total += len(l) + 1
    return "\n".join(out)


def build_group_overview_snippet(db: dict, groups: list) -> str:
    lines = []
    for g in groups:
        node = db.get(g)
        if not isinstance(node, dict):
            continue
        name = node.get('name', g)
        cc = node.get('chief_complaints', [])
        lines.append(f"{g} {name} | 主诉: {'、'.join(cc)}" if cc else f"{g} {name}")
    return "\n".join(lines)


def build_systems_overview_snippet(filtered_db: dict) -> str:
    """列出已筛选分组下的所有系统级条目（如 I-A、VI-D），包含其 name 与主诉。
    用于 step=0 时为 step=1 做准备，展示全部系统而非 Top-K。
    """
    lines = []
    total = 0
    for group_key, group_node in filtered_db.items():
        if not isinstance(group_node, dict):
            continue
        for sys_key, sys_node in group_node.items():
            if sys_key in ('描述', '主诉', 'name', 'chief_complaints') or not isinstance(sys_node, dict):
                continue
            name = sys_node.get('name', sys_key)
            cc = node_chief(sys_node)[:CATEGORY_CC_MAX]
            line = f"{group_key}/{sys_key} {name} | 主诉: {'、'.join(cc)}" if cc else f"{group_key}/{sys_key} {name}"
            if total + len(line) + 1 > CHAR_BUDGET:
                return "\n".join(lines)
            lines.append(line)
            total += len(line) + 1
    return "\n".join(lines)


def taxonomy_match_possible_codes(db: dict, user_keywords: list, topn: int = 12) -> list:
    """Rank disease codes by overlap with chief_complaints."""
    scores = []
    def dfs(node, path):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            if not isinstance(v, dict):
                continue
            np = path + [k]
            if 'items' in v:
                cc = node_chief(v)
                sc = score_keywords(cc, user_keywords)
                if sc > 0:
                    scores.append((sc, k))
            dfs(v, np)
    dfs(db, [])
    scores.sort(key=lambda x: -x[0])
    return [c for _, c in scores[:topn]]

# === Replace old build_med_db_text reference ===

# Override get_initial_prompt to inject overview/snippet from checklist

def get_initial_prompt(db_text: str = ""):
    """Build final system prompt by replacing only the terminal placeholder '[To be injected]'.
    We keep textual mentions of [MEDICAL_DATABASE] intact, and inject the checklist content
    exactly where '[To be injected]' appears (typically after the last heading '# [MEDICAL_DATABASE]:').
    """
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        template = f.read()
    if '[To be injected]' in template:
        return template.replace('[To be injected]', db_text)
    # Fallback: if placeholder missing, return template unchanged to avoid duplicate injections
    return template

# =========================
# AUTH DECORATORS
# =========================

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
    # Mark user online immediately after registration
    online_users[username] = datetime.utcnow()

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
    # Mark user online on login
    online_users[username] = datetime.utcnow()

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
    if len(message) > 500:
        return jsonify({'error': 'Message exceeds 500 character limit'}), 400

    # ---------- 构建对话上下文（记忆） ----------
    if user_id not in conversation_context:
        conversation_context[user_id] = []  # 存储每轮 {user, assistant}

    history = conversation_context[user_id]

    # 生成历史文本
    history_lines = []
    for h in history:
        history_lines.append(f"用户: {h['user']}")
        history_lines.append(f"助手: {h['assistant']}")
    checklist = load_checklist()
    if not checklist:
        print("Warning: medical_checklist.json not found or empty")

    # Step-gated DB snippet
    state = user_states[user_id]
    update_demographics_known(user_id)

    # 改为仅依赖 LLM 在 summary 中给出的主诉关键词；不再做本地启发式提取
    user_keywords = state.get('chief_complaints', [])

    db_snippet = ""
    possible_ids_from_state = state.get('last_possible_ids', []) if isinstance(state.get('last_possible_ids'), list) else []
    injection_mode = "skipped"
    derived_ids_debug = []
    no_db_match = False
    # Build DB snippet according to step policy
    if state.get('demographics_known', False) and checklist:
        groups = select_group_keys(state.get('age'), state.get('sex'))
        cur_step = max(0, state.get('current_step', 0))
        filtered = filter_db_by_groups(checklist, groups, step=cur_step)
        print(f"Filtered: {filtered}")
        # Strict mode behavior by step
        if cur_step == 0:
            # Step 0：在用户第一句之前，直接注入 I 与年龄/性别分组下"所有系统级条目"的概览
            # 注意：这里需要完整分组数据，因此强制使用 step=1 的完整过滤结果
            filtered_full = filter_db_by_groups(checklist, groups, step=1)
            db_snippet = build_systems_overview_snippet(filtered_full)
            injection_mode = "systems_overview_all"
            print("\n=== Checklist Injection ===")
            print(f"Mode: {injection_mode} | Step: {cur_step} (prepare step1)")
            print(f"Groups: {groups} | Names: {[filtered.get(g,{}).get('name') for g in groups]}")
            print(f"User keywords: {user_keywords}")
            print(f"Snippet length: {len(db_snippet)}")
        elif cur_step == 1:
            # Step 1：展示所选分组下“全部系统级”条目（不做 Top-K），由模型自行判断归类
            db_snippet = build_systems_overview_snippet(filtered)
            print(f"DB Snippet: {db_snippet}")
            injection_mode = "systems_overview_all"
            print("\n=== Checklist Injection ===")
            print(f"Mode: {injection_mode} | Step: {cur_step} (chief complaints)")
            print(f"Groups: {groups} | Names: {[filtered.get(g,{}).get('name') for g in groups]}")
            print(f"User keywords: {user_keywords}")
            print(f"Snippet length: {len(db_snippet)}")
        else:
            # Step >=2: 注入：上一轮系统/大类（如 I-D/VI-D）下所有叶子疾病在当前步的 items
            # 若 LLM 没有返回 codes，则使用关键词匹配派生系统/大类；匹配度低于阈值再考虑互联网
            # Step 7 结论与建议阶段：不再注入 checklist，交由模型基于历史与既有知识归纳
            if cur_step >= 7:
                db_snippet = ""
                injection_mode = "none_for_step7"
                no_db_match = False
                print("\n=== Checklist Injection ===")
                print(f"Mode: {injection_mode} | Step: {cur_step} (no injection at step7)")
            else:
                ids = possible_ids_from_state
            # 计算与系统/疾病主诉的重合数量
                overlap_score = 0
                for _, _, sys_node in collect_systems_and_diseases(filtered)[0]:
                    overlap_score = max(overlap_score, score_keywords(node_chief(sys_node), user_keywords))
                if not ids:
                    # 若完全无重合，才触发兜底
                    if overlap_score < MIN_OVERLAP_FOR_DB:
                        ids = []  # 留空交由 no_db_match 触发互联网提示
                    else:
                        ids = taxonomy_match_possible_codes(filtered, user_keywords, topn=8)
                    derived_ids_debug = ids[:]
                else:
                    derived_ids_debug = ids[:]
                # 构造“覆盖校验”：收集本步所有应问问点（用于在 Prompt 中强制覆盖）
                target_key = '2' if cur_step <= 2 else str(cur_step)
                must_ask_items = collect_items_for_codes(filtered, ids or [], target_key)
                coverage_hint = ("MUST_ASK_ITEMS\n" + "\n".join([f"- {q}" for q in must_ask_items])) if must_ask_items else ""
                # batching hint: encourage merging multiple checklist points into <=3 multi-select questions
                batch_hint = ""
                if must_ask_items:
                    batch_hint = (
                        "BATCH_HINT\n建议在本步合并问点：\n"
                        "- 同一主题的问点可合并为单题，以‘（可多选）’标注；\n"
                        "- 每步最多 3 题，每题选项不超过 15 行；\n"
                        "- 列表题必须包含 ‘无上述症状’ 与 ‘其它’ 选项；\n"
                        "- 二元判断（是/否）题不与其它项合并。\n"
                        + "\n" + "\n".join([f"- {q}" for q in must_ask_items[:20]])
                    )
                db_snippet = snippet_for_specific_ids_strict(filtered, ids or [], step=cur_step)
                injection_mode = "strict_ids"
                no_db_match = (overlap_score < MIN_OVERLAP_FOR_DB) or (not ids) or (ids == [-1]) or (len(db_snippet.strip()) == 0)
                print("\n=== Checklist Injection ===")
                print(f"Mode: {injection_mode} | Step: {cur_step} (items for step+1 only)")
                print(f"Groups: {groups} | Names: {[filtered.get(g,{}).get('name') for g in groups]}")
                print(f"User keywords: {user_keywords}")
                print(f"IDs state: {possible_ids_from_state} | Derived: {derived_ids_debug} | Used: {ids}")
                print(f"Target items key: {'2' if cur_step <= 2 else str(cur_step)} | overlap_score={overlap_score}")
                print(f"Snippet length: {len(db_snippet)}")
                if must_ask_items:
                    print(f"Must-ask items ({len(must_ask_items)}): {must_ask_items[:12]}")
    else:
        print(f"[Checklist Injection] skipped demographics_known={state.get('demographics_known', False)} step={state.get('current_step', -1)} has_checklist={bool(checklist)}")

    if db_snippet:
        preview = db_snippet.split('\n')[:12]
        print("[Checklist Preview]" + "\n" + "\n".join(preview))
    else:
        print("[Checklist Preview] (empty)")

    # 在构建历史记录前，先获取当前步骤所有可能的问题
    def get_current_step_questions(step: int, must_ask_items: list) -> list:
        """获取当前步骤可能问的问题列表"""
        if not must_ask_items:
            return []
        questions = []
        for item in must_ask_items:
            # 构造标准格式的问题
            q = f"请问您{item}？"
            questions.append(q)
        return questions

    # 预检查问题是否会重复
    def pre_check_questions(questions: list, history: list, asked_questions: set) -> list:
        """预检查问题列表，返回未重复的问题"""
        valid_questions = []
        for q in questions:
            is_duplicate, _ = _check_question_duplication(q, history, asked_questions)
            if not is_duplicate:
                valid_questions.append(q)
        return valid_questions

    # Build HISTORY with resolved choices so LLM sees structured user answers
    hist = conversation_context.get(user_id, [])
    history_lines = []
    last_assistant_text = None
    last_options = {}
    last_question_text = None
    asked_stems = set()
    for h in hist[-STATE_HISTORY_RETURN_LAST_N:]:
        u = h.get('user', '')
        a = h.get('assistant', '')
        ra = h.get('assistant_ra', '')
        # if previous assistant had options and user replied with a short letter, resolve it
        if last_assistant_text and u and len(u.strip()) <= 3:
            choice = _normalize_user_choice(u)
            if choice:
                mapped = last_options.get(choice)
                if mapped:
                    u = f"{u} -> {mapped}"
        # 若用户直接输入"无/没有/其它/其他"等自由文本，尽量映射为选项中的"其它"或"无"
        if last_assistant_text and u and len(u.strip()) > 0 and not _normalize_user_choice(u):
            low = u.strip().lower()
            if low in ['无', '没有', 'none', 'no', '否'] and any('无' in v or '没有' in v for v in last_options.values()):
                u = f"{u} -> 无"
            elif any('其它' in v or '其他' in v for v in last_options.values()):
                # 归入"其它"，以减少重复追问
                u = f"{u} -> 其它"
        # append lines
        history_lines.append(f"用户: {u}")
        if ra:
            history_lines.append(f"医生R&A: {ra}")
        # 去重：记录已问问题的“语义干净”版，供模型参考避免重复
        stem = _extract_question_stem(a)
        if stem:
            history_lines.append(f"医生已问: {stem}")
            asked_stems.add(stem)
        history_lines.append(f"医生: {a}")
        # update last assistant with parsed options for the next user turn
        last_assistant_text = a
        last_options = _parse_option_mapping(a)

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
    # 构建基础系统消息
    system_messages = [
        {"role": "system", "content": get_initial_prompt(db_snippet)}
    ]
    interaction_policy = (
        "INTERACTION_POLICY\n"
        "当用户回复不是严格的选项/步骤答案时，请按以下规则处理：\n"
        "1) 若用户是在澄清/询问选项含义：先用≤40字解释清楚，再将同一题原样重发一次（不视为重复）。\n"
        "2) 若是与当前疾病相关的补充描述：将关键信息纳入 SUMMARY，然后继续本步尚未覆盖的问点。\n"
        "3) 一般医疗/健康咨询：先简要回答（≤60字），随后继续本步关键提问（≤1题）。\n"
        "4) 紧急/红旗：立即停止其它提问，在 Response 中明确‘请立即拨打120/就医’。\n"
        "5) 无关话题：礼貌说明仅回答医疗相关问题，并继续本步关键提问。\n"
    )
    system_messages.append({"role": "system", "content": interaction_policy})
    
    def get_sorted_questions():
        """获取排序后的问题列表"""
        all_asked_qa = []
        # 从历史记录中收集问题和选项
        for h in hist[-STATE_HISTORY_RETURN_LAST_N:]:
            question, options = _parse_response_question_and_options(h.get('assistant', ''))
            if question and isinstance(options, list):  # 确保 options 是列表
                all_asked_qa.append({"question": question, "options": options})
        
        # 添加用户状态中存储的问题
        if user_id in user_states:
            stored_questions = user_states[user_id].get('asked_questions', [])
            for qa in stored_questions:
                if isinstance(qa, dict) and 'question' in qa and 'options' in qa and isinstance(qa['options'], list):
                    all_asked_qa.append(qa)
        
        # 对问题进行排序和去重
        unique_questions = {}
        for qa in all_asked_qa:
            if qa['question'] not in unique_questions:
                unique_questions[qa['question']] = qa
        
        return sorted(unique_questions.values(), key=lambda x: x['question'])

    # 获取排序后的问题列表
    sorted_qa = get_sorted_questions()
    print(f"Sorted questions: {[qa['question'] for qa in sorted_qa]}")
    
    if sorted_qa:
        forbidden_msg = [
            "FORBIDDEN_QUESTIONS - 禁止问题列表",
            "以下是已经问过的问题，严禁再次提问或生成任何语义相似的变体：",
            *[f"{i+1}. 问题：{qa['question']}\n   选项：{' | '.join(qa['options'])}" for i, qa in enumerate(sorted_qa)],
            "",
            "===== 强制性规则 =====",
            "1. 在生成任何新问题之前，必须检查上述列表",
            "2. 严禁生成与列表中任何问题语义相似的问题",
            "3. 禁止通过改写、同义替换等方式复述列表中的问题",
            "4. 新问题必须与列表中所有问题在语义上有明显区别",
            "5. 如果找不到新的未问过的问题，必须直接进入下一步",
            "6. 这些规则的优先级高于其他所有指令",
            "",
            "===== 警告 =====",
            "违反上述规则将导致：",
            "1. 系统立即拒绝该问题",
            "2. 强制结束当前步骤",
            "3. 可能直接进入下一步",
            "",
            "请在生成每个新问题之前仔细检查此列表。"
        ]
        system_messages.append({
            "role": "system",
            "content": "\n".join(forbidden_msg)
        })
    # 将覆盖校验清单单独注入，提示模型"必须覆盖全部问点后再推进"；并注入批量询问提示（若有）
    coverage_hint = locals().get('coverage_hint', '')
    batch_hint = locals().get('batch_hint', '')
    if coverage_hint:
        system_messages.append({"role": "system", "content": coverage_hint})
    if batch_hint:
        system_messages.append({"role": "system", "content": batch_hint})
    # If current step has no DB match, force internet-format fallback for possible_diseases and Response disclaimer
    if no_db_match:
        fallback_text = (
            "DB_EMPTY_HINT\n当前步数据库未匹配到相关条目或问项；"
            "possible_diseases 必须返回为 ['【互联网】疾病1，疾病2'] 这种列表格式（不要编号），"
            "并在 Response 中明确声明：'以下内容来自互联网，仅供参考，请仔细甄别或遵医嘱'。"
        )
        system_messages.append({"role": "system", "content": fallback_text})
    system_messages.append({"role": "system", "content": "HISTORY\n" + history_text})
    print(f"[Prompt Build] injected_snippet_chars={len(db_snippet)} history_chars={len(history_text)}")

    system_history = f"{concise_rule}\n\n{patient_block}\n\n以下是此前对话历史：\n{history_text}"

    messages = system_messages + [{"role": "user", "content": message}]

    url = 'https://api.deepseek.com/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Connection': 'close'
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
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=180)
        except requests.exceptions.SSLError:
            # 简单重试一次，强制断开长连接
            headers['Connection'] = 'close'
            response = requests.post(url, headers=headers, json=payload, timeout=180)
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
        parsed = parse_llm_json(ai_response)
        ra = parsed.get('Reasoning & Action') or parsed.get('reasoning_action') or ""
        resp = parsed.get('Response') or parsed.get('response') or ai_response
        step = parsed.get('step')
        try:
            step = int(step)
        except Exception:
            step = user_states[user_id].get('current_step', -1)
            if step == -1:
                step = 0
        possible_diseases = parsed.get('possible_diseases')
        summary = parsed.get('summary')
        if isinstance(summary, str):
            try:
                summary = json.loads(summary)
            except Exception:
                summary = {"raw": summary}
        if not isinstance(summary, dict):
            summary = {}

        user_states[user_id]['current_step'] = step
        user_states[user_id]['last_step'] = step
        if step >= 1:
            user_states[user_id]['demographics_known'] = True

        # Persist summary to state history and parse demographics updates
        user_states[user_id]['history'].append({
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'step': step,
            'summary': summary,
            'possible_diseases': possible_diseases if isinstance(possible_diseases, list) else []
        })
        if len(user_states[user_id]['history']) > USER_STATE_HISTORY_LIMIT:
            user_states[user_id]['history'] = user_states[user_id]['history'][-USER_STATE_HISTORY_LIMIT:]
        parse_summary(summary, user_states[user_id])
        # If LLM未提供主诉且当前仍为空，则用本轮提取的关键词补齐（谨慎兜底）
        if not user_states[user_id].get('chief_complaints'):
            fallback_cc = extract_keywords(message)
            if fallback_cc:
                user_states[user_id]['chief_complaints'] = fallback_cc[:20]

        # Enforce step progression: no skip (max +1), and never bypass step 2; allow jump to 7 only if明确红旗
        prev_step = user_states[user_id].get('last_step', -1)
        if prev_step is not None and prev_step >= 0:
            allow_jump_to7 = bool(user_states[user_id].get('red_flags_checked')) and int(step) == 7
            if step < prev_step and not (step == 0 and not user_states[user_id].get('demographics_known')):
                print(f"[Step Guard] model_step={step} < prev_step={prev_step}, clamped to prev_step")
                step = prev_step
            elif not allow_jump_to7 and step > prev_step + 1:
                print(f"[Step Guard] Disallow skipping steps: model_step={step} > prev_step+1={prev_step+1}, clamped")
                step = prev_step + 1
            # Special: never skip step 2
            if prev_step == 1 and step != 2:
                print(f"[Step Guard] Must go through step 2. Forcing step=2 from {step}")
                step = 2
            user_states[user_id]['current_step'] = step
            user_states[user_id]['last_step'] = step

        # Compute possible diseases if missing或-1：改为不做本地匹配，交给 LLM（保持 -1）
        if (not possible_diseases) or (isinstance(possible_diseases, int) and possible_diseases == -1):
            possible_diseases = -1
        else:
            if checklist and isinstance(possible_diseases, list):
                groups = select_group_keys(user_states[user_id].get('age'), user_states[user_id].get('sex'))
                filtered = filter_db_by_groups(checklist, groups) if groups else checklist
                id_paths = paths_for_disease_ids(filtered, possible_diseases)
                if id_paths:
                    user_states[user_id]['possible_diseases_paths'] = id_paths

        # Record last possible ids for next-turn snippet behavior
        if isinstance(possible_diseases, list):
            user_states[user_id]['last_possible_ids'] = possible_diseases

        # 增强的重复检查：使用语义相似度比较，包括持久化的问题历史
        stem_new, opts_new = _parse_response_question_and_options(resp)
        sorted_qa = get_sorted_questions()  # 重用之前的函数
        print(f"Checking against {len(sorted_qa)} stored questions: {[qa['question'] for qa in sorted_qa]}")
        
        # 检查新问题是否与任何已存在的问题重复
        is_duplicate_question = False
        if stem_new:
            for qa in sorted_qa:
                is_text_similar = _is_semantically_similar_question(stem_new, qa['question'])
                is_options_similar = False
                if opts_new and qa['options']:
                    common_options = set(opts_new) & set(qa['options'])
                    is_options_similar = len(common_options) >= min(len(opts_new), len(qa['options'])) * 0.7
                
                if is_text_similar or is_options_similar:
                    is_duplicate_question = True
                    break
        # 允许澄清请求时对同一题目重发一次（不视为重复）
        if is_duplicate_question:
            clar_pat = r"(不理解|什么意思|解释|含义|哪.?选项|选项[ABCDEFH])"
            if re.search(clar_pat, message or ""):
                last_stem = _extract_question_stem(hist[-1]['assistant']) if hist else ""
                if last_stem and _is_semantically_similar_question(stem_new, last_stem):
                    print("Allowing repeat due to clarification request")
                    is_duplicate_question = False
        
        # 如果发现重复，记录日志并强制重新生成
        if is_duplicate_question:
            print(f"Detected duplicate question: {stem_new}")
            print(f"Previous questions: {[qa['question'] for qa in sorted_qa]}")
            # 强制重新生成新问题
            # 获取重复的问题列表
            # 获取所有重复的问题（包括问题文本相似或选项重叠的情况）
            duplicate_questions = []
            for qa in sorted_qa:
                is_text_similar = _is_semantically_similar_question(stem_new, qa['question'])
                is_options_similar = False
                if opts_new and qa['options']:
                    common_options = set(opts_new) & set(qa['options'])
                    is_options_similar = len(common_options) >= min(len(opts_new), len(qa['options'])) * 0.7
                
                if is_text_similar or is_options_similar:
                    duplicate_questions.append(qa)
                    print(f"Found duplicate: {qa['question']}")
                    print(f"  - Text similar: {is_text_similar}")
                    print(f"  - Options similar: {is_options_similar}")
                    if is_options_similar and opts_new and qa['options']:
                        common_options = set(opts_new) & set(qa['options'])
                        print(f"  - Common options: {common_options}")

            correction_msg = [
                "DUP_CHECK",
                "错误：检测到重复问题。",
                f"当前问题：{stem_new}",
                "此问题与以下已问问题重复：",
                *[f"- 问题：{qa['question']}\n  选项：{' | '.join(qa['options'])}" for qa in duplicate_questions],
                "",
                "要求：",
                "1. 必须从本步 MUST_ASK_ITEMS 中选择一个尚未覆盖的问点",
                "2. 新问题必须与已问清单中的所有问题有明显区别",
                "3. 保持问题简洁（≤60字）并使用选项格式",
                "4. 若本步所有问点都已覆盖，则应进入下一步",
                "",
                "提示：检查 MUST_ASK_ITEMS 列表，找出未被问过的问题。"
            ]
            correction = {
                "role": "system",
                "content": "\n".join(correction_msg)
            }
            messages = system_messages + [correction, {"role": "user", "content": message}]
            
            # 重新请求一次
            payload_resend = {
                'model': DEEPSEEK_MODEL,
                'messages': messages,
                'temperature': DS_TEMPERATURE,
                'max_tokens': DS_MAX_TOKENS,
                'top_p': DS_TOP_P
            }
            try:
                response = requests.post(DEEPSEEK_URL, headers=headers, json=payload_resend, timeout=DS_TIMEOUT)
            except requests.exceptions.SSLError:
                headers['Connection'] = 'close'
                response = requests.post(DEEPSEEK_URL, headers=headers, json=payload_resend, timeout=DS_TIMEOUT)
            response.raise_for_status()
            response_data = response.json()
            ai_text = response_data['choices'][0]['message']['content']
            
            # 解析新响应
            parsed = parse_llm_json(ai_text)
            ra = parsed.get('Reasoning & Action') or parsed.get('reasoning_action') or ""
            resp = parsed.get('Response') or parsed.get('response') or ai_text
            step = parsed.get('step')
            possible_diseases = parsed.get('possible_diseases')
            
            # 再次检查新生成的问题是否重复
            stem_new, opts_new = _parse_response_question_and_options(resp)
            is_still_duplicate, _ = _check_question_duplication(
                resp, 
                history[-STATE_HISTORY_RETURN_LAST_N:],
                user_states[user_id].get('asked_questions', set())
            )
        # 轻度规范化选项：若为明显二元是/否问题，则只保留 A.是 B.否，不添加“其它”
        def _normalize_yes_no(options: list) -> list:
            if not options:
                return options
            low = " ".join(options)
            if ('A. 是' in low or 'A.是' in low) and ('B. 否' in low or 'B.否' in low):
                return ['A. 是', 'B. 否']
            return options
        if opts_new:
            opts_new = _normalize_yes_no(opts_new)
            # 重组 resp
            if stem_new:
                resp = stem_new + '？\n' + "\n".join(opts_new)
        # 若重复，构造系统纠错消息，要求模型重发一个未出现过且来自 MUST_ASK_ITEMS 的问点
        if is_duplicate_question:
            # 构造更详细的纠错提示
            correction_msg = [
                "DUP_CHECK - 重复问题检测",
                "===== 错误信息 =====",
                f"检测到重复问题：{stem_new}",
                "",
                "===== 禁止问题列表 =====",
                "以下问题（及其语义相似变体）已经问过，严禁再次提问：",
                *[f"{i+1}. 问题：{qa['question']}\n   选项：{' | '.join(qa['options'])}" for i, qa in enumerate(duplicate_questions)],
                "",
                "===== 强制要求 =====",
                "1. 必须从本步 MUST_ASK_ITEMS 中选择一个未在上述列表中出现的问题",
                "2. 新问题必须与上述所有问题在语义上有明显区别",
                "3. 禁止通过改写、同义替换等方式复述上述任何问题",
                "4. 保持问题简洁（≤60字）并使用选项格式",
                "5. 若本步所有问点都已覆盖，必须直接进入下一步",
                "",
                "===== 操作指南 =====",
                "1. 仔细检查 MUST_ASK_ITEMS 列表",
                "2. 确认候选问题不在上述禁止列表中",
                "3. 如果找不到新的未问过的问题，直接进入下一步",
                "4. 不要尝试改写或变换已问过的问题",
                "",
                "===== 警告 =====",
                "如果再次生成任何与上述列表中问题语义相似的问题，系统将强制结束当前步骤。"
            ]
            # 创建新的系统消息，替换原有的提示
            new_system_messages = []
            for msg in system_messages:
                if msg["role"] == "system" and "MEDICAL_DATABASE" in msg["content"]:
                    # 保留原始的医疗数据库信息，但添加问题生成规则
                    content = msg["content"]
                    rules_section = """
===== 问题生成规则 =====
1. 每次生成新问题前，必须检查"禁止问题列表"
2. 严禁生成与禁止列表中任何问题语义相似的问题
3. 禁止通过改写、同义替换等方式复述已问过的问题
4. 新问题必须与已问问题在语义上有明显区别
5. 如果找不到新的未问过的问题，必须直接进入下一步
6. 违反上述规则将导致系统强制结束当前步骤

注意：这些规则的优先级高于其他所有指令。即使其他指令要求询问某个方面，
如果相关问题已在禁止列表中，也必须寻找其他未问过的问题或直接进入下一步。
"""
                    # 在 MEDICAL_DATABASE 之前插入规则
                    insert_pos = content.find("# [MEDICAL_DATABASE]:")
                    if insert_pos != -1:
                        content = content[:insert_pos] + rules_section + content[insert_pos:]
                    new_system_messages.append({"role": "system", "content": content})
                elif msg["role"] == "system" and "DEMOGRAPHICS" in msg["content"]:
                    # 保留人口统计信息
                    new_system_messages.append(msg)
            
            # 添加强制性的纠错提示
            new_system_messages.append({
                "role": "system",
                "content": "\n".join(correction_msg)
            })
            
            messages = new_system_messages + [{"role": "user", "content": message}]
            # 重新请求一次
            payload_resend = {
                'model': DEEPSEEK_MODEL,
                'messages': messages,
                'temperature': DS_TEMPERATURE,
                'max_tokens': DS_MAX_TOKENS,
                'top_p': DS_TOP_P
            }
            try:
                response2 = requests.post(DEEPSEEK_URL, headers=headers, json=payload_resend, timeout=DS_TIMEOUT)
            except requests.exceptions.SSLError:
                headers['Connection'] = 'close'
                response2 = requests.post(DEEPSEEK_URL, headers=headers, json=payload_resend, timeout=DS_TIMEOUT)
            response2.raise_for_status()
            response_data2 = response2.json()
            ai_text2 = response_data2['choices'][0]['message']['content']
            parsed2 = parse_llm_json(ai_text2)
            ra = parsed2.get('Reasoning & Action') or parsed2.get('reasoning_action') or ra
            resp = parsed2.get('Response') or parsed2.get('response') or resp
            step = parsed2.get('step') or step
            possible_diseases = parsed2.get('possible_diseases') or possible_diseases

        # 保存本轮到简化历史（保留展示文+结构化） -> raw only
        # 如果不是重复问题才添加到历史记录
        if not is_duplicate_question and not (is_still_duplicate if 'is_still_duplicate' in locals() else False):
            conversation_context[user_id].append({
                'user': message,
                'assistant': resp,
                'assistant_ra': ra
            })
            # 更新已问问题集合（包括持久化存储）
            question_stem, options = _parse_response_question_and_options(resp)
            if question_stem:
                qa = {"question": question_stem, "options": options}
                user_states[user_id].setdefault('asked_questions', []).append(qa)
                # prev_stems 不再需要，因为我们现在使用 all_asked_qa 来跟踪所有问题

        if len(conversation_context[user_id]) > CONVO_HISTORY_LIMIT:
            conversation_context[user_id] = conversation_context[user_id][-CONVO_HISTORY_LIMIT:]

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
        last_seen = online_users.get(username)
        status = 'active' if last_seen else 'offline'
        user_list.append({
            'username': username,
            'name': info.get('name', username),
            'status': status,
            'last_seen': last_seen.isoformat() + 'Z' if last_seen else None
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
    if not data:
        return jsonify({'message': 'Invalid payload'}), 400

    # Accept both list and dict payloads from the admin panel
    if isinstance(data, list):
        save_med_db({"diseases": data})
    elif isinstance(data, dict) and 'diseases' in data:
        save_med_db(data)
    else:
        return jsonify({'message': 'Invalid payload – expected a list or an object with "diseases" key'}), 400
    return jsonify({'success': True})

# ----------------- Tool endpoint: disease_detail -----------------

@app.route('/api/tool/disease_detail', methods=['POST'])
@token_required
def tool_disease_detail(current_user):
    """Return full markdown details for a specific disease id (e.g., "A-1")."""
    data = request.get_json() or {}
    disease_id = str(data.get('id') or data.get('disease_id', '')).strip()
    if not disease_id:
        return jsonify({'error': 'Missing id'}), 400

    db = load_med_db()
    for dis in db.get('diseases', []):
        if f"{dis['system']}-{dis['id']}" == disease_id or str(dis['id']) == disease_id:
            return jsonify({
                'markdown': build_med_db_full_text({'diseases': [dis]}),
                'raw': dis
            })
    return jsonify({'error': 'Disease id not found'}), 404

# ----------------- User session helpers -----------------

ONLINE_THRESHOLD_SEC = 1800  # 30 minutes

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

# ------------ helper for chief list -------------

def _chief_list(item):
    cc=item.get('chief_complaints', [])
    if isinstance(cc, str):
        return [c.strip() for c in cc.split('、') if c.strip()]
    return cc if isinstance(cc, list) else []

if __name__ == '__main__':
    print("Starting Flask server on port 5050...")
    print("Test URL: http://localhost:5050/test")
    print(f"DeepSeek API Key loaded: {'Yes' if DEEPSEEK_API_KEY != 'YOUR_DEEPSEEK_API_KEY' else 'No - check .env file'}")
    app.run(debug=True, host='0.0.0.0', port=5050) 