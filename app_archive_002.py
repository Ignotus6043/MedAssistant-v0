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

# --- Advanced duplicate detection helpers (additive, non-breaking) ---

def _is_semantically_similar_question(q1: str, q2: str) -> bool:
    """Normalize and compare two question stems to detect semantic duplicates."""
    def normalize(q: str) -> str:
        q = re.sub(r"[？?。!！]$", "", (q or "").strip())
        q = re.sub(r"\s+", " ", q)
        for prefix in COMMON_PREFIXES:
            if q.startswith(prefix):
                q = q[len(prefix):].strip()
        return q.lower()
    return normalize(q1) == normalize(q2)

def _check_question_duplication(new_question: str, history: list, asked_questions: list | None = None) -> bool:
    """Return True if `new_question` duplicates anything in history or asked_questions."""
    if not new_question:
        return False
    existing_qa = []
    for h in history:
        q, opts = _parse_response_question_and_options(h.get('assistant', ''))
        if q:
            existing_qa.append({"question": q, "options": opts})
    if asked_questions:
        for qa in asked_questions:
            if isinstance(qa, dict) and 'question' in qa:
                existing_qa.append({"question": qa['question'], "options": qa.get('options', [])})
    new_q, new_opts = _parse_response_question_and_options(new_question)
    if not new_q:
        return False
    for qa in existing_qa:
        if _is_semantically_similar_question(new_q, qa['question']):
            return True
        if new_opts and qa.get('options'):
            common = set(new_opts) & set(qa['options'])
            if len(common) >= max(1, int(0.7 * min(len(new_opts), len(qa['options'])))):
                return True
    return False

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

# In-memory trackers
conversation_context = {}  # {username: [history]}

# ----------------- Admin credentials -----------------

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

# Persistent chat history -----------------

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

# ----------------- Medical DB helpers -----------------

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

# Keep old detailed builder for disease_detail endpoint

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

# === Checklist helpers (NEW) ===

_CHECKLIST_CACHE = None
_CHECKLIST_MTIME = None

def load_checklist() -> dict:
    """Cached load of taxonomy with mtime check; returns {} on failure."""
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

def save_checklist(data: dict):
    with open(CHECKLIST_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Map chief complaints from medical_db.json into checklist if names match (best-effort)

def _normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ''
    n = name
    n = n.replace('_', '').replace('-', '').replace(' ', '')
    n = n.replace('/', '').replace('、', '')
    return n


def enrich_checklist_from_med_db(checklist: dict, med_db: dict) -> dict:
    if not med_db or not isinstance(med_db, dict):
        return checklist
    diseases = med_db.get('diseases', [])
    # Build map by normalized name
    name_to_cc = {}
    for d in diseases:
        cc = d.get('chief_complaints')
        if isinstance(cc, str):
            cc = [c.strip() for c in cc.split('、') if c.strip()]
        if not isinstance(cc, list):
            continue
        name_to_cc[_normalize_name(d.get('name', ''))] = cc
    if not name_to_cc:
        return checklist

    def traverse(node: dict):
        if not isinstance(node, dict):
            return
        # Try match this node if it looks like a leaf disease entry (has '问诊' or no dict children)
        child_dicts = [(k, v) for k, v in node.items() if isinstance(v, dict)]
        for k, v in child_dicts:
            # Attempt enrich child
            if isinstance(v, dict):
                # leaf?
                grand = [(kk, vv) for kk, vv in v.items() if isinstance(vv, dict)]
                if '问诊' in v or not grand:
                    parsed = parse_disease_key(k)
                    disease_display_name = parsed['name'] if parsed['name'] else k
                    norm = _normalize_name(disease_display_name)
                    cc_from_db = name_to_cc.get(norm)
                    if cc_from_db and (("主诉" not in v) or not v.get('主诉')):
                        v['主诉'] = cc_from_db[:CATEGORY_CC_MAX]
                # Recurse
                traverse(v)
    for topk, topv in checklist.items():
        if topk in ('描述', '主诉', '_schema_note'):
            continue
        traverse(topv)
    return checklist


def _clean_question_to_keywords(q: str) -> list:
    if not isinstance(q, str):
        return []
    q = q.strip()
    q = re.sub(r"[？?。!！]", "", q)
    q = re.sub(r"（.*?）", "", q)
    for p in COMMON_PREFIXES:
        if q.startswith(p):
            q = q[len(p):]
            break
    q = q.replace("（", "").replace("）", "")
    parts = re.split(SPLIT_DELIMS, q)
    parts = [p.strip() for p in parts if 1 < len(p.strip()) <= 8]
    # Deduplicate while keeping order
    seen = set()
    res = []
    for p in parts:
        if p not in seen:
            seen.add(p)
            res.append(p)
    return res[:KEYPOINTS_TOPK + 2]


def ensure_checklist_has_chief_complaints(db: dict) -> dict:
    """Ensure each big category and leaf disease has a '主诉' list; derive from '问诊' if missing."""
    def process_node(node):
        if isinstance(node, dict):
            # If node has '问诊', derive 主诉 if missing
            if '问诊' in node:
                if '主诉' not in node or not isinstance(node['主诉'], list):
                    # derive keywords from first several questions
                    questions = node.get('问诊', [])[:8]
                    keywords = []
                    for q in questions:
                        keywords.extend(_clean_question_to_keywords(q))
                    # cap and unique
                    seen = set()
                    cc = []
                    for k in keywords:
                        if k not in seen:
                            seen.add(k)
                            cc.append(k)
                    node['主诉'] = cc[:CATEGORY_CC_MAX]
            # Recursively process children and build category-level 主诉 as union of children
            child_keys = [k for k in node.keys() if isinstance(node[k], dict)]
            aggregate = []
            for ck in child_keys:
                process_node(node[ck])
                child_cc = node[ck].get('主诉', []) if isinstance(node[ck], dict) else []
                aggregate.extend(child_cc)
            if child_keys:
                # set category 主诉 if not present
                if '主诉' not in node or not isinstance(node['主诉'], list) or not node['主诉']:
                    cnt = Counter(aggregate)
                    node['主诉'] = [w for w, _ in cnt.most_common(CATEGORY_CC_MAX + 2)][:CATEGORY_CC_MAX]
        return node

    return process_node(db)


def build_checklist_snippet_for_step(db: dict, user_state: dict) -> str:
    """Legacy snippet builder (kept for compatibility, not used for Top-K segmentation)."""
    step = user_state.get('current_step', -1)
    demographics_known = user_state.get('demographics_known', False)
    if not demographics_known or step <= 0:
        return ""

    candidate_paths = user_state.get('possible_diseases_paths', [])

    lines = []
    def path_to_str(path_list):
        return "/".join(path_list)

    def add_category(name, node, path):
        cc = node.get('主诉', []) if isinstance(node, dict) else []
        if cc:
            lines.append(f"{path_to_str(path+[name])} | 主诉: {'、'.join(cc[:CATEGORY_CC_MAX])}")
        else:
            lines.append(f"{path_to_str(path+[name])}")

    def add_disease(name, node, path):
        cc = node.get('主诉', [])
        if cc:
            lines.append(f"- {path_to_str(path+[name])} | 主诉: {'、'.join(cc[:CATEGORY_CC_MAX])}")
        else:
            lines.append(f"- {path_to_str(path+[name])}")
        if step >= 4 and '问诊' in node:
            qk = [", ".join(_clean_question_to_keywords(q)) for q in node.get('问诊', [])[:6]]
            qk = [s for s in qk if s]
            if qk:
                lines.append(f"  · 关键问点: {'；'.join(qk[:KEYPOINTS_TOPK])}")

    def traverse(node, path):
        if step == 1:
            for k, v in node.items():
                if k in ("描述", "主诉"):
                    continue
                if isinstance(v, dict):
                    add_category(k, v, path)
        else:
            for k, v in node.items():
                if k in ("描述", "主诉"):
                    continue
                if isinstance(v, dict):
                    current_path = path + [k]
                    is_candidate_branch = any(cp.startswith("/"+"/".join(current_path)+"/") or cp.startswith("/"+"/".join(current_path)) for cp in candidate_paths)
                    if step in (2,3) and not is_candidate_branch and candidate_paths:
                        return
                    has_child_dict = any(isinstance(v2, dict) for v2 in v.values())
                    if has_child_dict:
                        add_category(k, v, path)
                        traverse(v, current_path)
                    else:
                        add_disease(k, v, path)

    traverse(db, [])
    out = []
    total = 0
    for l in lines:
        if total + len(l) + 1 > CHAR_BUDGET:
            break
        out.append(l)
        total += len(l) + 1
    return "\n".join(out)

# =========================
# USERS & AUTH HELPERS
# =========================

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

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
    "asked_questions": []  # track normalized asked questions to suppress duplicates
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
    return jwt.encode({
        'username': username,
        'is_admin': is_admin,
        'exp': datetime.utcnow() + timedelta(days=7)
    }, SECRET_KEY, algorithm='HS256')


def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
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

# =========================
# ROUTES
# =========================

@app.route('/')
def index():
    return send_from_directory('.', 'login.html')

@app.route('/<path:filename>')
def static_files(filename):
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

    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    users[username] = {
        'username': username,
        'password': hashed_pw,
        'name': name
    }

    save_users(users)
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
    
    if email == ADMIN_EMAIL and ADMIN_PW_HASH and bcrypt.checkpw(password.encode('utf-8'), ADMIN_PW_HASH.encode('utf-8')):
        token = generate_token("admin", is_admin=True)
        return jsonify({"success": True, "message": "Login successful", "token": token})
    else:
        return jsonify({"success": False, "message": "Invalid credentials"}), 401

@app.route('/api/chat', methods=['OPTIONS'])
def chat_preflight():
    return '', 200

@app.route('/api/chat', methods=['POST'])
@token_required
def chat(current_user):
    user_id = current_user['username']

    online_users[user_id] = datetime.utcnow()

    print(f"Received {request.method} request to /api/chat")
    print(f"Origin: {request.headers.get('Origin', 'None')}")

    if request.method == 'OPTIONS':
        print("Handling OPTIONS preflight request")
        return '', 200

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

    if user_id not in conversation_context:
        conversation_context[user_id] = []

    history = conversation_context[user_id]

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

    # 在 HISTORY 末尾补充已问清单，强提示模型禁止重复
    if asked_stems:
        history_lines.append("\n已问清单(禁止重复):")
        for s in list(asked_stems)[:30]:
            history_lines.append(f"- {s}")
    history_text = "\n".join(history_lines)

    # Inject demographics hint and step policy
    demo_msg = None
    if state.get('demographics_known') and state.get('age') is not None and state.get('sex') in ('男', '女'):
        demo_text = f"DEMOGRAPHICS\n年龄: {state.get('age')}\n性别: {state.get('sex')}"
        demo_msg = {"role": "system", "content": demo_text}

    system_messages = [
        {"role": "system", "content": get_initial_prompt(db_snippet)}
    ]
    # 将覆盖校验清单单独注入，提示模型“必须覆盖全部问点后再推进”；并注入批量询问提示（若有）
    try:
        if coverage_hint:
            system_messages.append({"role": "system", "content": coverage_hint})
        if 'batch_hint' in locals() and batch_hint:
            system_messages.append({"role": "system", "content": batch_hint})
    except NameError:
        pass
    if demo_msg:
        system_messages.append(demo_msg)
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

    messages = system_messages + [
        {"role": "user", "content": message}
    ]
    print("\n===== HISTORY (Resolved) =====\n" + history_text)

    headers = {
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = {
        'model': DEEPSEEK_MODEL,
        'messages': messages,
        'temperature': DS_TEMPERATURE,
        'max_tokens': DS_MAX_TOKENS,
        'top_p': DS_TOP_P
    }

    print("===== Payload 发往 DeepSeek =====")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print("===== End Payload =====")

    try:
        print("Calling DeepSeek API...")
        response = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=DS_TIMEOUT)
        print(f"DeepSeek API Status Code: {response.status_code}")
        print(f"DeepSeek API Response Headers: {dict(response.headers)}")

        response.raise_for_status()

        response_data = response.json()
        print(f"DeepSeek API Response: {response_data}")

        ai_text = response_data['choices'][0]['message']['content']
        print("DeepSeek responded successfully")

        parsed = parse_llm_json(ai_text)
        ra = parsed.get('Reasoning & Action') or parsed.get('reasoning_action') or ""
        resp = parsed.get('Response') or parsed.get('response') or ai_text
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

        # 自查重复：若本轮问题（stem）已出现在已问清单中，则打回改写（避免重复/过度提问）
        stem_new, opts_new = _parse_response_question_and_options(resp)
        prev_stems = set()
        for h in history[-STATE_HISTORY_RETURN_LAST_N:]:
            s = _extract_question_stem(h.get('assistant', ''))
            if s:
                prev_stems.add(s)
        is_duplicate_question = bool(stem_new) and (stem_new in prev_stems)
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
            correction = {
                "role": "system",
                "content": (
                    "DUP_CHECK\n当前问题与已问清单重复，请改为本步 MUST_ASK_ITEMS 中尚未覆盖的问点之一；"
                    "禁止重复，并保持≤60字与选项格式。"
                )
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
        conversation_context[user_id].append({
            'user': message,
            'assistant': resp,
            'assistant_ra': ra
        })
        if len(conversation_context[user_id]) > CONVO_HISTORY_LIMIT:
            conversation_context[user_id] = conversation_context[user_id][-CONVO_HISTORY_LIMIT:]

        chat_record = {
            'userId': user_id,
            'userMessage': message,
            'assistantMessage': resp,
            'reasoningAction': ra,
            'step': step,
            'possible_diseases': possible_diseases,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        chat_records.append(chat_record)
        save_chat_records()

        patient_histories.setdefault(user_id, []).append({
            'user': message,
            'assistant': resp,
            'reasoningAction': ra,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
        save_patient_histories(patient_histories)

        online_users[user_id] = datetime.utcnow()

        return jsonify({
            'reasoningAction': ra,
            'reasoning_action': ra,
            'reasoning': ra,
            'Reasoning & Action': ra,
            'response': resp,
            'step': step,
            'possible_diseases': possible_diseases,
            'summary': summary,
            'debug': {
                'groups': select_group_keys(state.get('age'), state.get('sex')) if state.get('demographics_known') and checklist else [],
                'group_names': [checklist.get(g,{}).get('name') for g in select_group_keys(state.get('age'), state.get('sex'))] if state.get('demographics_known') and checklist else [],
                'current_step': state.get('current_step', -1),
                'injection_mode': injection_mode,
                'user_keywords': user_keywords,
                'ids_state': user_states[user_id].get('last_possible_ids', []),
                'snippet_len': len(db_snippet),
                'snippet_preview_first_lines': db_snippet.split('\n')[:12],
                'no_db_match': no_db_match
            }
        })
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
    
    patient_histories[username] = []
    save_patient_histories(patient_histories)
    
    if username in conversation_context:
        del conversation_context[username]
    
    global chat_records
    chat_records = [r for r in chat_records if r['userId'] != username]
    save_chat_records()
    
    if username in user_states:
        user_states[username] = {"current_step": -1, "demographics_known": False, "possible_diseases_paths": [], "age": None, "sex": None, "chief_complaints": [], "collected_answers": {}, "red_flags_checked": False, "history": []}
    
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
    if user_id in user_states:
        user_states[user_id] = {"current_step": -1, "demographics_known": False, "possible_diseases_paths": [], "age": None, "sex": None, "chief_complaints": [], "collected_answers": {}, "red_flags_checked": False, "history": []}
    return jsonify({'success': True})

# ---------- User demographics/state endpoints ----------

@app.route('/api/user/state', methods=['GET'])
@token_required
def get_user_state(current_user):
    user_id = current_user['username']
    st = user_states[user_id]
    return jsonify({
        'age': st.get('age'),
        'sex': st.get('sex'),
        'demographics_known': st.get('demographics_known', False),
        'current_step': st.get('current_step', -1)
    })

@app.route('/api/user/demographics', methods=['POST'])
@token_required
def set_user_demographics(current_user):
    user_id = current_user['username']
    data = request.get_json() or {}
    age = data.get('age')
    sex = data.get('sex')
    try:
        age = int(age) if age is not None else None
    except Exception:
        return jsonify({'success': False, 'message': 'Invalid age'}), 400
    if sex not in ('男', '女'):
        return jsonify({'success': False, 'message': 'Invalid sex'}), 400
    st = user_states[user_id]
    st['age'] = age
    st['sex'] = sex
    st['demographics_known'] = True
    # Reset workflow to step=0 when demographics entered
    st['current_step'] = 0
    st['last_step'] = st['current_step']
    return jsonify({'success': True, 'age': age, 'sex': sex, 'current_step': st['current_step']})

# ----------------- Admin endpoints for medical db -----------------

@app.route('/api/admin/medical_db', methods=['GET', 'POST'])
@admin_required
def admin_medical_db():
    if request.method == 'GET':
        return jsonify(load_med_db())
    data = request.get_json()
    if not data:
        return jsonify({'message': 'Invalid payload'}), 400

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

def cleanup_online_users():
    stale = [u for u, ts in online_users.items() if (datetime.utcnow() - ts).total_seconds() > ONLINE_THRESHOLD_SEC]
    for u in stale:
        del online_users[u]

@app.route('/api/logout', methods=['POST'])
@token_required
def logout(current_user):
    username = current_user['username']
    online_users.pop(username, None)
    save_patient_histories(patient_histories)
    return jsonify({'success': True})

# ------------ helper for chief list -------------

def _chief_list(item):
    cc=item.get('chief_complaints', [])
    if isinstance(cc, str):
        return [c.strip() for c in cc.split('、') if c.strip()]
    return cc if isinstance(cc, list) else []

if __name__ == '__main__':
    print(f"Starting Flask server on port {SERVER_PORT}...")
    print(f"Test URL: http://localhost:{SERVER_PORT}/test")
    print(f"DeepSeek API Key loaded: {'Yes' if DEEPSEEK_API_KEY != 'YOUR_DEEPSEEK_API_KEY' else 'No - check .env file'}")
    app.run(debug=SERVER_DEBUG, host=SERVER_HOST, port=SERVER_PORT) 