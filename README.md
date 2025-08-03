# MedAssistant-v0

一、项目简介
----------------
MedAssistant-v0 是一个基于 Flask + 前端静态页的智能问诊 Demo，结合
1. ReAct 推理框架
2. 外接结构化医疗知识库 `medical_db.json`
3. DeepSeek LLM 接口
实现分步骤问诊、风险分级（L1-L3）与数据库优先的回答策略。

二、主要目录
--------------
| 路径 | 说明 |
| --- | --- |
| `app.py` | Flask 后端，动态拼装 prompt、JWT 认证、患者历史持久化、Admin API |
| `prompt_react_cn.txt` | ReAct 中文系统提示，已动态占位 `[MEDICAL_DATABASE]` |
| `medical_db.json` | 28 种疾病条目，含 `chief_complaints`、`symptoms` 等字段 |
| `sample_patients.txt` | 50 例覆盖 L1-L3 / corner-case 的测试病例 |
| `test_prompts.txt` | 100 条对话输入脚本（附期望行为），用于回归测试 |
| `admin.html / chat.html` | 前端静态页面 |

三、快速启动
-------------
```bash
# 安装依赖
pip install -r requirements.txt  # flask, flask-cors, bcrypt, python-dotenv, requests

# 设置环境变量
export SECRET_KEY="your_jwt_key"
export DEEPSEEK_API_KEY="your_deepseek_key"

# 运行
python app.py  # 默认 0.0.0.0:5050
```
浏览器访问 `http://localhost:5050`。

四、功能亮点
--------------
1. **动态知识库注入**：`get_initial_prompt()` 仅注入 *id / name / chief_complaints* 概览，提高上下文效率；锁定疾病后可调用 `/api/tool/disease_detail` 获取完整字段。
2. **严格分步骤**：prompt 强制 0→7 流程；Step0/1 为必问；仅 L3 红旗可直接跳到 Step7。
3. **患者历史跨会话**：登录后如果检测到历史会弹窗询问是否复诊；历史注入到 system prompt。
4. **Admin Panel**：可在线编辑知识库、用户、会话记录。
5. **安全策略**：数据库优先、联网搜索需声明；药物/诊断限制；多项提示工程抵御 (Ignore all …) 注入。

五、测试方法
--------------
### 1. 手工测试
- 用 `sample_patients.txt` 复制主诉到聊天框，观察模型按步骤问诊。
- 对照 `test_prompts.txt` 中括号期望，检查红旗处理、步骤遵守、信息补问等。

### 2. 自动脚本示例
```bash
while IFS= read -r line; do
  prompt="${line%%（*}"   # 取括号前文本
  curl -X POST http://localhost:5050/api/chat \
       -H "Authorization: Bearer $TOKEN" \
       -H "Content-Type: application/json" \
       -d "{\"message\": \"$prompt\"}"
done < test_prompts.txt > results.json
```
结合你的评分脚本统计 5 维度得分。

六、常见问题
-------------
Q1. 为什么 prompt 看不到完整症状？
> 仅注入概览以节省 token；模型可通过工具接口按需拉取详情。

Q2. chief_complaints 缺失？
> 请直接在 `medical_db.json` 补全字符串；后端不再自动推断。

Q3. DeepSeek 超时？
> 检查 API Key、网络，或调整 `timeout` 参数。

七、贡献和许可
----------------
欢迎 Fork & PR 进行疾病数据库扩充与前端优化。本项目仅供教学 / Demo，**不用于实际医疗诊断**。 
