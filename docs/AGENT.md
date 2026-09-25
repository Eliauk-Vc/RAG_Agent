# Agent 使用与工程说明

## 启动及模式

```powershell
powershell -ExecutionPolicy Bypass -File D:\Pycode\AI_agent_RAG\run.ps1 web
```

打开 http://127.0.0.1:8000/ 。选择“自主检索 Agent”或“普通 RAG”。Ctrl+C 停止；修改代码后重启并刷新网页。启动器固定使用 `D:\Anaconda3\envs\lightrag` 环境。

```powershell
.\run.ps1 chat                               # 终端 Agent
.\run.ps1 chat --session-id <会话ID>          # 恢复会话
.\run.ps1 test                               # 离线测试
.\run.ps1 check                              # 实际索引启动检查，无远程模型调用
.\run.ps1 evaluate --limit 1                 # 真实模型小规模对比
```

终端 `/exit` 停止但保留会话，`/clear` 删除当前会话及执行记录。普通 RAG 终端入口仍为 `python -m rag_app.runtime --skip-insert --interactive`。

## 执行流程

1. 保留原始问题；回顾上一条问题时直接查会话记录。
2. 将近期历史、证据、工具观察结果和剩余预算交给决策模型。
3. Pydantic 校验 JSON 动作，拒绝未知工具、额外字段和越界参数。
4. 调用只读工具并保存执行事件；按文本块 ID 去重证据。
5. 根据缺口补充检索、向用户澄清，或选择证据生成答案。
6. 达到次数限制后，使用已有证据给出受限回答；无证据则说明不足。服务异常与真正没有资料分开处理。

| 工具/动作 | 输入及限制 |
|---|---|
| `search_knowledge` | 问题、模式、1–10 个文本块；支持 naive/local/global/hybrid/mix |
| `read_source` | 合法 `chunk_id`；同一文档向前/后各 0–2 个相邻块；不接受磁盘路径 |
| `query_graph` | 明确实体名称；最大深度 2、最大节点 30；拒绝全图通配符，并读取来源文本 |
| `clarify` | 一个简短澄清问题，结束本轮等待用户 |
| `answer` | 只能选择已经获得的证据 ID，随后统一生成答案 |
| `insufficient` | 说明现有资料不足 |

相同工具和参数在同一轮不重复执行。知识库文本只作为证据，提示词明确不执行资料中的指令；这不等于完整的提示注入防护。证据检查由模型辅助完成，不能保证每句话都准确。当前资料主要是一份书籍，不用于个人诊断、药物处方或独立处理心理危机。

## 上下文与预算

完整消息保存在 SQLite，发送模型时只选近期完整轮次。工具证据去重并限制长度，原始问题超过上限则拒绝，不静默截断。当前未做自动摘要、长期向量记忆或跨用户记忆检索。

在 `.env` 中可配置：

| 变量 | 默认值 | 含义 |
|---|---:|---|
| `AGENT_MAX_SEARCHES` | 3 | 每轮知识搜索次数 |
| `AGENT_MAX_TOOLS` | 5 | 每轮实际工具调用总数 |
| `AGENT_MAX_DECISIONS` | 7 | 决策次数上限，最终回答另计一次 |
| `AGENT_REQUEST_TIMEOUT` | 120 | 整轮超时，秒 |
| `AGENT_TOOL_TIMEOUT` | 40 | 单工具超时，秒 |
| `AGENT_MODEL_TIMEOUT` | 45 | 单次决策/回答超时，秒 |
| `AGENT_HISTORY_TURNS` | 4 | 最近完整对话轮数上限 |
| `AGENT_HISTORY_BYTES` | 6000 | 历史 JSON UTF-8 字节预算 |
| `AGENT_EVIDENCE_BYTES` | 18000 | 去重证据 UTF-8 字节预算 |
| `AGENT_PROMPT_BYTES` | 36000 | Agent 系统提示及输入正文 UTF-8 字节预算 |
| `AGENT_QUESTION_BYTES` | 12000 | 当前问题 UTF-8 字节上限 |

字节预算不等于真实 token 数。LightRAG 沿用原字符分词器，检索预算预留 16000 个字符分词单位，实体、关系分别限制在 1500，避免模板占用挤掉原文。Agent 的最终输入再受字节预算限制。

## 持久化和接口

默认数据库：`data/sessions.sqlite3`，可用 `RAG_SESSION_DB` 修改相对项目根目录的路径。
浏览器标签页的 `sessionStorage` 保存会话 ID；刷新本标签页或重启服务后恢复。关闭标签页后不自动找回 ID，已知 ID 仍可通过 API/终端恢复。
“结束对话”删除本会话原文和执行记录；独立的模型缓存、日志和备份不随之删除。

每轮先写入 `running`，执行中保存工具事件，最后原子提交消息和结果。失败内容不进入后续历史。重启后未完成任务标记为 `interrupted`，保留已落盘事件，不自动重放。

`POST /chat` 示例：

```json
{"question":"要检索的问题","engine":"agent","mode":"hybrid"}
```

后续请求带 `session_id`。`engine` 可为 `rag`；Agent 中 `mode` 是首选建议，工具仍可选择其他模式。
响应包含 `answer/session_id/turns/engine/run_id/stop_reason/committed`。

- `GET /health`：服务及索引准备情况，不探测模型可用性。
- `GET /sessions/{session_id}`：消息、轮数、最近执行记录列表。
- `GET /sessions/{session_id}/runs/{run_id}`：执行详情，必须同时匹配会话与运行 ID。
- `DELETE /sessions/{session_id}`：删除会话及关联记录。

记录包括动作、参数、结果 ID、耗时、停止原因、最终提供给模型的证据片段和用量。不要求或保存模型长篇内部思考。
`prompt_tokens/completion_tokens/total_tokens` 来自 LLM 服务，用量字段不包含 embedding；后者另存 `embedding_tokens`。
`llm_invocations/embedding_invocations` 是进入本地模型封装的次数，不等于服务商内部重试次数。`usage_reported=false` 表示没有收到 LLM 用量统计，不能据此认定调用免费。

## 评估

```powershell
.\run.ps1 evaluate --strategies "agent,hybrid+rerank" --limit 1 --cache-policy cold
.\run.ps1 evaluate --strategies "agent,hybrid+rerank" --cache-policy cold
```

默认对比 Agent 和普通 hybrid 重排序检索。`cold` 绕过 LightRAG 回答/关键词缓存，不删除缓存；`warm` 允许命中。服务商缓存独立存在。
JSON/CSV 位于 `outputs/evaluation`，记录答案关键词覆盖、原文证据覆盖、来源命中、延迟、工具调用、停止原因和 LLM/embedding 用量。JSON 还保存 Agent 的执行事件。
兼容字段 `citation_present` 现在表示内部有文本证据和出处，不再依赖答案里出现 References 标题；它不证明每句话都被证据支持。关键词指标需要结合语义与专业人工评审。

## 模块与部署边界

| 模块 | 职责 |
|---|---|
| `api.py` | 网页、接口、模式切换、恢复与清除 |
| `agent.py` / `agent_types.py` | 决策循环、动作校验、预算、结果 |
| `tools.py` | 知识检索、来源读取、图谱查询 |
| `context.py` | 历史及证据长度控制 |
| `session_store.py` | SQLite 原子写入、执行事件、重启处理 |
| `telemetry.py` | 请求隔离的服务用量统计 |
| `runtime.py` | 现有模型/embedding、普通 RAG 和建库入口 |
| `evaluate.py` | 对比评估及报告 |

LightRAG 队列增加了调用方上下文传播和取消传播，确保连续请求统计隔离，并取消超时后尚未完成的底层协程。取消不能撤销服务商已经发生的工作或费用。

当前是本机单用户、单 worker、串行聊天版本，没有登录和用户权限。随机会话 ID 不等于身份认证，不应直接公开到互联网。不要多个服务进程或终端同时操作同一会话库；正式多用户部署还需认证、并发控制和数据库迁移。
数据默认未加密。备份前停止服务，将原文、完整知识库目录和 SQLite 文件一起备份，并按敏感数据管理。
