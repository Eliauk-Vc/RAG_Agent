# 本地知识检索 Agent

现支持问题优化与自动拆分、子任务 Agent 检索、汇总自评、有限异常恢复和带标记的模型知识兜底。聊天页与专题工作台可查看 Token 和工具调用；`/manage` 提供知识库管理和固定题目质量评估。配置及边界见 [自动任务与管理说明](docs/RESEARCH_TASKS.md)。

新增[专题资料工作台](docs/RESEARCH_TASKS.md)：启动后访问 `/tasks`，编辑任务清单、查看执行进度、取消与重试、创建新版本并导出 TXT 资料。

此目录从现有 LightRAG 项目精选迁移，保留可运行的检索、网页问答、多轮记忆和评估。
现已加入工具选择、多次检索、证据整理、执行预算及 SQLite 会话持久化。网页默认使用 Agent，可切换普通 RAG。详细说明见 [Agent 使用与工程说明](docs/AGENT.md)。

## 启动

使用现有环境 `D:\Anaconda3\envs\lightrag`，不用复制环境或重新安装 LightRAG。
启动器会先切换到本项目目录，优先导入本目录中的 `lightrag`，不会使用旧项目源码。

```powershell
cd D:\Pycode\AI_agent_RAG
powershell -ExecutionPolicy Bypass -File .\run.ps1 web
```

打开 http://127.0.0.1:8000/ 即可聊天，接口文档位于 `/docs`。Ctrl+C 停止服务。
可从任意目录使用启动器的绝对路径启动；需要其他端口时添加 `-Port 8001`。

```powershell
.\run.ps1 chat                         # 终端 Agent，输出可恢复的会话 ID
.\run.ps1 evaluate --limit 1           # 对比 Agent 与普通 RAG，会调用真实服务
.\run.ps1 test                         # 离线回归测试，不调用模型
.\run.ps1 check                        # 真实索引及网页启动检查，不调用模型
```

如系统限制直接运行 PowerShell 脚本，统一使用上面的 `powershell -ExecutionPolicy Bypass -File` 形式。
Python 命令也可直接在本目录运行：

```powershell
& 'D:\Anaconda3\envs\lightrag\python.exe' -m uvicorn rag_app.api:app --host 127.0.0.1 --port 8000 --workers 1
```

## 保留内容

- `rag_app/api.py`：原网页、HTTP 接口、会话隔离和记忆清除。
- `rag_app/runtime.py`：模型配置、索引写入、问题改写、重排序和命令行入口。
- `rag_app/config.py`：统一的文档、索引、工作空间和输出路径。
- `rag_app/evaluate.py`、`data/book_eval_dataset.json`：现有检索评估基线。
- `lightrag/`：核心源码、OpenAI 兼容绑定和当前使用的四种本地存储。
- 本地运行使用 `data/book.txt` 和 `rag_storage/book_project_full/demo/`；原文、索引、会话数据库及生成文件不上传到 GitHub。克隆后需自行提供有权使用的原文并建立索引。
- `tests/`：上述自定义功能的离线测试；`LICENSE`：上游授权。
- `.env`：复制的本机模型配置；不要提交或公开其中的密钥。

未复制其他 demo、React 管理端、部署脚本、无关数据集、旧评估结果、日志、Git 历史或虚拟环境。
当前网页由 `api.py` 直接提供，不需要 Bun 或 Node。
仅包含 JSON KV、JSON 文档状态、NetworkX 图、NanoVectorDB 向量存储及 OpenAI 兼容模型绑定；扩展其他后端时再补充适配器和依赖。

## 路径及索引

`.env` 中的 `RAG_BOOK_PATH`、`RAG_INDEX_DIR`、`RAG_OUTPUT_DIR` 都相对本项目根目录解析。
三个入口统一使用 `rag_storage/book_project_full` 和工作空间 `demo`。
已有索引的文档路径、图属性、向量元数据及缓存引用已改为新目录的 `data/book.txt`；向量和实体 ID 保持不变。
默认使用远程 embedding 和 hybrid 检索；保持原索引的模型和维度配置，当前为 BAAI/bge-m3、1024 维。
如更换 embedding 模型，应指定新的索引目录重新构建，即使新模型维度相同也不能混用已有向量。

重新构建到独立目录（会调用模型，执行后再将 `.env` 的 `RAG_INDEX_DIR` 指向该目录）：

```powershell
.\run.ps1 index --working-dir rag_storage/rebuilt
```

服务保持单 worker。聊天及执行记录存储在 `data/sessions.sqlite3`，重启后保留；刷新同一浏览器标签页可以恢复。“结束对话”删除该会话及执行记录，磁盘模型缓存独立保留。

## Agent 入口

`rag_app/agent.py` 控制决策与停止，`rag_app/tools.py` 封装只读知识工具，`rag_app/context.py` 控制输入长度，`rag_app/session_store.py` 持久化消息和执行记录。
完整原文保存在本地，模型默认只接收最近 4 轮且在预算内的历史；目前没有自动摘要或长期向量记忆。
普通 RAG 的原命令仍可运行：`python -m rag_app.runtime --skip-insert --interactive`。
这是单用户本地工程版本，未包含公网部署所需的登录与权限系统。
