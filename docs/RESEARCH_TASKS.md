# 自动知识任务与资料管理

启动方式不变：`powershell -ExecutionPolicy Bypass -File D:\Pycode\AI_agent_RAG\run.ps1 web`。

## 使用入口

- `/` 知识问答：默认勾选自动优化、拆分与汇总检查。取消勾选可使用基础 Agent；普通 RAG 不经过新工作流。当前请求显示实时阶段和用量。
- `/tasks` 专题任务：填写主题、完整问题和要求，清单可留空。执行时模型优化问题并生成 1–5 个子任务；模板清单只是规划参考。刷新后恢复任务、汇总、自评和执行记录。
- `/manage` 知识库与评估：分页查看文档，导入 UTF-8 TXT/Markdown 或粘贴正文，替换/删除文档，运行固定题目评估，下载操作记录 JSON。

## 执行与恢复

子任务汇总前使用 RRF（Reciprocal Rank Fusion）融合保留的原文证据列表：`score(d) = Σ 1 / (k + rank_i(d))`，排名从 1 开始，默认 `WORKFLOW_RRF_K=60`。输入顺序是各子任务返回的证据顺序（包含 Agent 的选择），不是重新获取全量底层检索排名；不对生成的子答案文本进行排名。一个片段在同一列表内只计一次，不同子任务中的命中分别累加。同分保留首次出现顺序，按分数排序后在证据字节预算内选取。

融合参数、分数和命中列表数保存于 workflow_state.fusion；后续补充检索加入同一融合过程。该过程不增加模型调用。RRF 分数不是可信度，多个子任务也不等于多个独立来源；不能因此证明或过滤事实错误，现有原文检查、模型自评与补查逻辑继续保留。基础 Agent 内部证据合并维持原实现。

保留原始问题 → 优化并拆分 → 每个子任务独立执行 RetrievalAgent → 汇总去重 → 模型检查需求和证据 → 有限补充检索 → 交付或通用知识兜底。

简单问题通常只有一个子任务。模型自评不是独立事实证明。兜底会明确标记“未获知识库证据支持”，状态为 fallback / completed_with_fallback，不被评估计为检索成功。私有资料、未知实时事实不能靠兜底补齐；模型服务本身不可用时，兜底也可能失败。

每个基础 Agent 默认最多 7 次决策、5 次工具执行、3 次检索、120 秒。连续三次重复调用触发循环拦截。不同查询但没有资料依靠次数和轮次限制终止，不声称能识别所有语义重复。

可恢复的子任务异常最多重新运行一次；资料不足交给下一轮补充检索。规划、汇总、自评调用最多尝试两次。规划失败时保留原问题或手工清单继续。文档导入最多尝试两次。程序不会自行修改代码、安装依赖或更换密钥。

| 环境变量 | 默认 | 含义 |
| --- | --- | --- |
| WORKFLOW_MAX_ROUNDS | 2 | 汇总检查轮次，范围 1–3 |
| WORKFLOW_TIMEOUT | 900 | 工作流运行秒数，不含排队；超时后兜底最多再进行两次模型调用 |
| WORKFLOW_TOKEN_BUDGET | 100000 | 已上报 LLM 与嵌入 Token 总预算 |

Token 预算在阶段和子任务开始前检查，不是逐 Token 硬截断；接口在调用结束后返回用量，正在执行的调用或子任务可能超出预算。兜底单独保留有限额度。监控显示供应商报告的 Token，不是费用账单；未报告用量不等于没有消耗。

新自动工作流手动重试会重新规划并检索；若要保留旧执行记录，请创建新版本。旧非自动任务仍支持只重试未完成部分。服务重启会标记执行中任务为中断，需要手动重试，不从模型调用内部恢复。

## 知识库管理

文本最多 10 万字；导入不接受任意本机文件路径。替换先建立新文档索引并验证 processed 状态，再删除旧文档关联索引。失败时记录状态，跨存储更新不是原子事务。操作不删除用户磁盘上的源文件。

变更后清理回答缓存，保留实体提取缓存。历史任务与证据快照不随知识库更新自动重算。请以后台操作完成状态及文档列表为准，提交成功不代表索引成功。

写入、删除、检索和评估串行执行，防止本地存储并发变更。任务保存在 research_tasks 表，操作和评估保存在 management_jobs 表，默认数据库 data/sessions.sqlite3。聊天清空不删除专题任务。继续采用本机单进程、无账号认证部署。

## 质量评估

使用 data/book_eval_dataset.json 固定题目，对比基础 Agent 与完整工作流。运行时关闭 LightRAG 查询缓存，结束或异常后恢复原设置；供应商缓存不受项目控制。

记录答案关键词覆盖、原文关键词覆盖、来源命中、耗时、Token、工具调用、循环拦截和兜底情况。关键词覆盖适合回归检查，不证明全部结论正确。每题失败单独记录，不中断后面的题目。记录可下载 JSON。

## 文档导出

TXT 和 Word 先输出汇总，再附子任务正文。Word 是原生可编辑 DOCX，带标题、段落、列表和章节跳转；没有在线 Word 编辑器，下载后修改不会回写任务。

## API

- POST/GET /research-tasks；workflow:true 开启新流程，question 是完整问题，topics 是规划参考（API 至少一项，网页自动补原问题）。不传 workflow 保持旧接口行为。
- GET/DELETE /research-tasks/{id}；POST /research-tasks/{id}/start、/cancel。
- GET /research-tasks/{id}/export?format=txt|docx。
- POST /chat 可带 workflow:true、monitor_id；GET /activity/{monitor_id} 获取实时状态，内存最多保留 20 条。
- GET/POST /management/documents；DELETE /management/documents/{id}；导入参数 replaces 指定旧文档 ID。
- GET /management/evaluation/cases；POST /management/evaluation，传 case_ids。
- GET /management/jobs 和 /management/jobs/{id} 获取状态与评估结果。
