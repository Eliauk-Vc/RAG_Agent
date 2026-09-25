# Agent 检索中的 MMR

已在 `KnowledgeTools.search_knowledge` 接入，覆盖基础 Agent 和完整工作流的子任务检索。普通非 Agent 的 RAG 查询接口不经过此工具，未改动其排序行为。

流程：LightRAG 检索及原有余弦重排序 → 扩大候选集合 → MMR 选出 top_k → Agent 原文读取/图谱查询及回答 → 工作流跨子任务 RRF 融合与汇总审核。

公式：`lambda * cosine(query, candidate) - (1-lambda) * max(cosine(candidate, selected))`。首条按问题相关性选取，之后逐条平衡相关性和与已选片段的重复程度；同分保持候选顺序。实现与参数见 `rag_app/mmr.py`、`rag_app/tools.py`。

| 环境变量 | 默认值 | 含义 |
|---|---:|---|
| AGENT_MMR_ENABLED | true | 关闭后恢复原候选数量与相关性排序 |
| AGENT_MMR_LAMBDA | 0.7 | 范围 0–1，越大越偏重问题相关性 |
| AGENT_MMR_FETCH_MULTIPLIER | 3 | 候选数量为 top_k 的倍数，范围 1–5，总数最多 30 |
| AGENT_MMR_TIMEOUT | 8 | 额外嵌入和筛选的总超时秒数，范围 (0,30] |

候选上下文预算随候选数扩大，最多 64000 个项目 tokenizer 单位；仅用于 `aquery_data` 构造候选，最终选取数量和 Agent 写作上下文预算仍受现有限制。候选不足不伪造补齐。先按 chunk_id 去重，传给 Agent 的片段仍以 6000 字节为限；MMR 使用与相关性排序一致的原始候选文本向量。

使用项目配置的嵌入模型分别编码问题与候选原文。2026-09-14 起，相关性排序与 MMR 共享一次检索内的向量缓存：相同原文、相同 query/document 角色只计算一次，MMR 直接读取同一向量。重复文本和并发请求在该检索内合并；异常响应不写入缓存。缓存通过 ContextVar 隔离检索请求，结束时清理，不跨会话复用。LightRAG 对 EmbeddingFunc 的包装副本共享该检索的模型身份。

缓存命中不调用嵌入服务、不额外扣 Token。若自定义重排序器未生成相应向量，则只为缺失文本计算嵌入。底层图谱/向量候选召回仍可能因不同查询文本或角色调用嵌入模型；本次复用针对相关性排序和 MMR，不代表整个 Agent 只调用一次嵌入 API。嵌入失败、维度不匹配、零向量、非有限值和超时均回退到候选原排序。取消操作继续向上传播，整个工具仍受 Agent 工具总超时控制。

工具结果和 Agent trace 的 `selection` 记录 method、status、candidates、selected、lambda，以及 embedding_reused_texts（复用文本次数）、embedding_computed_texts（本轮计算的不同文本与角色数量）；失败只记录异常类型，不写入服务异常正文。RRF 的支持次数和 MMR 分数均不是事实置信度，多样性也不保证召回率提高。

向量复用验证：75 项测试通过。模拟相关性排序 + MMR 共用问题一次、候选批次一次嵌入，包含超长候选（显示截短不重复嵌入）、包装函数身份差异、并发去重、失败重试和请求隔离。真实检索返回 16 个候选，复用问题与候选共 17 份向量，成功选择 6 个片段，记录于 `outputs/qa/embedding_reuse_live.json`。本次运行命中了关键词缓存，不能据总调用数直接推断所有问题的节省比例。

验证：新增 5 项测试覆盖公式与权重边界、同分/数量边界、非法向量、候选扩展、关闭路径、异常降级、超时和取消。全套 66 项测试通过。真实知识检索问题“抑郁发作与躁狂发作的主要症状和病程要求”返回 16 个候选并成功选择 6 个，记录位于 `outputs/qa/mmr_live.json`。此为连通性验证，不是 MMR 效果提升评测；旧的 RRF 评估不代表当前 MMR 版本。

算法来源：[Carbonell 的 MMR 讲义](https://www.cs.cmu.edu/afs/cs.cmu.edu/academic/class/15381-s01/public/www/lec/ir/jgc-ir-handout.pdf)。
