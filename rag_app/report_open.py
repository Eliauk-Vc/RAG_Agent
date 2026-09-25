"""Render the saved ablation measurements, without inventing improvement claims."""
import json,statistics
from pathlib import Path
from rag_app.evaluate_open import OUT,CASES,VARIANTS

names={'A_base':'A 基础（无附加重排序/MMR/RRF）','B_cosine':'B 余弦重排序',
 'C_cosine_rrf':'C 余弦 + RRF','D_cosine_mmr':'D 余弦 + MMR（复用）',
 'E_full_no_reuse':'E 余弦 + MMR + RRF（不复用）','F_full_reuse':'F 当前完整配置（复用）'}
summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
pct=lambda v:f'{v*100:.1f}%'
text=['# 开放性问题算法与成本对照','',
'日期：2026-09-14。4 道开放题 × 6 种配置 = 24 次真实完整工作流，每个组合一次。另做相同输入下向量复用的 3 组配对测量。',
'', '## 答案和检索结果','',
'| 配置 | 直接答案片段召回率 | 直接答案片段精确率 | 要点完整正确率（模型辅助评分） | 工作流完整完成率 | 部分完成率 | 通用兜底率 |',
'|---|---:|---:|---:|---:|---:|---:|']
for r in summary:
 text.append('| '+names[r['variant']]+' | '+' | '.join(pct(r[k]) for k in ['recall','precision','criterion_accuracy','complete_rate','partial_rate','fallback_rate'])+' |')
text+=['','## 每题平均消耗与耗时','','| 配置 | LLM tokens | 嵌入 tokens | 总耗时（秒） | 工具调用 |','|---|---:|---:|---:|---:|']
for r in summary:
 text.append(f"| {names[r['variant']]} | {r['llm_tokens']:,.0f} | {r['embedding_tokens']:,.0f} | {r['elapsed_seconds']:.1f} | {r['tool_calls']:.2f} |")
text+=['','## 相同输入的向量复用测量','']
bench=json.loads((OUT/'reuse_microbenchmark.json').read_text(encoding='utf-8'))
groups={reuse:[r for r in bench['runs'] if r['reuse']==reuse] for reuse in [False,True]}
text+=['只测“相关性重排序 + MMR”，排除规划、初始召回及答案生成。候选文本固定，前后各测 3 次并交替顺序。',
'','| 配置 | 平均嵌入 tokens | 平均嵌入 API 调用次数 | 耗时中位数（秒） |','|---|---:|---:|---:|']
for reuse,rows in groups.items():
 text.append(f"| {'复用' if reuse else '不复用'} | {statistics.mean(r['usage']['embedding_tokens'] for r in rows):,.0f} | {statistics.mean(r['usage']['embedding_invocations'] for r in rows):.1f} | {statistics.median(r['elapsed_seconds'] for r in rows):.2f} |")
text+=['','所有配对测量的 MMR 最终片段 ID 和顺序一致。这里只能说明同输入下的向量复用效果，不能把阶段节省比例直接套到整个工作流。',
'','## 逐题结果','','| 问题 | 配置 | 命中/标注片段 | 答案要点通过/5 | 状态 | 耗时（秒） |','|---|---|---:|---:|---|---:|']
for case in CASES:
 for variant,*_ in VARIANTS:
  r=json.loads((OUT/f'{case["id"]}_{variant}.json').read_text(encoding='utf-8'))
  j=json.loads((OUT/f'{case["id"]}_{variant}.judge.json').read_text(encoding='utf-8'))
  text.append(f"| {case['id']} | {variant} | {r['metrics']['hits']}/{len(case['gold'])} | {sum(x['pass'] for x in j['criteria'])}/5 | {r['result']['stop_reason']} | {r['elapsed_seconds']:.1f} |")
text+=['','## 本轮结果解读','',
'- 六组最终证据召回率均为 100%，不能据此宣称算法提高了召回。四题每题仅标注 1～3 个直接答案片段，当前测试对漏召回的区分能力有限。',
'- C（余弦 + RRF）的直接答案片段精确率最高，为 48.3%，比 B 的 42.5% 高 5.8 个百分点；要点通过率由 85% 到 100%。这只是本轮观测，规划和生成不固定，不能将全部差异归因于 RRF。',
'- A 和 C 的要点通过率均为 100%，F 为 95%。该指标只检查预定义的二十个要点，不表示整篇答案完全正确，也不代表全部开放问题准确率。额外断言的自动判定存在误报，未作为人工验证的错误率使用。',
'- 当前完整配置 F 相比不复用的 E，每题平均嵌入用量从 100,246.5 降至 59,839.25 tokens（约降低 40.3%），但耗时从 41.6 增至 47.1 秒。工作流拆分和模型调用波动使总耗时不能直接归因于复用；固定输入的配对测量才用于验证复用本身。',
'- MMR 在本轮没有显示稳定的答案质量优势。它引入多样性并扩展候选池，可能保留更多背景片段；不能把多样性分数当成事实正确性分数，也不能认定堆叠算法必然更好。',
'- 24 次中仅 5 次完整完成，19 次为部分完成，0 次通用知识兜底。保留原文证据使部分回答仍可覆盖要点，但不意味着任务完成；应优先检查预算预估、上下文长度及最终审核成本。',
'- 下一轮应扩大独立题集、完善相关片段人工标注，并对同题多次运行；在固定候选下单独测试 MMR，再评估候选扩展，避免把多个变化混为一项算法收益。',
'','## 口径与限制','',
'- 召回率和精确率按最终保留的证据计算，并取四题宏平均。标注只覆盖直接回答要点的原文段落，不把所有背景资料算作相关；不能当作整个知识库所有相关段落的完备标注。',
'- 每题预先定义五个必要要点，共二十个；要点完整且与原文一致才计通过。不是关键词覆盖率，也不是临床正确率。额外断言的支持情况另保存在 judge.json 中。',
'- 评分模型与项目使用同一模型，评分时隐藏算法配置名称并提供原文，但仍存在同模型评审偏差；未经独立人工逐项复核。评分 Token 和时间不算项目运行消耗。',
'- 六组都保留 LightRAG、图谱/向量检索、Agent 规划与工具循环、汇总审核、当前部分结果处理。A 并非“不使用任何算法”，而是关闭附加的余弦重排序、MMR 和 RRF。',
'- MMR 组沿用当前三倍候选扩展及对应候选预算；因此其与非 MMR 组的差别包含候选扩展，不是只替换 MMR 公式。RRF 关闭时使用原轮流合并，为兼容 trace 添加空分数字段。',
'- 使用相同模型、索引副本、无会话历史；关闭 LLM/关键词缓存读取。LLM 预算 50000，嵌入预算 250000，最多两轮；测试总超时统一为180秒，其他使用项目当前设置。服务预估预算包含收尾预留，实际用量未达到上限也可能停止。',
'- 运行顺序按题轮换；真实模型会产生不同拆分和工具调用，网络也有波动。每组合只测一次，不能声称因果提升或统计显著性；未人为挑选最好的结果。',
'- 复用成本微基准使用实际返回证据的固定候选集合，连续重复测量；生产旧版与新版的历史时延不能直接比较。本次是按当前代码关闭对应功能模拟对照，未完整回滚旧版本。',
'- 评估在独立索引副本和独立进程完成，不修改在线服务配置、不重写历史会话。setup_run_A.excluded.json 是调试阶段记录，不计入24次结果。',
'','## 测试问题','']
text += [f"- {c['id']}：{c['question']}" for c in CASES]
text+=['','## 复现与原始记录','',
'使用 lightrag 环境依次运行 `python -m rag_app.evaluate_open`、`python -m rag_app.score_open`、`python -m rag_app.bench_reuse`、`python -m rag_app.report_open`。已有逐题输出会复用，要重新采样须修改 OUT 为新的目录。',
'','原始结果位于 `outputs/evaluation/open_ablation_20260914/`，包含 dataset.json、逐题回答与 trace、逐题 judge.json、summary.json 和 reuse_microbenchmark.json。']
target=Path(__file__).resolve().parents[1]/'docs/OPEN_ABLATION_REPORT.md'
target.write_text('\n'.join(text)+'\n',encoding='utf-8')
print(target)
