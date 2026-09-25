"""Paired fusion ablation with frozen Agent evidence and source-based labels.

Run: python -m rag_app.evaluate_rrf
This is a small source-consistency benchmark, not a clinical validation.
"""
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from rag_app.api import build_rag
from rag_app.agent import RetrievalAgent
from rag_app.context import merge_evidence, rrf_evidence, bounded_payload
from rag_app.lifecycle import finalize_rag
from rag_app.telemetry import ACTIVE_USAGE, RequestUsage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/evaluation/rrf_ablation'
# Direct answer sections checked against the indexed book, excluding its TOC.
GOLD = [
    'chunk-a86535a2753302c895e7ba23a4cda01e',
    'chunk-a86535a2753302c895e7ba23a4cda01e',
    'chunk-ed1a6c32057e0d7f5dff1705d2b43a00',
    'chunk-8e37b9f67b680ed6b3aef53d722181d2',
    'chunk-2b7c87b39bcca1e85fd0c70351b936d5',
    'chunk-2b7c87b39bcca1e85fd0c70351b936d5',
    'chunk-e5c6faa6cb860a66daa1011dcbb0eb99',
]
GROUPS = [[2, 3], [4, 5, 6], [0, 1], [0, 2, 3, 4, 5]]

def metrics(rows, gold):
    ids = {r['chunk_id'] for r in rows}
    hits = len(ids & gold)
    return {'recall': hits / len(gold), 'precision': hits / len(ids) if ids else 0,
            'hits': hits, 'selected': len(ids), 'relevant': len(gold)}

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = json.loads((ROOT / 'data/book_eval_dataset.json').read_text(encoding='utf-8-sig'))['cases']
    save('labels.json', [{'question': c['question'], 'direct_answer_chunk': g}
                         for c, g in zip(cases, GOLD)])
    rag = build_rag()
    rag.enable_llm_cache = False
    await rag.initialize_storages()
    usage = RequestUsage()
    token = ACTIVE_USAGE.set(usage)
    agent = RetrievalAgent(rag)
    try:
        frozen = []
        for i, case in enumerate(cases):
            path = OUT / f'subtask_{i}.json'
            if path.exists():
                result = json.loads(path.read_text(encoding='utf-8'))
            else:
                result = asdict(await agent.run(case['question']))
                save(path.name, result)
            frozen.append(result)
            print(f'Subtask {i+1}/7: {result["stop_reason"]}, evidence={len(result["evidence"])}', flush=True)
        results = []
        for index, group in enumerate([[i] for i in range(7)] + GROUPS):
            batches = [frozen[i]['evidence'] for i in group]
            gold = {GOLD[i] for i in group}
            row = {'case': index, 'subtasks': group, 'question': '\n'.join(cases[i]['question'] for i in group), 'variants': {}}
            for name, fuse in [('before', merge_evidence), ('after', rrf_evidence)]:
                evidence = list(fuse(batches, agent.settings.evidence_bytes).values())
                item = {'metrics': metrics(evidence, gold), 'at_1': metrics(evidence[:1], gold),
                        'at_3': metrics(evidence[:3], gold), 'evidence_ids': [r['chunk_id'] for r in evidence]}
                if len(group) > 1:
                    path = OUT / f'answer_{index}_{name}.json'
                    if path.exists():
                        item['answer'] = json.loads(path.read_text(encoding='utf-8'))['answer']
                    else:
                        system = '根据证据分别回答用户的每一个问题，简明汇总。只能使用证据，不足时明确说明。保留病程条件和例外，不要编造。'
                        payload = {'question': row['question'], 'evidence': evidence}
                        item['answer'] = await asyncio.wait_for(rag.llm_model_func(
                            bounded_payload(system, payload, agent.settings.prompt_bytes),
                            system_prompt=system, history_messages=[], stream=False, max_tokens=1800), 90)
                        save(path.name, {'answer': item['answer']})
                row['variants'][name] = item
            results.append(row)
            save('results.json', {'evidence_budget': agent.settings.evidence_bytes, 'rrf_k': 60,
                 'method': 'Frozen real Agent results; same fusion budget and synthesis prompt. Fixed decomposition; no final reviewer or repair loop.',
                 'results': results, 'usage_this_run': usage.get_usage()})
            print(f'Case {index+1}/11: ' + str({k: v['metrics'] for k,v in row['variants'].items()}), flush=True)
    finally:
        save('usage_last_run.json', usage.get_usage())
        ACTIVE_USAGE.reset(token)
        await finalize_rag(rag)

if __name__ == '__main__':
    asyncio.run(main())
