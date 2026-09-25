"""Paired repeated timing of ranking+MMR with identical texts and no LLM."""
import asyncio,json,time,statistics
from rag_app.evaluate_open import OUT
from rag_app.runtime import make_embedding_func,make_embedding_rerank_func
from rag_app.embedding_reuse import SEARCH_EMBEDDINGS,SearchEmbeddings,embed_once
from rag_app.mmr import mmr_indices
from rag_app.telemetry import ACTIVE_USAGE,RequestUsage

async def main():
 source=json.loads((OUT/'isolated_index/demo/kv_store_text_chunks.json').read_text(encoding='utf-8'))
 # Freeze actual evidence returned in the benchmark, not hand-made vector inputs.
 ids=[]
 for path in sorted(OUT.glob('mood_compare_[A-F]*.json')):
  if '.judge.' in path.name:continue
  d=json.loads(path.read_text(encoding='utf-8'))
  for s in d['state']['steps']:
   for e in s.get('evidence',[]):
    if e['chunk_id'] not in ids:ids.append(e['chunk_id'])
 ids=ids[:16];texts=[source[i]['content'] for i in ids]
 query='抑郁发作和躁狂发作在情绪、精力、睡眠及病程方面有哪些区别和联系？'
 embed=make_embedding_func('remote');rerank=make_embedding_rerank_func(embed);rows=[]
 for repeat in range(3):
  for reuse in ([False,True] if repeat%2==0 else [True,False]):
   usage=RequestUsage();u=ACTIVE_USAGE.set(usage)
   cache=SEARCH_EMBEDDINGS.set(SearchEmbeddings('benchmark')) if reuse else None
   start=time.monotonic()
   try:
    ranks=await rerank(query,texts,len(texts));ordered=[texts[r['index']] for r in ranks]
    q=await embed_once(embed,[query],'query');vec=await embed_once(embed,ordered,'document')
    selected=mmr_indices(q[0],vec,6,.7)
    rows.append({'repeat':repeat,'reuse':reuse,'elapsed_seconds':time.monotonic()-start,
      'usage':usage.get_usage(),'selected_ids':[ids[ranks[i]['index']] for i in selected]})
    print(repeat,reuse,round(rows[-1]['elapsed_seconds'],2),usage.get_usage()['embedding_tokens'],flush=True)
   finally:
    if cache is not None:SEARCH_EMBEDDINGS.reset(cache)
    ACTIVE_USAGE.reset(u)
 assert all(r['selected_ids']==rows[0]['selected_ids'] for r in rows),'Provider outputs changed ordering'
 (OUT/'reuse_microbenchmark.json').write_text(json.dumps({'query':query,'candidate_ids':ids,'runs':rows},ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':asyncio.run(main())
