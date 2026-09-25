"""Reproducible open-question workflow ablation; production configuration unchanged."""
import asyncio
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
from rag_app.api import build_rag
from rag_app.config import INDEX_DIR
from rag_app.agent import RetrievalAgent
from rag_app.tools import KnowledgeTools
from rag_app.workflow import ResearchWorkflow
from rag_app.context import merge_evidence
from rag_app.telemetry import ACTIVE_USAGE, RequestUsage
from rag_app.lifecycle import finalize_rag

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/evaluation/open_ablation_20260914'
A='chunk-a86535a2753302c895e7ba23a4cda01e'
D='chunk-ed1a6c32057e0d7f5dff1705d2b43a00'
M='chunk-8e37b9f67b680ed6b3aef53d722181d2'
X='chunk-2b7c87b39bcca1e85fd0c70351b936d5'
CASES=[
 {'id':'mood_compare','question':'抑郁发作和躁狂发作在情绪、精力、睡眠及病程方面有哪些区别和联系？',
  'gold':[D,M], 'criteria':['比较情绪表现','比较精力表现','比较睡眠表现','分别给出病程要求并保留合并分裂症状时的条件','解释与双相障碍的联系，不添加资料外的治疗结论']},
 {'id':'colloquial_anxiety','question':'总是担心事情、很难放松，和突然一阵特别害怕有什么区别？请结合资料比较表现、持续方式和病程条件，不要直接给个人下诊断。',
  'gold':[X], 'criteria':['对应广泛性焦虑与惊恐障碍，但不直接诊断个人','比较持续担忧与突然发作及其身体表现','广泛性焦虑的六个月条件','惊恐障碍一个月三次或首次后担忧持续一个月的两种情形','说明需要排除其他原因，日常描述不足以确诊']},
 {'id':'schizophrenia_compare','question':'请比较精神分裂症的一般病程标准和单纯型的要求，结合主要症状说明为什么不能只凭持续时间判断，并保留例外与附加条件。',
  'gold':[A], 'criteria':['一般症状和严重标准至少一个月，单纯型另有规定','合并情感障碍时情感症状不再符合标准后分裂症状仍持续至少两周','单纯型阴性症状为主且从无明显阳性症状','单纯型起病隐袭缓慢、病程至少两年','诊断还需症状、社会功能损害及排除条件，不只看时长']},
 {'id':'course_exceptions','question':'请跨章节整理精神分裂症、抑郁发作、躁狂发作的病程要求，比较异同，并逐一解释例外或附加条件。',
  'gold':[A,D,M], 'criteria':['分裂症一般一个月，单纯型两年的例外','分裂症合并情感障碍时减轻到不符合情感标准后至少两周','抑郁症状与严重标准至少两周及合并分裂症状缓解后两周条件','躁狂症状与严重标准至少一周及合并分裂症状缓解后一周条件','明确时间标准不同，不能仅凭时长确诊或混用不同疾病条件']},
]
# Baseline still uses LightRAG and Agent; only optional ranking/fusion vary.
VARIANTS=[('A_base',False,False,False,False),('B_cosine',True,False,False,False),
 ('C_cosine_rrf',True,True,False,False),('D_cosine_mmr',True,False,True,True),
 ('E_full_no_reuse',True,True,True,False),('F_full_reuse',True,True,True,True)]

def save(name,value):
 (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

class NoReuseTools(KnowledgeTools):
 async def execute(self,name,arguments):
  return await self._execute(name,arguments)

async def identity_rerank(query,documents,top_n=None):
 return [{'index':i,'relevance_score':1.0} for i in range(min(len(documents),top_n or len(documents)))]

def round_robin(batches,budget,k):
 # Workflow trace schema expects these keys; None denotes no RRF score.
 return {key:{**row,'rrf_score':None,'rrf_support':None} for key,row in merge_evidence(batches,budget).items()}

async def main():
 OUT.mkdir(parents=True,exist_ok=True)
 index=OUT/'isolated_index'
 if not index.exists(): shutil.copytree(INDEX_DIR,index)
 source=json.loads((index/'demo/kv_store_text_chunks.json').read_text(encoding='utf-8'))
 save('dataset.json',{'cases':CASES,'gold_records':{i:source[i] for i in {k for c in CASES for k in c['gold']}},
  'relevance_definition':'Direct answer sections only, not every background-relevant chunk. Labels checked before runs.',
  'variants':VARIANTS,'limits':{'llm':50000,'embedding':250000},'repetitions':1})
 with patch('rag_app.api.INDEX_DIR',index): rag=build_rag()
 await rag.initialize_storages()
 rag.enable_llm_cache=False
 rag.llm_response_cache.global_config['enable_llm_cache']=False
 normal_rerank=rag.rerank_model_func
 try:
  for ci,case in enumerate(CASES):
   # Rotate execution order to reduce a fixed warm-up/order advantage.
   variants=VARIANTS[ci:]+VARIANTS[:ci]
   for name,cosine,rrf,mmr,reuse in variants:
    filename=f'{case["id"]}_{name}.json'
    if (OUT/filename).exists(): continue
    print('START',case['id'],name,flush=True)
    rag.rerank_model_func=normal_rerank if cosine else identity_rerank
    with patch.dict(os.environ,{'AGENT_MMR_ENABLED':str(mmr).lower(),'WORKFLOW_LLM_TOKEN_BUDGET':'50000',
       'WORKFLOW_EMBEDDING_TOKEN_BUDGET':'250000','WORKFLOW_MAX_ROUNDS':'2','WORKFLOW_TIMEOUT':'180'}):
     agent=RetrievalAgent(rag,tools=(KnowledgeTools(rag) if reuse else NoReuseTools(rag)))
     usage=RequestUsage(); token=ACTIVE_USAGE.set(usage);state={};started=time.monotonic()
     try:
      if rrf:
       result=await ResearchWorkflow(agent).run(case['question'],on_state=lambda s:state.update(s))
      else:
       with patch('rag_app.workflow.rrf_evidence',round_robin):
        result=await ResearchWorkflow(agent).run(case['question'],on_state=lambda s:state.update(s))
      elapsed=time.monotonic()-started
      ids={e['chunk_id'] for e in result.evidence};gold=set(case['gold']);hits=len(ids&gold)
      save(filename,{'case':case['id'],'variant':name,'question':case['question'],'result':asdict(result),
       'state':state,'elapsed_seconds':elapsed,'usage':usage.get_usage(),
       'metrics':{'recall':hits/len(gold),'precision':hits/len(ids) if ids else 0,'hits':hits,'evidence_count':len(ids),
        'tool_calls':sum(e.get('action') in {'search_knowledge','read_source','query_graph'} and e.get('status') not in {'duplicate','budget_blocked'} for e in result.trace)}})
      print('DONE',case['id'],name,result.stop_reason,round(elapsed,1),usage.get_usage()['total_tokens'],usage.get_usage()['embedding_tokens'],flush=True)
     finally:ACTIVE_USAGE.reset(token)
 finally: await finalize_rag(rag)

if __name__=='__main__':asyncio.run(main())
