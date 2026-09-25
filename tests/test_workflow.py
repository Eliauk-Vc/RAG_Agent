import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from rag_app.agent_types import AgentResult, AgentSettings
from rag_app.workflow import ResearchWorkflow, FALLBACK_LABEL
from rag_app.telemetry import ACTIVE_USAGE, RequestUsage


EVIDENCE = [{'chunk_id': 'chunk-' + 'a' * 32, 'content': '原文证据', 'file_path': 'book.txt'}]
PLAN = json.dumps({'optimized_question': '完整问题', 'subtasks': ['子问题一', '子问题二']})
REVIEW = json.dumps({'meets_need': True, 'grounded': True, 'issues': [], 'followup': ''})


class WorkflowTests(unittest.IsolatedAsyncioTestCase):
    def agent(self, responses):
        return SimpleNamespace(llm=AsyncMock(side_effect=responses), settings=AgentSettings(),
                               run=AsyncMock(return_value=AgentResult('子答案', 'answered', EVIDENCE)))

    async def test_plan_each_agent_synthesis_and_review(self):
        agent = self.agent([PLAN, '汇总结果', REVIEW])
        states = []
        result = await ResearchWorkflow(agent).run('原始问题', on_state=lambda state: states.append(json.loads(json.dumps(state))))
        self.assertEqual(result.answer, '汇总结果')
        self.assertEqual(agent.run.await_count, 2)
        self.assertIn('原始问题', agent.run.await_args_list[0].args[0])
        self.assertEqual(states[-1]['phase'], 'completed')
        self.assertTrue(states[-1]['review']['grounded'])
        self.assertEqual(states[-1]['fusion']['method'], 'rrf')
        self.assertEqual(states[-1]['fusion']['selected'][0]['support'], 2)
        synthesis = json.loads(agent.llm.await_args_list[1].args[0])
        self.assertAlmostEqual(synthesis['evidence'][0]['rrf_score'], 2 / 61)

    async def test_repair_round_and_maximum_fallback_label(self):
        bad = json.dumps({'meets_need': False, 'grounded': True, 'issues': ['缺内容'], 'followup': '补充问题'})
        agent = self.agent([PLAN, '初稿', bad, '修订稿', bad, '通用知识'])
        workflow = ResearchWorkflow(agent)
        result = await workflow.run('原始问题')
        self.assertEqual(agent.run.await_count, 3)
        self.assertEqual(result.stop_reason, 'partial')
        self.assertFalse(result.answer.startswith(FALLBACK_LABEL))
        self.assertTrue(result.evidence)
        self.assertIn('缺内容', result.answer)
        self.assertEqual(result.evaluation_payload()['status'], 'failure')

    async def test_transient_agent_error_is_retried(self):
        plan = json.dumps({'optimized_question': '问题', 'subtasks': ['子问题']})
        agent = self.agent([plan, '最终答案', REVIEW])
        agent.run.side_effect = [AgentResult('', 'timeout'), AgentResult('恢复', 'answered', EVIDENCE)]
        result = await ResearchWorkflow(agent).run('问题')
        self.assertEqual(result.stop_reason, 'answered')
        self.assertEqual(agent.run.await_count, 2)

    async def test_bad_plan_uses_original_then_no_evidence_fallback(self):
        agent = self.agent(['bad json', 'bad json', '通用知识'])
        agent.run.return_value = AgentResult('', 'insufficient')
        result = await ResearchWorkflow(agent).run('原始问题')
        self.assertEqual(result.stop_reason, 'fallback')
        self.assertEqual(agent.run.await_count, 2)

    async def test_cancellation_does_not_invoke_fallback(self):
        agent = self.agent([])
        async def slow(*args, **kwargs):
            await asyncio.sleep(30)
        agent.llm = AsyncMock(side_effect=slow)
        task = asyncio.create_task(ResearchWorkflow(agent).run('问题'))
        await asyncio.sleep(.02)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(agent.llm.await_count, 1)

    async def test_provider_failure_is_finite(self):
        agent = self.agent([])
        agent.llm.side_effect = RuntimeError('secret')
        agent.run.return_value = AgentResult('', 'model_error')
        result = await ResearchWorkflow(agent).run('问题')
        self.assertEqual(result.stop_reason, 'model_error')
        self.assertNotIn('secret', json.dumps(result.trace))
        self.assertLessEqual(agent.llm.await_count, 4)

    async def test_token_budget_stops_retrieval_but_allows_labelled_fallback(self):
        usage = RequestUsage()
        usage.add_usage({'total_tokens': 100001})
        token = ACTIVE_USAGE.set(usage)
        agent = self.agent(['通用知识'])
        try:
            result = await ResearchWorkflow(agent).run('问题')
        finally:
            ACTIVE_USAGE.reset(token)
        self.assertEqual(result.stop_reason, 'insufficient')
        agent.run.assert_not_awaited()
        agent.llm.assert_not_awaited()

    async def test_embedding_budget_preserves_completed_and_unstarted_topics(self):
        usage = RequestUsage()
        token = ACTIVE_USAGE.set(usage)
        agent = self.agent([PLAN, '按原文整理的部分答案'])
        async def run(*args, **kw):
            usage.embedding_usage.add_usage({'total_tokens': 250001})
            return AgentResult('已完成第一题', 'answered', EVIDENCE)
        agent.run.side_effect = run
        try:
            result = await ResearchWorkflow(agent).run('问题')
        finally:
            ACTIVE_USAGE.reset(token)
        self.assertEqual(result.stop_reason, 'partial')
        self.assertEqual(agent.run.await_count, 1)
        self.assertIn('子问题二', result.answer)
        self.assertIn('嵌入模型 Token 预算超限', result.answer)
        self.assertIn('250,000', result.answer)
        payload = json.loads(agent.llm.await_args.args[0])
        self.assertTrue(payload['evidence'])
        self.assertEqual(payload['completed'], ['子问题一'])

    async def test_embedding_usage_no_longer_consumes_llm_budget(self):
        usage = RequestUsage()
        usage.embedding_usage.add_usage({'total_tokens': 120000})
        token = ACTIVE_USAGE.set(usage)
        try:
            result = await ResearchWorkflow(self.agent([PLAN, '汇总', REVIEW])).run('问题')
        finally:
            ACTIVE_USAGE.reset(token)
        self.assertEqual(result.stop_reason, 'answered')

    async def test_no_llm_budget_returns_evidence_without_extra_call(self):
        usage = RequestUsage()
        token = ACTIVE_USAGE.set(usage)
        agent = self.agent([PLAN])
        async def run(*args, **kw):
            usage.add_usage({'total_tokens': 50000})
            return AgentResult('子答案', 'answered', EVIDENCE)
        agent.run.side_effect = run
        try:
            result = await ResearchWorkflow(agent).run('问题')
        finally:
            ACTIVE_USAGE.reset(token)
        self.assertEqual(result.stop_reason, 'partial')
        self.assertIn('原文证据', result.answer)
        self.assertIn('LLM 模型 Token 预算超限', result.answer)
        self.assertIn('50,000', result.answer)
        self.assertEqual(agent.llm.await_count, 1)

    async def test_recall_does_not_replan_old_question(self):
        agent = self.agent([])
        result = await ResearchWorkflow(agent).run('我刚才问了什么', [{'role':'user','content':'旧问题'}, {'role':'assistant','content':'回答'}])
        self.assertEqual(result.stop_reason, 'recall')
        self.assertIn('旧问题', result.answer)
        agent.llm.assert_not_awaited()
