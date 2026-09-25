import asyncio
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

from rag_app.agent import RetrievalAgent
from rag_app.agent_types import ACTION_ADAPTER, AgentSettings
from rag_app.context import bounded_payload, merge_evidence, packed, recent_history
from rag_app.tools import KnowledgeTools

A = "chunk-" + "a" * 32
B = "chunk-" + "b" * 32


def chunk(identifier=A, content="source evidence"):
    return {"chunk_id": identifier, "content": content, "file_path": "book.txt"}


def search(query="question"):
    return json.dumps({"action": "search_knowledge", "arguments": {"query": query}})


def answer(*identifiers):
    return json.dumps({"action": "answer", "evidence_ids": list(identifiers or [A])})


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_mmr_only_after_answer_not_budget_stop(self):
        for enough in (True, False):
            tools = KnowledgeTools(SimpleNamespace())
            tools.execute = AsyncMock(return_value={'status': 'ok', 'chunks': [chunk()]})
            tools.finalize_evidence = AsyncMock(return_value=([chunk()], {'status': 'ok'}))
            responses = [search(), answer(), 'final'] if enough else [search(), 'partial']
            async def model(*args, **kwargs):
                if len(responses) > 1:
                    tools.finalize_evidence.assert_not_awaited()
                return responses.pop(0)
            settings = replace(AgentSettings(), max_decisions=3 if enough else 1)
            result = await RetrievalAgent(None, model, settings=settings, tools=tools).run('question')
            self.assertEqual(result.stop_reason, 'answered' if enough else 'budget_answer')
            self.assertEqual(tools.finalize_evidence.await_count, int(enough))

    async def test_decision_failure_retains_already_retrieved_evidence(self):
        llm = AsyncMock(side_effect=[search(), RuntimeError('budget')])
        tools = SimpleNamespace(execute=AsyncMock(return_value={'status':'ok','chunks':[chunk()]}))
        result = await RetrievalAgent(None, llm, tools=tools).run('question')
        self.assertEqual(result.stop_reason, 'model_error')
        self.assertEqual(result.evidence[0]['chunk_id'], A)

    async def test_search_read_answer_preserves_original_question_and_deduplicates(self):
        read = json.dumps({"action": "read_source", "arguments": {"chunk_id": A}})
        llm = AsyncMock(side_effect=[search(), search(), read, answer(A, B), "final answer"])
        tools = SimpleNamespace(execute=AsyncMock(side_effect=[
            {"status": "ok", "chunks": [chunk()]},
            {"status": "ok", "chunks": [chunk(), chunk(B)]},
        ]))
        agent = RetrievalAgent(None, llm, tools=tools)
        history = [{"role": "user", "content": "previous topic"},
                   {"role": "assistant", "content": "old answer"}]
        result = await agent.run("current topic", history)
        self.assertEqual(result.answer, "final answer")
        self.assertEqual(result.stop_reason, "answered")
        self.assertEqual(tools.execute.await_count, 2)
        self.assertEqual([item["chunk_id"] for item in result.evidence], [A, B])
        self.assertTrue(any(item["status"] == "duplicate" for item in result.trace))
        final = json.loads(llm.await_args.args[0])
        self.assertEqual(final["question"], "current topic")
        self.assertEqual(history[0]["content"], "previous topic")
        self.assertEqual(llm.await_args.kwargs["history_messages"], [])

    async def test_second_search_can_complete_comparison(self):
        llm = AsyncMock(side_effect=[search("topic one"), search("topic two"), answer(A, B), "comparison"])
        tools = SimpleNamespace(execute=AsyncMock(side_effect=[
            {"status": "ok", "chunks": [chunk()]}, {"status": "ok", "chunks": [chunk(B)]}]))
        result = await RetrievalAgent(None, llm, tools=tools).run("compare topics")
        self.assertEqual(result.answer, "comparison")
        self.assertEqual(len(result.evidence), 2)

    async def test_budget_blocks_extra_tool_and_still_allows_answer(self):
        llm = AsyncMock(side_effect=[search("one"), search("two"), answer(), "answer"])
        tools = SimpleNamespace(execute=AsyncMock(return_value={"status": "ok", "chunks": [chunk()]}))
        settings = replace(AgentSettings(), max_tools=1, max_searches=1)
        result = await RetrievalAgent(None, llm, settings, tools).run("question")
        self.assertEqual(tools.execute.await_count, 1)
        self.assertEqual(result.stop_reason, "answered")
        self.assertTrue(any(row["status"] == "budget_blocked" for row in result.trace))

    async def test_decision_limit_generates_from_available_evidence(self):
        llm = AsyncMock(side_effect=[search(), "limited answer"])
        tools = SimpleNamespace(execute=AsyncMock(return_value={"status": "ok", "chunks": [chunk()]}))
        result = await RetrievalAgent(None, llm, replace(AgentSettings(), max_decisions=1), tools).run("question")
        self.assertEqual(result.stop_reason, "budget_answer")
        self.assertTrue(json.loads(llm.await_args.args[0])["budget_stop"])

    async def test_invalid_actions_and_fabricated_ids_never_execute_or_answer(self):
        for responses in (["not json", '{"action":"delete_file"}'], [answer(B), answer(B)]):
            with self.subTest(responses=responses):
                tools = SimpleNamespace(execute=AsyncMock(return_value={'status':'empty','chunks':[]}))
                result = await RetrievalAgent(None, AsyncMock(side_effect=responses), tools=tools).run("question")
                self.assertEqual(result.stop_reason, "invalid_action")
                self.assertEqual(result.evidence, [])
                tools.execute.assert_awaited_once_with('search_knowledge', {'query': 'question', 'mode': 'hybrid', 'top_k': 6})

    async def test_invalid_decisions_recover_via_bounded_search(self):
        llm = AsyncMock(side_effect=['not json', 'not json', 'recovered answer'])
        tools = SimpleNamespace(execute=AsyncMock(return_value={'status':'ok','chunks':[chunk()]}))
        result = await RetrievalAgent(None, llm, tools=tools).run('question')
        self.assertEqual(result.answer, 'recovered answer')
        self.assertEqual(result.stop_reason, 'budget_answer')
        self.assertTrue(any(e['action']=='recovery' for e in result.trace))
        self.assertEqual(tools.execute.await_count, 1)

    async def test_clarification_and_empty_results(self):
        tools = SimpleNamespace(execute=AsyncMock(return_value={"status": "empty", "chunks": []}))
        llm = AsyncMock(side_effect=[search(), '{"action":"insufficient"}'])
        result = await RetrievalAgent(None, llm, tools=tools).run("unknown fact")
        self.assertEqual(result.stop_reason, "insufficient")
        clarify = AsyncMock(return_value='{"action":"clarify","question":"Which concept?"}')
        result = await RetrievalAgent(None, clarify, tools=tools).run("What about that?")
        self.assertEqual(result.stop_reason, "clarify")

    async def test_tool_errors_are_sanitized_and_recoverable(self):
        llm = AsyncMock(side_effect=[search("first"), search("second"), answer(), "answer"])
        tools = SimpleNamespace(execute=AsyncMock(side_effect=[
            RuntimeError("secret provider error"), {"status": "ok", "chunks": [chunk()]}]))
        result = await RetrievalAgent(None, llm, tools=tools).run("question")
        self.assertEqual(result.stop_reason, "answered")
        self.assertNotIn("secret", packed(result.trace))

    async def test_timeout_cancels_pending_call(self):
        cancelled = asyncio.Event()

        async def slow(*args, **kwargs):
            try:
                await asyncio.sleep(1)
            finally:
                cancelled.set()

        settings = replace(AgentSettings(), request_timeout=0.02, model_timeout=0.5)
        result = await RetrievalAgent(None, slow, settings).run("question")
        self.assertEqual(result.stop_reason, "timeout")
        self.assertTrue(cancelled.is_set())

    async def test_tool_timeout_releases_budgeted_loop(self):
        async def slow(*args):
            await asyncio.sleep(1)

        llm = AsyncMock(side_effect=[search(), '{"action":"insufficient"}'])
        result = await RetrievalAgent(None, llm, replace(AgentSettings(), tool_timeout=0.01),
                                      SimpleNamespace(execute=slow)).run("question")
        self.assertTrue(any(row["status"] == "timeout" for row in result.trace))

    async def test_recall_reads_history_without_llm(self):
        llm = AsyncMock()
        result = await RetrievalAgent(None, llm).run("我刚才问了什么？", [
            {"role": "user", "content": "焦虑的概念是什么？"}])
        self.assertEqual(result.stop_reason, "recall")
        self.assertIn("焦虑的概念是什么？", result.answer)
        llm.assert_not_awaited()


class ToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_returns_evidence_without_generation(self):
        rag = SimpleNamespace(aquery_data=AsyncMock(return_value={
            "status": "success", "data": {"chunks": [chunk()], "entities": []}}))
        result = await KnowledgeTools(rag).execute("search_knowledge", {"query": "q", "mode": "naive"})
        self.assertEqual(result["chunks"][0]["chunk_id"], A)
        self.assertEqual(rag.aquery_data.await_args.args[1].mode, "naive")

    async def test_source_uses_document_order_and_handles_missing_ids(self):
        rag = SimpleNamespace(
            text_chunks=SimpleNamespace(get_by_id=AsyncMock(return_value={"full_doc_id": "doc"}),
                                        get_by_ids=AsyncMock(return_value=[chunk(), chunk(B)])),
            doc_status=SimpleNamespace(get_by_id=AsyncMock(return_value={"chunks_list": [A, B]})))
        tools = KnowledgeTools(rag)
        result = await tools.execute("read_source", {"chunk_id": A})
        self.assertEqual(len(result["chunks"]), 2)
        rag.text_chunks.get_by_ids.assert_awaited_once_with([A, B])
        rag.text_chunks.get_by_id.return_value = None
        self.assertEqual((await tools.execute("read_source", {"chunk_id": A}))["status"], "not_found")

    async def test_graph_bounded_and_returns_source_evidence(self):
        rag = SimpleNamespace(get_knowledge_graph=AsyncMock(return_value={
            "nodes": [{"id": "entity", "properties": {"source_id": A}}],
            "edges": [], "is_truncated": True}),
            text_chunks=SimpleNamespace(get_by_ids=AsyncMock(return_value=[chunk()])))
        result = await KnowledgeTools(rag).execute("query_graph", {"entity_name": "entity"})
        self.assertEqual(result["chunks"][0]["chunk_id"], A)
        self.assertTrue(result["truncated"])
        rag.get_knowledge_graph.assert_awaited_once_with("entity", max_depth=1, max_nodes=12)

    async def test_rejects_arbitrary_paths_unknown_tools_and_invalid_limits(self):
        tools = KnowledgeTools(None)
        for name, arguments in [
            ("read_source", {"chunk_id": "../../.env"}),
            ("query_graph", {"entity_name": "*"}),
            ("query_graph", {"entity_name": "entity", "max_depth": 100}),
            ("search_knowledge", {"query": "q", "top_k": 0}),
            ("search_knowledge", {"query": "q", "top_k": "5"}),
            ("delete", {}),
        ]:
            with self.subTest(name=name, arguments=arguments), self.assertRaises(ValueError):
                await tools.execute(name, arguments)


class ContextTests(unittest.TestCase):
    def test_later_search_receives_space_when_first_search_fills_budget(self):
        old = [chunk("chunk-" + c * 32, "old evidence" * 30) for c in "abc"]
        new = [chunk("chunk-" + "d" * 32, "new evidence" * 30)]
        budget = len(packed([old[0], new[0]]).encode("utf-8"))
        selected = merge_evidence([old, new], budget)
        self.assertIn(old[0]["chunk_id"], selected)
        self.assertIn(new[0]["chunk_id"], selected)
        self.assertLessEqual(len(packed(list(selected.values())).encode("utf-8")), budget)

    def test_limits_complete_turns_without_modifying_original_history(self):
        history = [{"role": role, "content": str(i) * 100}
                   for i in range(10) for role in ("user", "assistant")]
        selected = recent_history(history, 4, 600)
        self.assertLessEqual(len(packed(selected).encode("utf-8")), 600)
        self.assertEqual(len(selected) % 2, 0)
        self.assertEqual(selected[-1], history[-1])
        self.assertEqual(len(history), 20)

    def test_prompt_budget_keeps_original_question(self):
        payload = {"question": "original question", "history": [],
                   "evidence": [chunk(content="证据" * 10000)]}
        prompt = bounded_payload("system", payload, 1200)
        self.assertLessEqual(len(("system" + prompt).encode("utf-8")), 1200)
        self.assertEqual(json.loads(prompt)["question"], "original question")

    def test_schema_rejects_extra_keys(self):
        with self.assertRaises(ValueError):
            ACTION_ADAPTER.validate_python({"action": "insufficient", "execute": "danger"})

    def test_oversized_graph_observations_are_bounded(self):
        payload = {"question": "current question", "evidence": [chunk()],
                   "observations": [{"nodes": ["node" * 10000], "status": "ok"}] * 5}
        prompt = bounded_payload("system", payload, 1500)
        self.assertLessEqual(len(("system" + prompt).encode("utf-8")), 1500)
        self.assertEqual(json.loads(prompt)["evidence"][0]["chunk_id"], A)
