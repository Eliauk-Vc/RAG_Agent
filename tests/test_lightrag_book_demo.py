import asyncio
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np


DEMO_PATH = Path(__file__).resolve().parents[1] / "rag_app" / "runtime.py"
SPEC = importlib.util.spec_from_file_location("lightrag_book_demo", DEMO_PATH)
assert SPEC and SPEC.loader
DEMO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEMO)


class LightRAGBookDemoTests(unittest.TestCase):
    def test_recall_does_not_intercept_knowledge_followup(self):
        history = [{"role": "user", "content": "焦虑症的症状是什么？"}]
        for question in ("我刚才的问题是什么", "我刚刚问了什么？", "我上一轮问的是什么？"):
            self.assertEqual(
                DEMO.recall_previous_question(question, history),
                "你刚才的问题是：“焦虑症的症状是什么？”",
            )
        self.assertIsNone(DEMO.recall_previous_question("我刚才问的焦虑症有什么症状？", history))

    def test_original_question_and_history_scope_answer_cache(self):
        args = SimpleNamespace(history_turns=0, no_query_rewrite=False,
                               mode="hybrid", only_context=False,
                               rerank="embedding", rerank_top_k=10)
        rag = SimpleNamespace(aquery=AsyncMock(return_value="answer"))
        llm = AsyncMock(return_value='{"query":"same retrieval query"}')
        history = [{"role": "user", "content": "previous topic"}]
        prompts = []
        for question, messages in [
            ("new topic", history),
            ("different question", history),
            ("new topic", [{"role": "user", "content": "different history"}]),
            ("new topic", history),
        ]:
            asyncio.run(DEMO.query_rag(rag, args, question, llm, messages))
            instructions = rag.aquery.await_args.kwargs["param"].user_prompt
            self.assertIn(question, instructions)
            prompts.append(instructions)
        self.assertEqual(len(set(prompts)), 3)
        self.assertEqual(prompts[0], prompts[3])

    def test_rewrite_query_uses_history_and_returns_standalone_question(self) -> None:
        llm_model_func = AsyncMock(
            return_value='{"query":"精神分裂症的病程标准是什么？"}'
        )
        history = [
            {"role": "user", "content": "介绍精神分裂症。"},
            {"role": "assistant", "content": "精神分裂症是一类精神障碍。"},
        ]

        rewritten = asyncio.run(
            DEMO.rewrite_query_for_retrieval(
                llm_model_func,
                "它的病程标准呢？",
                history,
            )
        )

        self.assertEqual(rewritten, "精神分裂症的病程标准是什么？")
        self.assertEqual(
            llm_model_func.await_args.kwargs["history_messages"], history
        )

    def test_query_rag_retrieves_with_rewritten_query_and_reranks(self) -> None:
        rag = SimpleNamespace(aquery=AsyncMock(return_value="answer"))
        llm_model_func = AsyncMock(
            return_value='{"query":"精神分裂症的病程标准是什么？"}'
        )
        args = SimpleNamespace(
            history_turns=0,
            no_query_rewrite=False,
            mode="hybrid",
            only_context=False,
            rerank="embedding",
            rerank_top_k=7,
        )
        history = [
            {"role": "user", "content": "介绍精神分裂症。"},
            {"role": "assistant", "content": "回答内容。"},
        ]

        answer = asyncio.run(
            DEMO.query_rag(
                rag,
                args,
                "它的病程标准呢？",
                llm_model_func,
                history,
            )
        )

        self.assertEqual(answer, "answer")
        self.assertEqual(
            rag.aquery.await_args.args[0], "精神分裂症的病程标准是什么？"
        )
        query_param = rag.aquery.await_args.kwargs["param"]
        self.assertEqual(query_param.conversation_history, history)
        self.assertTrue(query_param.enable_rerank)
        self.assertEqual(query_param.chunk_top_k, 7)

    def test_embedding_reranker_orders_documents_by_cosine_similarity(self) -> None:
        embedding_func = AsyncMock()
        embedding_func.side_effect = [
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array(
                [
                    [0.0, 1.0],
                    [0.9, 0.1],
                    [-1.0, 0.0],
                ],
                dtype=np.float32,
            ),
        ]
        rerank = DEMO.make_embedding_rerank_func(embedding_func)

        results = asyncio.run(
            rerank("query", ["unrelated", "relevant", "opposite"], top_n=2)
        )

        self.assertEqual([result["index"] for result in results], [1, 0])
        self.assertGreater(
            results[0]["relevance_score"], results[1]["relevance_score"]
        )


if __name__ == "__main__":
    unittest.main()
