import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import numpy as np

from rag_app.mmr import mmr_indices
from rag_app.tools import KnowledgeTools


class MMRTests(unittest.TestCase):
    def test_diversity_and_relevance_extremes(self):
        docs = [[1, 0], [.99, .01], [.7, .7]]
        self.assertEqual(mmr_indices([1, 0], docs, 2, .4), [0, 2])
        self.assertEqual(mmr_indices([1, 0], docs, 2, 1), [0, 1])
        self.assertEqual(mmr_indices([1, 0], docs, 2, 0), [0, 2])

    def test_bounds_stable_ties_and_invalid_vectors(self):
        self.assertEqual(mmr_indices([1, 0], [[1, 0], [1, 0]], 8), [0, 1])
        for query, docs in [([0, 0], [[1, 0]]), ([1, 0], [[np.nan, 1]]), ([1], [[1, 0]])]:
            with self.assertRaises(ValueError):
                mmr_indices(query, docs, 2)
        with self.assertRaises(ValueError):
            mmr_indices([1], [[1]], 2, 1.1)


class MMRToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_top_k_does_not_expand_with_mmr_or_legacy_multiplier(self):
        for enabled in (True, False):
            with patch.dict('os.environ', {'AGENT_MMR_FETCH_MULTIPLIER': '5'}):
                tool, rag = self.setup_tool()
            tool.mmr_enabled = enabled
            result = await tool.execute('search_knowledge', {'query': 'q', 'top_k': 2})
            self.assertEqual(rag.aquery_data.await_args.args[1].chunk_top_k, 2)
            self.assertEqual(rag.aquery_data.await_args.args[1].max_total_tokens, 16000)
            self.assertEqual([r['content'] for r in result['chunks']], ['a', 'b'])
            rag.embedding_func.assert_not_awaited()

    def setup_tool(self):
        rows = [{'chunk_id': 'chunk-' + c*32, 'content': c, 'file_path': 'book.txt'} for c in 'abc']
        rag = SimpleNamespace(aquery_data=AsyncMock(return_value={'data': {'chunks': rows}}),
                              embedding_func=AsyncMock(side_effect=[[[1, 0]], [[1, 0], [.99, .01], [.7, .7]]]))
        with patch.dict('os.environ', {'AGENT_MMR_ENABLED': 'true', 'AGENT_MMR_LAMBDA': '.4'}):
            tool = KnowledgeTools(rag)
        return tool, rag

    async def test_returns_requested_candidates_then_finalizes_diversity(self):
        tool, rag = self.setup_tool()
        result = await tool.execute('search_knowledge', {'query': 'q', 'top_k': 3})
        self.assertEqual(rag.aquery_data.await_args.args[1].chunk_top_k, 3)
        self.assertEqual([r['content'] for r in result['chunks']], ['a', 'b', 'c'])
        rag.embedding_func.assert_not_awaited()
        chunks, selection = await tool.finalize_evidence('q', result['chunks'], top_k=2)
        self.assertEqual([r['content'] for r in chunks], ['a', 'c'])
        result['selection'] = selection
        self.assertEqual(result['selection']['status'], 'ok')

    async def test_provider_failure_falls_back_and_disabled_skips_embedding(self):
        tool, rag = self.setup_tool()
        rag.embedding_func.side_effect = RuntimeError('provider secret')
        result = await tool.execute('search_knowledge', {'query': 'q', 'top_k': 3})
        chunks, selection = await tool.finalize_evidence('q', result['chunks'], top_k=2)
        self.assertEqual([r['content'] for r in chunks], ['a', 'b', 'c'])
        result['selection'] = selection
        self.assertEqual(result['selection']['status'], 'fallback')
        self.assertNotIn('provider secret', str(result))
        tool.mmr_enabled = False
        rag.embedding_func.reset_mock()
        await tool.execute('search_knowledge', {'query': 'q', 'top_k': 3})
        rag.embedding_func.assert_not_awaited()
        self.assertEqual(rag.aquery_data.await_args.args[1].chunk_top_k, 3)

    async def test_timeout_falls_back_but_cancellation_propagates(self):
        tool, rag = self.setup_tool()
        async def slow(*a, **kw):
            await asyncio.sleep(1)
        rag.embedding_func.side_effect = slow
        tool.mmr_timeout = .001
        result = await tool.execute('search_knowledge', {'query': 'q', 'top_k': 3})
        _, selection = await tool.finalize_evidence('q', result['chunks'], top_k=2)
        self.assertEqual(selection['status'], 'fallback')
        rag.embedding_func.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await tool.finalize_evidence('q', result['chunks'], top_k=2)
