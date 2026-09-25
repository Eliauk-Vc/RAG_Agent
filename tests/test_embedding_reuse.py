import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import numpy as np
from rag_app.embedding_reuse import SEARCH_EMBEDDINGS, SearchEmbeddings, embed_once
from rag_app.runtime import make_embedding_rerank_func
from rag_app.tools import KnowledgeTools


class ReuseTests(unittest.IsolatedAsyncioTestCase):
    async def test_rerank_and_mmr_share_exact_vectors_including_long_text(self):
        contents = ['原文' * 2200, 'b', 'c']
        embedding = AsyncMock(side_effect=[[[1, 0]], [[1, 0], [.99, .01], [.7, .7]]])
        rerank = make_embedding_rerank_func(embedding)
        async def retrieve(query, param):
            ranking = await rerank(query, contents, param.chunk_top_k)
            return {'data': {'chunks': [{'chunk_id':'chunk-'+('abc'[r['index']]*32),
                'content':contents[r['index']], 'file_path':'book.txt'} for r in ranking]}}
        async def wrapped_embedding(texts, **kwargs):
            return await embedding(texts, **kwargs)
        rag = SimpleNamespace(embedding_func=wrapped_embedding, aquery_data=retrieve)
        token = SEARCH_EMBEDDINGS.set(SearchEmbeddings(id(rag)))
        try:
            with patch.dict('os.environ', {'AGENT_MMR_LAMBDA':'.4','AGENT_MMR_ENABLED':'true'}):
                tool = KnowledgeTools(rag)
                result = await tool.execute('search_knowledge', {'query':'q','top_k':3})
                self.assertEqual(len(result['chunks']), 3)
                chunks, selection = await tool.finalize_evidence('q', result['chunks'], top_k=2)
            self.assertEqual(embedding.await_count, 2)
            self.assertEqual(SEARCH_EMBEDDINGS.get().hits, 4)
            self.assertEqual(chunks[1]['content'], 'c')
            self.assertTrue(chunks[0]['truncated'])
        finally:
            SEARCH_EMBEDDINGS.reset(token)
        self.assertIsNone(SEARCH_EMBEDDINGS.get())

    async def test_duplicate_texts_and_concurrent_calls_only_compute_once(self):
        embedding=AsyncMock(return_value=[[1,2]])
        token=SEARCH_EMBEDDINGS.set(SearchEmbeddings())
        try:
            a,b=await asyncio.gather(embed_once(embedding,['same','same'],'document'),
                                     embed_once(embedding,['same'],'document'))
            self.assertEqual(embedding.await_count,1)
            np.testing.assert_array_equal(a[0],b[0])
            await embed_once(embedding,['same'],'query')
            self.assertEqual(embedding.await_count,2)
        finally:
            SEARCH_EMBEDDINGS.reset(token)

    async def test_failure_does_not_cache_and_scopes_do_not_leak(self):
        embedding=AsyncMock(side_effect=[RuntimeError('offline'),[[1,2]],[[3,4]]])
        token=SEARCH_EMBEDDINGS.set(SearchEmbeddings())
        try:
            with self.assertRaises(RuntimeError):
                await embed_once(embedding,['x'],'document')
            await embed_once(embedding,['x'],'document')
        finally:
            SEARCH_EMBEDDINGS.reset(token)
        token=SEARCH_EMBEDDINGS.set(SearchEmbeddings())
        try:
            values=await embed_once(embedding,['x'],'document')
            np.testing.assert_array_equal(values,[[3,4]])
        finally:
            SEARCH_EMBEDDINGS.reset(token)
