import asyncio
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from lightrag.kg.json_kv_impl import JsonKVStorage


class QueryCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_document_changes_invalidate_answers_not_extraction(self):
        cache = SimpleNamespace(_storage_lock=asyncio.Lock(), _data={
            'hybrid:query:123': {}, 'default:extract:456': {}, 'hybrid:keywords:789': {}},
            delete=AsyncMock(), index_done_callback=AsyncMock())
        await JsonKVStorage.delete_query_cache(cache)
        cache.delete.assert_awaited_once_with(['hybrid:query:123'])
        cache.index_done_callback.assert_awaited_once()
