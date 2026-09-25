import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from rag_app.agent_types import AgentResult
from rag_app.api import create_app
from rag_app.telemetry import ACTIVE_USAGE


class ManagementTests(unittest.TestCase):
    def wait(self, client, identifier):
        for _ in range(300):
            job = client.get('/management/jobs/' + identifier).json()
            if job['status'] not in {'queued', 'running'}:
                return job
            time.sleep(.01)
        self.fail('Job did not finish')

    def test_import_verified_before_replace_and_delete(self):
        rag = AsyncMock()
        rag.doc_status.get_by_id.return_value = {'status': 'processed'}
        rag.adelete_by_doc_id.return_value = SimpleNamespace(status='success')
        rag.doc_status.get_docs_paginated.return_value = ([('doc-old', {'file_path': 'old.txt', 'status': 'processed', 'chunks_count': 1})], 1)
        old = 'doc-' + 'a' * 32
        with TestClient(create_app(lambda: rag, db_path=':memory:')) as client:
            self.assertEqual(client.get('/management/documents').json()['total'], 1)
            job = client.post('/management/documents', json={'name': '../new.txt', 'content': 'new content', 'replaces': old}).json()
            self.assertEqual(self.wait(client, job['id'])['status'], 'completed')
            self.assertEqual(rag.ainsert.await_args.kwargs['file_paths'], 'new.txt')
            rag.adelete_by_doc_id.assert_awaited_once_with(old)
            rag.llm_response_cache.delete_query_cache.assert_awaited_once()
            deleted = client.delete('/management/documents/' + old).json()
            self.assertEqual(self.wait(client, deleted['id'])['status'], 'completed')

    def test_failed_import_keeps_old_document_and_retries_once(self):
        rag = AsyncMock()
        rag.doc_status.get_by_id.return_value = {'status': 'processed'}
        rag.ainsert.side_effect = RuntimeError('secret-key')
        with TestClient(create_app(lambda: rag, db_path=':memory:')) as client:
            job = client.post('/management/documents', json={'name': 'new.txt', 'content': 'new', 'replaces': 'doc-'+'a'*32}).json()
            result = self.wait(client, job['id'])
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(rag.ainsert.await_count, 2)
            rag.adelete_by_doc_id.assert_not_awaited()
            self.assertNotIn('secret-key', str(result))
            self.assertEqual(client.post('/management/documents', json={'name': ' ', 'content': ' '}).status_code, 422)

    def test_evaluation_records_both_strategies_and_usage_restores_cache(self):
        rag = AsyncMock()
        rag.enable_llm_cache = True
        rag.llm_response_cache.global_config = {'enable_llm_cache': True}
        async def answer(*args, **kwargs):
            usage = ACTIVE_USAGE.get()
            usage.add_usage({'total_tokens': 12})
            return AgentResult('1个月 2周', 'answered', [{'chunk_id':'chunk-'+'a'*32,'content':'1个月 2周','file_path':'book.txt'}])
        app = create_app(lambda: rag, db_path=':memory:')
        with TestClient(app) as client, patch('rag_app.management.ResearchWorkflow.run', side_effect=answer):
            app.state.agent.run = AsyncMock(side_effect=answer)
            case = client.get('/management/evaluation/cases').json()[0]
            job = client.post('/management/evaluation', json={'case_ids':[case['id']]}).json()
            result = self.wait(client, job['id'])
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(len(result['results']), 2)
            self.assertEqual(result['usage']['total_tokens'], 24)
            self.assertEqual([r['total_tokens'] for r in result['results']], [12,12])
            self.assertTrue(rag.enable_llm_cache)
            self.assertTrue(rag.llm_response_cache.global_config['enable_llm_cache'])
            self.assertEqual(client.post('/management/evaluation', json={'case_ids':['missing']}).status_code, 422)
