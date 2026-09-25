import asyncio
import tempfile
import time
import unittest
from io import BytesIO
from zipfile import ZipFile
from xml.etree import ElementTree as ET
from pathlib import Path
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from rag_app.agent_types import AgentResult
from rag_app.api import create_app


class ResearchTaskTests(unittest.TestCase):
    def wait_finished(self, client, identifier):
        for _ in range(100):
            task = client.get('/research-tasks/' + identifier).json()
            if task['status'] not in {'running', 'queued'}:
                return task
            time.sleep(.01)
        self.fail('Task did not finish')

    def test_complete_export_restart_revision_delete(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sessions.sqlite3'
            app = create_app(lambda: AsyncMock(), db_path=path)
            with TestClient(app) as client:
                app.state.agent.run = AsyncMock(return_value=AgentResult('**正文**\n- 要点', 'answered'))
                task = client.post('/research-tasks', json={
                    'title': '专题', 'topics': ['概念', '区别']}).json()
                identifier = task['id']
                self.assertEqual(task['status'], 'draft')
                app.state.agent.run.assert_not_awaited()
                self.assertEqual(client.get(f'/research-tasks/{identifier}/export').status_code, 409)
                client.post(f'/research-tasks/{identifier}/start')
                result = self.wait_finished(client, identifier)
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(app.state.agent.run.await_count, 2)
                self.assertEqual(result['steps'][0]['answer'], '正文\n• 要点')
                self.assertEqual(client.post(f'/research-tasks/{identifier}/start').status_code, 409)
                export = client.get(f'/research-tasks/{identifier}/export')
                self.assertIn('正文', export.text)
                self.assertIn('attachment', export.headers['content-disposition'])
                word = client.get(f'/research-tasks/{identifier}/export?format=docx')
                self.assertEqual(word.status_code, 200)
                self.assertIn('wordprocessingml', word.headers['content-type'])
                with ZipFile(BytesIO(word.content)) as archive:
                    xml = ET.fromstring(archive.read('word/document.xml'))
                    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
                    self.assertIn('正文', ''.join(xml.itertext()))
                    anchors = [e.attrib['{' + ns['w'] + '}anchor'] for e in xml.findall('.//w:hyperlink', ns)]
                    bookmarks = [e.attrib['{' + ns['w'] + '}name'] for e in xml.findall('.//w:bookmarkStart', ns)]
                    self.assertEqual(anchors, bookmarks)
                    self.assertEqual(len(anchors), 2)
                self.assertEqual(client.get(f'/research-tasks/{identifier}/export?format=pdf').status_code, 422)
            with TestClient(create_app(lambda: AsyncMock(), db_path=path)) as client:
                self.assertEqual(client.get(f'/research-tasks/{identifier}').json()['status'], 'completed')
                revision = client.post('/research-tasks', json={
                    'title': '专题', 'topics': ['概念'], 'instructions': '更简洁', 'parent_id': identifier}).json()
                self.assertNotEqual(revision['id'], identifier)
                self.assertEqual(revision['parent_id'], identifier)
                self.assertEqual(revision['status'], 'draft')
                self.assertEqual(client.delete(f'/research-tasks/{identifier}').status_code, 204)
                self.assertEqual(client.get(f'/research-tasks/{identifier}').status_code, 404)
                self.assertEqual(client.get(f'/research-tasks/{revision["id"]}').status_code, 200)

    def test_cancel_visible_progress_and_resume_keeps_completed_sections(self):
        app = create_app(lambda: AsyncMock(), db_path=':memory:')
        async def slow_run(question, on_event):
            on_event([{'action': 'search_knowledge', 'status': 'ok'}])
            await asyncio.sleep(30)
            return AgentResult('never', 'answered')
        with TestClient(app) as client:
            task = client.post('/research-tasks', json={'title': 't', 'topics': ['a', 'b']}).json()
            identifier = task['id']
            app.state.agent.run = AsyncMock(side_effect=[AgentResult('first', 'answered'), AgentResult('不足', 'insufficient')])
            client.post(f'/research-tasks/{identifier}/start')
            self.assertEqual(self.wait_finished(client, identifier)['status'], 'partial')
            app.state.agent.run = slow_run
            client.post(f'/research-tasks/{identifier}/start')
            for _ in range(100):
                running = client.get(f'/research-tasks/{identifier}').json()
                if running['steps'][1]['trace']:
                    break
                time.sleep(.01)
            self.assertTrue(running['steps'][1]['trace'])
            self.assertEqual(client.post(f'/research-tasks/{identifier}/start').status_code, 409)
            cancelled = client.post(f'/research-tasks/{identifier}/cancel').json()
            self.assertEqual(cancelled['status'], 'cancelled')
            self.assertEqual(cancelled['steps'][0]['answer'], 'first')
            app.state.agent.run = AsyncMock(return_value=AgentResult('second', 'answered'))
            client.post(f'/research-tasks/{identifier}/start')
            self.assertEqual(self.wait_finished(client, identifier)['status'], 'completed')
            self.assertEqual(app.state.agent.run.await_count, 1)

    def test_validation_and_interrupted_recovery(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'sessions.sqlite3'
            app = create_app(lambda: AsyncMock(), db_path=path)
            with TestClient(app) as client:
                for topics in [[], [' '], ['a'] * 6, ['a' * 501]]:
                    self.assertEqual(client.post('/research-tasks', json={'title': 't', 'topics': topics}).status_code, 422)
                task = client.post('/research-tasks', json={'title': 't', 'topics': ['a']}).json()
                task['status'] = 'running'
                task['steps'][0]['status'] = 'running'
                app.state.research.save(task)
            with TestClient(create_app(lambda: AsyncMock(), db_path=path)) as client:
                restored = client.get('/research-tasks/' + task['id']).json()
                self.assertEqual(restored['status'], 'interrupted')
                self.assertEqual(restored['steps'][0]['status'], 'interrupted')
                self.assertEqual(client.get('/tasks').status_code, 200)


if __name__ == '__main__':
    unittest.main()
