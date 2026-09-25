import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from rag_app.api import create_app
from rag_app.session_store import SessionStore

A = "chunk-" + "a" * 32


def fake_rag():
    rag = AsyncMock()
    rag.aquery_data.return_value = {"status": "success", "data": {
        "chunks": [{"chunk_id": A, "content": "evidence", "file_path": "book.txt"}]}}
    rag.llm_model_func.side_effect = [
        '{"action":"search_knowledge","arguments":{"query":"question"}}',
        json.dumps({"action": "answer", "evidence_ids": [A]}),
        "**Answer**\n\n- Point\n\n### References\nbook.txt",
    ]
    return rag


class AgentApiTests(unittest.TestCase):
    def test_partial_workflow_is_saved_as_partial_and_keeps_evidence(self):
        rag = fake_rag()
        rag.llm_model_func.side_effect = [
            json.dumps({'optimized_question': '问题', 'subtasks': ['子问题']}),
            '{"action":"search_knowledge","arguments":{"query":"question"}}',
            json.dumps({'action':'answer', 'evidence_ids':[A]}),
            '子答案', '初稿',
            json.dumps({'meets_need':False,'grounded':True,'issues':['缺少例外条件'],'followup':'查询例外'}),
            '已有证据支持的部分']
        with patch.dict('os.environ', {'WORKFLOW_MAX_ROUNDS': '1'}):
            with TestClient(create_app(lambda: rag, db_path=':memory:')) as client:
                identifier = 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb'
                result = client.post('/chat', json={'question':'问题','workflow':True,'monitor_id':identifier}).json()
                self.assertEqual(result['stop_reason'], 'partial')
                self.assertTrue(result['committed'])
                self.assertIn('缺少例外条件', result['answer'])
                state = client.get('/activity/' + identifier).json()
                self.assertEqual(state['phase'], 'partial')
                self.assertEqual(state['llm_token_budget'], 50000)
                self.assertEqual(state['embedding_token_budget'], 250000)

    def test_workflow_chat_and_monitor(self):
        rag = fake_rag()
        rag.llm_model_func.side_effect = [
            json.dumps({'optimized_question': '问题', 'subtasks': ['子问题']}),
            '{"action":"search_knowledge","arguments":{"query":"question"}}',
            json.dumps({'action':'answer', 'evidence_ids':[A]}),
            '子答案', '汇总答案',
            json.dumps({'meets_need':True,'grounded':True,'issues':[],'followup':''})]
        with TestClient(create_app(lambda: rag, db_path=':memory:')) as client:
            identifier = 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'
            result = client.post('/chat', json={'question':'问题','workflow':True,'monitor_id':identifier}).json()
            self.assertEqual(result['answer'], '汇总答案')
            state = client.get('/activity/' + identifier).json()
            self.assertEqual(state['phase'], 'completed')
            self.assertEqual(len(state['steps']), 1)
            self.assertTrue(any(e['action']=='search_knowledge' for e in state['events']))

    def test_unfinished_run_is_marked_interrupted_and_progress_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.sqlite3"
            store = SessionStore(path)
            run_id = store.start_run("session", "question", "agent")
            store.progress("session", run_id, "question", [
                {"action": "search_knowledge", "status": "ok", "evidence_ids": [A]}], {})
            store.close()
            store = SessionStore(path)
            self.assertEqual(store.run("session", run_id)["status"], "interrupted")
            self.assertEqual(len(store.run("session", run_id)["trace"]), 1)
            self.assertEqual(store.history("session"), [])
            store.close()

    def test_default_agent_persists_restart_recall_trace_and_cascade_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sessions.sqlite3"
            with TestClient(create_app(fake_rag, db_path=path)) as client:
                response = client.post("/chat", json={"question": "original question"})
                self.assertEqual(response.status_code, 200)
                first = response.json()
                self.assertEqual(first["engine"], "agent")
                self.assertEqual(first["answer"], "Answer\n\n• Point")
                self.assertTrue(first["committed"])
                route = f"/sessions/{first['session_id']}/runs/{first['run_id']}"
                run = client.get(route).json()
                self.assertEqual(run["evidence"][0]["chunk_id"], A)
                self.assertEqual(run["status"], "answered")
                self.assertIn("usage", run)
                self.assertEqual(client.get(f"/sessions/other/runs/{first['run_id']}").status_code, 404)
            with TestClient(create_app(fake_rag, db_path=path)) as client:
                history = client.get(f"/sessions/{first['session_id']}").json()
                self.assertEqual(history["turns"], 1)
                recall = client.post("/chat", json={"session_id": first["session_id"],
                                                    "question": "我刚才的问题是什么？"}).json()
                self.assertIn("original question", recall["answer"])
                self.assertEqual(recall["stop_reason"], "recall")
                self.assertEqual(recall["turns"], 2)
                self.assertEqual(client.delete(f"/sessions/{first['session_id']}").status_code, 204)
                self.assertEqual(client.get(route).status_code, 404)
                self.assertEqual(client.get(f"/sessions/{first['session_id']}").status_code, 404)
            store = SessionStore(path)
            self.assertEqual(store.db.execute("SELECT count(*) FROM messages").fetchone()[0], 0)
            self.assertEqual(store.db.execute("SELECT count(*) FROM runs").fetchone()[0], 0)
            store.close()

    def test_engine_switch_and_failure_does_not_pollute_history(self):
        rag = fake_rag()
        rag.aquery.return_value = "baseline answer"
        with TestClient(create_app(lambda: rag, db_path=":memory:")) as client:
            first = client.post("/chat", json={"question": "q", "engine": "rag"}).json()
            self.assertEqual(first["answer"], "baseline answer")
            rag.llm_model_func.side_effect = RuntimeError("secret-key")
            failed = client.post("/chat", json={"question": "new topic", "session_id": first["session_id"]}).json()
            self.assertFalse(failed["committed"])
            self.assertEqual(failed["turns"], 1)
            self.assertNotIn("secret-key", json.dumps(failed))
            messages = client.get(f"/sessions/{first['session_id']}").json()["messages"]
            self.assertEqual(len(messages), 2)
            run = client.get(f"/sessions/{first['session_id']}/runs/{failed['run_id']}").json()
            self.assertEqual(run["status"], "model_error")
            self.assertNotIn("secret-key", json.dumps(run))

    def test_validation_happens_before_model_calls(self):
        rag = fake_rag()
        with TestClient(create_app(lambda: rag, db_path=":memory:")) as client:
            self.assertEqual(client.post("/chat", json={"question": "q", "engine": "unknown"}).status_code, 422)
            self.assertEqual(client.post("/chat", json={"question": "字" * 5000}).status_code, 422)
            rag.llm_model_func.assert_not_awaited()
