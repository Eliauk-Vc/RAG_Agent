"""Offline HTTP contract checks: python -m unittest discover -s tests -p test_book_api.py."""

import unittest
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from rag_app.api import create_app


class BookApiTests(unittest.TestCase):
    def test_previous_question_uses_current_session_without_model_or_retrieval(self):
        rag = AsyncMock()
        rag.aquery.return_value = "知识回答"
        with TestClient(create_app(lambda: rag, db_path=":memory:", default_engine="rag")) as client:
            first = client.post("/chat", json={"question": "焦虑症的症状是什么？"}).json()
            other = client.post("/chat", json={"question": "睡眠是什么？"}).json()
            rag.aquery.reset_mock()
            rag.llm_model_func.reset_mock()
            for session, expected in [
                (first, "焦虑症的症状是什么？"),
                (other, "睡眠是什么？"),
            ]:
                result = client.post("/chat", json={
                    "session_id": session["session_id"],
                    "question": "我刚才的问题是什么",
                })
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()["answer"], f"你刚才的问题是：“{expected}”")
            empty = client.post("/chat", json={"question": "我刚才问了什么？"})
            self.assertEqual(empty.json()["answer"], "当前会话中还没有你之前的问题。")
            rag.aquery.assert_not_awaited()
            rag.llm_model_func.assert_not_awaited()

    def test_chat_returns_plain_answer_and_remembers_clean_text(self):
        rag = AsyncMock()
        rag.aquery.return_value = (
            "## Answer\n\n- **First point** [1]\n- **Second point**\n\n"
            "### References\n\n- [1] D:\\private\\book.txt"
        )
        rag.llm_model_func.return_value = '{"query":"follow-up"}'
        with TestClient(create_app(lambda: rag, db_path=":memory:", default_engine="rag")) as client:
            response = client.post("/chat", json={"question": "question"})
            self.assertEqual(response.status_code, 200)
            first = response.json()
            expected = "Answer\n\n• First point\n• Second point"
            self.assertEqual(first["answer"], expected)
            client.post("/chat", json={
                "session_id": first["session_id"], "question": "follow-up"
            })
            history = rag.aquery.await_args.kwargs["param"].conversation_history
            self.assertEqual(history[1]["content"], expected)

    def test_history_isolation_deletion_and_lifecycle(self):
        rag = AsyncMock()
        rag.aquery.return_value = "answer"
        rag.llm_model_func.return_value = '{"query":"rewritten follow-up"}'
        with TestClient(create_app(lambda: rag, db_path=":memory:", default_engine="rag")) as client:
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn("text/html", page.headers["content-type"])
            self.assertIn('id="chat-form"', page.text)
            self.assertEqual(client.get("/health").status_code, 200)
            first = client.post("/chat", json={"question": "first"}).json()
            session_id = first["session_id"]
            for turn in range(2, 14):
                result = client.post("/chat", json={
                    "session_id": session_id, "question": f"turn {turn}"
                })
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()["turns"], turn)
            history = rag.aquery.await_args.kwargs["param"].conversation_history
            self.assertEqual(len(history), 8)
            self.assertEqual(history[0]["content"], "turn 9")
            client.post("/chat", json={"question": "separate"})
            self.assertEqual(
                rag.aquery.await_args.kwargs["param"].conversation_history, []
            )
            self.assertEqual(client.delete(f"/sessions/{session_id}").status_code, 204)
            self.assertEqual(client.post("/chat", json={
                "session_id": session_id, "question": "again"
            }).status_code, 404)
        rag.initialize_storages.assert_awaited_once()
        rag.finalize_storages.assert_awaited_once()

    def test_validation_and_failed_query_preserve_history(self):
        rag = AsyncMock()
        rag.aquery.return_value = "answer"
        rag.llm_model_func.return_value = '{"query":"rewritten follow-up"}'
        with TestClient(create_app(lambda: rag, db_path=":memory:", default_engine="rag")) as client:
            self.assertEqual(client.post("/chat", json={"question": " "}).status_code, 422)
            first = client.post("/chat", json={"question": "first"}).json()
            payload = {"session_id": first["session_id"], "question": "next"}
            rag.aquery.side_effect = RuntimeError("private upstream details")
            failed = client.post("/chat", json=payload)
            self.assertEqual(failed.status_code, 502)
            self.assertNotIn("private upstream details", failed.text)
            rag.aquery.side_effect = None
            result = client.post("/chat", json=payload)
            self.assertEqual(result.json()["turns"], 2)
            self.assertEqual(len(
                rag.aquery.await_args.kwargs["param"].conversation_history
            ), 2)


if __name__ == "__main__":
    unittest.main()
