"""Persistent terminal entry point for the knowledge retrieval agent."""

import argparse
import asyncio
from uuid import uuid4

from rag_app.agent import RetrievalAgent
from rag_app.answer_text import plain_answer
from rag_app.api import build_rag
from rag_app.config import SESSION_DB
from rag_app.runtime import EXIT_COMMANDS, configure_console
from rag_app.session_store import SessionStore
from rag_app.lifecycle import finalize_rag
from rag_app.agent_types import AgentResult
from rag_app.telemetry import ACTIVE_USAGE, RequestUsage


async def chat(session_id=None):
    store = SessionStore(SESSION_DB)
    rag = None
    try:
        if session_id and not store.exists(session_id):
            raise ValueError("Session not found")
        session_id = session_id or str(uuid4())
        rag = build_rag()
        await rag.initialize_storages()
        agent = RetrievalAgent(rag)
        print(f"Session: {session_id}")
        print("/exit stops; /clear deletes this session and its execution records.")
        while True:
            try:
                question = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if question.lower() in EXIT_COMMANDS:
                break
            if question == "/clear":
                store.delete(session_id)
                session_id = str(uuid4())
                print(f"New session: {session_id}")
                continue
            if not question:
                continue
            usage = RequestUsage()
            token = ACTIVE_USAGE.set(usage)
            run_id = store.start_run(session_id, question, "agent")
            try:
                result = await agent.run(question, store.history(session_id),
                    on_event=lambda trace: store.progress(session_id, run_id, question, trace, usage.get_usage()))
                result.answer = plain_answer(result.answer)
                result.usage = usage.get_usage()
                committed = result.stop_reason in {"answered", "budget_answer", "recall", "clarify", "insufficient"}
                store.save_turn(session_id, question, result, "agent", committed, run_id)
                print("\nAssistant:\n" + result.answer)
            except ValueError as exc:
                store.save_turn(session_id, question, AgentResult("", "invalid_input"),
                                "agent", commit_messages=False, run_id=run_id)
                print(f"Input rejected: {exc}")
            finally:
                ACTIVE_USAGE.reset(token)
    finally:
        try:
            if rag is not None:
                await finalize_rag(rag)
        finally:
            store.close()


def main():
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", help="Resume a persisted session")
    args = parser.parse_args()
    asyncio.run(chat(args.session_id))


if __name__ == "__main__":
    main()
