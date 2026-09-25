"""Interactive, persisted research tasks for the local single-process service."""

import asyncio
import json
from uuid import uuid4
from typing import Literal

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from rag_app.answer_text import plain_answer
from rag_app.telemetry import ACTIVE_USAGE, RequestUsage


class TaskDraft(BaseModel):
    title: str = Field(min_length=1, max_length=120)
    topics: list[str] = Field(min_length=1, max_length=5)
    instructions: str = Field(default="", max_length=1000)
    parent_id: str | None = None
    workflow: bool = False
    question: str = Field(default='', max_length=4000)

    @field_validator("title")
    @classmethod
    def title_not_blank(cls, value):
        if not value.strip():
            raise ValueError("主题不能为空")
        return value.strip()

    @field_validator("topics")
    @classmethod
    def valid_topics(cls, value):
        if any(not item.strip() or len(item) > 500 for item in value):
            raise ValueError("每项任务需为 1–500 字")
        return list(dict.fromkeys(item.strip() for item in value))


class ResearchTasks:
    def __init__(self, app):
        self.app = app
        self.jobs = {}
        self.db = app.state.store.db
        self.db.execute("""CREATE TABLE IF NOT EXISTS research_tasks (
            id TEXT PRIMARY KEY, payload TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
        for row in self.db.execute("SELECT id,payload FROM research_tasks").fetchall():
            task = json.loads(row["payload"])
            if task["status"] in {"queued", "running"}:
                task["status"] = "interrupted"
                for step in task["steps"]:
                    if step["status"] == "running":
                        step["status"] = "interrupted"
                self.save(task)

    def save(self, task):
        with self.db:
            self.db.execute("INSERT INTO research_tasks(id,payload) VALUES (?,?) "
                            "ON CONFLICT(id) DO UPDATE SET payload=excluded.payload",
                            (task["id"], json.dumps(task, ensure_ascii=False)))

    def get(self, identifier):
        row = self.db.execute("SELECT payload FROM research_tasks WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise HTTPException(404, "任务不存在")
        return json.loads(row["payload"])

    def create(self, draft):
        if draft.parent_id:
            parent = self.get(draft.parent_id)
            if parent["status"] in {"queued", "running"}:
                raise HTTPException(409, "请先完成或取消原任务")
        task = {"id": str(uuid4()), **draft.model_dump(), "status": "draft",
                "steps": [{"topic": topic, "status": "pending", "answer": "", "trace": []}
                          for topic in draft.topics]}
        self.save(task)
        return task

    def start(self, identifier):
        if len(self.jobs) >= 5:
            raise HTTPException(429, '后台任务较多，请稍后再试')
        task = self.get(identifier)
        if task["status"] not in {"draft", "failed", "cancelled", "interrupted", "partial"}:
            raise HTTPException(409, "此任务不能重复启动")
        task["status"] = "queued"
        self.save(task)
        job = asyncio.create_task(self.execute(identifier))
        self.jobs[identifier] = job
        job.add_done_callback(lambda finished: self.jobs.pop(identifier, None))
        return task

    async def execute(self, identifier):
        task = self.get(identifier)
        try:
            async with self.app.state.query_lock:
                if task.get('workflow'):
                    from rag_app.workflow import ResearchWorkflow
                    task['status'] = 'running'
                    usage = RequestUsage()
                    token = ACTIVE_USAGE.set(usage)
                    def update(state):
                        task['workflow_state'] = state
                        task['steps'] = state['steps']
                        task['summary'] = state.get('summary', '')
                        task['usage'] = usage.get_usage()
                        self.save(task)
                    try:
                        result = await ResearchWorkflow(self.app.state.agent).run(
                            task.get('question') or task['title'], topics=task['topics'],
                            instructions=task['instructions'], on_state=update)
                        task.update(summary=result.answer,
                                    status='completed' if result.stop_reason == 'answered' else
                                    'completed_with_fallback' if result.stop_reason == 'fallback' else
                                    'partial' if result.stop_reason == 'partial' else 'failed',
                                    stop_reason=result.stop_reason)
                    finally:
                        ACTIVE_USAGE.reset(token)
                        self.save(task)
                    return
                task["status"] = "running"
                self.save(task)
                for step in task["steps"]:
                    if step["status"] == "completed":
                        continue
                    step.update(status="running", trace=[], answer="")
                    self.save(task)
                    usage = RequestUsage()
                    token = ACTIVE_USAGE.set(usage)
                    def progress(trace):
                        step["trace"] = list(trace)
                        step["usage"] = usage.get_usage()
                        self.save(task)
                    try:
                        question = (f"请完成专题资料整理中的一个部分。总主题：{task['title']}\n"
                                    f"本部分任务：{step['topic']}\n整理要求：{task['instructions']}\n"
                                    "检索知识库后交付可直接阅读的整理正文，明确资料缺失；"
                                    "不要描述准备做什么，不要编造资料。")
                        result = await self.app.state.agent.run(question, on_event=progress)
                    finally:
                        ACTIVE_USAGE.reset(token)
                    step.update(answer=plain_answer(result.answer), trace=list(result.trace),
                                evidence=result.evidence, usage=usage.get_usage(),
                                stop_reason=result.stop_reason,
                                status="completed" if result.stop_reason == "answered" else "needs_attention")
                    self.save(task)
                task["status"] = ("completed" if all(s["status"] == "completed" for s in task["steps"])
                                  else "partial")
        except asyncio.CancelledError:
            task["status"] = "cancelled"
            if task.get('workflow_state'):
                task['workflow_state']['phase'] = 'cancelled'
            for step in task["steps"]:
                if step["status"] == "running":
                    step["status"] = "cancelled"
            raise
        except Exception as exc:
            task["status"] = "failed"
            for step in task["steps"]:
                if step["status"] == "running":
                    step.update(status="failed", error_type=type(exc).__name__)
        finally:
            self.save(task)

    async def cancel(self, identifier):
        task = self.get(identifier)
        job = self.jobs.get(identifier)
        if job:
            job.cancel()
            await asyncio.gather(job, return_exceptions=True)
        # A queued task may be cancelled before its coroutine starts.
        task = self.get(identifier)
        if task["status"] in {"queued", "running"}:
            task["status"] = "cancelled"
            self.save(task)
        return self.get(identifier)

    async def close(self):
        for identifier in list(self.jobs):
            await self.cancel(identifier)


def task_router(app):
    router = APIRouter(prefix="/research-tasks")

    @router.post("")
    async def create(draft: TaskDraft):
        return app.state.research.create(draft)

    @router.get("")
    async def listing():
        rows = app.state.store.db.execute(
            "SELECT payload FROM research_tasks ORDER BY rowid DESC LIMIT 50").fetchall()
        return [{key: task[key] for key in ("id", "title", "status", "parent_id")}
                for row in rows if (task := json.loads(row["payload"]))]

    @router.get("/{identifier}")
    async def get(identifier: str):
        return app.state.research.get(identifier)

    @router.post("/{identifier}/start")
    async def start(identifier: str):
        return app.state.research.start(identifier)

    @router.post("/{identifier}/cancel")
    async def cancel(identifier: str):
        return await app.state.research.cancel(identifier)

    @router.delete("/{identifier}", status_code=204)
    async def delete(identifier: str):
        await app.state.research.cancel(identifier)
        with app.state.store.db:
            app.state.store.db.execute("DELETE FROM research_tasks WHERE id=?", (identifier,))
        return Response(status_code=204)

    @router.get("/{identifier}/export")
    async def export(identifier: str, format: Literal['txt', 'docx'] = 'txt'):
        task = app.state.research.get(identifier)
        if not task.get('summary') and not any(step.get("answer") for step in task["steps"]):
            raise HTTPException(409, "尚无可导出的内容")
        if format == 'docx':
            from rag_app.word_export import word_report
            return Response(word_report(task),
                            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                            headers={'Content-Disposition': f'attachment; filename="research-{task["id"]}.docx"'})
        lines = [task["title"], ""]
        if task.get('summary'):
            lines.extend(['汇总答案', task['summary'], '', '子任务检索记录', ''])
        if task["status"] != "completed":
            lines.extend(["本资料尚未全部完成，部分内容需要补充或重试。", ""])
        for index, step in enumerate(task["steps"], 1):
            lines.extend([f"{index}、{step['topic']}", step.get("answer") or "此部分尚未完成。", ""])
        return Response("\ufeff" + "\n".join(lines), media_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="research-{task["id"]}.txt"'})

    return router
