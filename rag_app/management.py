"""Local knowledge management and repeatable quality evaluation jobs."""

import asyncio
from dataclasses import asdict, is_dataclass
from hashlib import md5
import json
import time
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from rag_app.config import REPO_ROOT
from rag_app.telemetry import ACTIVE_USAGE, RequestUsage
from rag_app.workflow import ResearchWorkflow


class DocumentInput(BaseModel):
    name: str = Field(min_length=1, max_length=150)
    content: str = Field(min_length=1, max_length=100000)
    replaces: str | None = Field(default=None, pattern=r'^doc-[a-f0-9]{32}$')

    @field_validator('name', 'content')
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError('内容不能为空')
        return value.strip()


class EvaluationInput(BaseModel):
    case_ids: list[str] = Field(min_length=1, max_length=10)


class Management:
    def __init__(self, app):
        self.app, self.jobs = app, {}
        self.db = app.state.store.db
        self.db.execute('CREATE TABLE IF NOT EXISTS management_jobs (id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
        for row in self.db.execute('SELECT payload FROM management_jobs').fetchall():
            job = json.loads(row['payload'])
            if job['status'] in {'queued', 'running'}:
                job.update(status='interrupted', message='服务中断，请检查文档状态后重新执行')
                self.save(job)

    def save(self, job):
        with self.db:
            self.db.execute('INSERT INTO management_jobs VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
                            (job['id'], json.dumps(job, ensure_ascii=False)))

    def get(self, identifier):
        row = self.db.execute('SELECT payload FROM management_jobs WHERE id=?', (identifier,)).fetchone()
        if not row:
            raise HTTPException(404, '操作不存在')
        return json.loads(row['payload'])

    def submit(self, kind, work):
        # Bound the background queue in this single-user application.
        if len(self.jobs) >= 5:
            raise HTTPException(429, '后台操作较多，请稍后再试')
        job = {'id': str(uuid4()), 'kind': kind, 'status': 'queued', 'usage': {}, 'results': []}
        self.save(job)
        task = asyncio.create_task(self.execute(job, work))
        self.jobs[job['id']] = task
        task.add_done_callback(lambda _: self.jobs.pop(job['id'], None))
        return job

    async def execute(self, job, work):
        usage = RequestUsage()
        token = ACTIVE_USAGE.set(usage)
        async def monitor():
            while True:
                job['usage'] = usage.get_usage()
                self.save(job)
                await asyncio.sleep(1)
        pulse = asyncio.create_task(monitor())
        try:
            async with self.app.state.query_lock:
                job['status'] = 'running'
                self.save(job)
                await work(job)
                job['status'] = 'completed'
        except asyncio.CancelledError:
            job['status'] = 'interrupted'
            raise
        except Exception as exc:
            job.update(status='failed', error_type=type(exc).__name__,
                       message='操作未完成。请检查文档状态或评估结果后重试。')
        finally:
            pulse.cancel()
            await asyncio.gather(pulse, return_exceptions=True)
            job['usage'] = usage.get_usage()
            ACTIVE_USAGE.reset(token)
            self.save(job)

    async def close(self):
        tasks = list(self.jobs.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def plain_object(row):
    if is_dataclass(row):
        return asdict(row)
    if hasattr(row, 'model_dump'):
        return row.model_dump()
    return dict(row)


def management_router(app):
    router = APIRouter(prefix='/management')

    @router.get('/jobs')
    async def jobs():
        return [json.loads(row['payload']) for row in app.state.store.db.execute(
            'SELECT payload FROM management_jobs ORDER BY rowid DESC LIMIT 30')]

    @router.get('/jobs/{identifier}')
    async def job(identifier: str):
        return app.state.management.get(identifier)

    @router.get('/documents')
    async def documents(page: int = Query(1, ge=1)):
        rows, total = await app.state.rag.doc_status.get_docs_paginated(page=page, page_size=50)
        return {'total': total, 'page': page, 'documents': [
            {'id': identifier, **{k: v for k, v in plain_object(row).items()
                                if k in {'file_path', 'status', 'chunks_count', 'created_at', 'updated_at'}}}
            for identifier, row in rows]}

    @router.post('/documents')
    async def insert(data: DocumentInput):
        rag = app.state.rag
        if data.replaces and not await rag.doc_status.get_by_id(data.replaces):
            raise HTTPException(404, '被替换文档不存在')
        identifier = 'doc-' + md5(data.content.encode('utf-8')).hexdigest()
        name = data.name.replace('\\', '/').rsplit('/', 1)[-1]
        async def work(job):
            job.update(document_id=identifier, name=name)
            for attempt in range(2):
                job['attempt'] = attempt + 1
                app.state.management.save(job)
                try:
                    await rag.ainsert(data.content, ids=identifier, file_paths=name)
                    row = await rag.doc_status.get_by_id(identifier)
                    if not row or row.get('status') != 'processed':
                        raise RuntimeError('DocumentNotProcessed')
                    break
                except Exception:
                    if attempt == 1:
                        raise
                    await asyncio.sleep(1)
            # Replace only after the new index has been verified. Old document survives import failures.
            if data.replaces and data.replaces != identifier:
                result = await rag.adelete_by_doc_id(data.replaces)
                if result.status != 'success':
                    raise RuntimeError('OldDocumentDeletionFailed')
            await rag.llm_response_cache.delete_query_cache()
            job['message'] = '文档已建立索引，查询缓存已清理'
        return app.state.management.submit('document_import', work)

    @router.delete('/documents/{identifier}')
    async def delete(identifier: str):
        if not await app.state.rag.doc_status.get_by_id(identifier):
            raise HTTPException(404, '文档不存在')
        async def work(job):
            result = await app.state.rag.adelete_by_doc_id(identifier)
            if result.status != 'success':
                raise RuntimeError('DocumentDeletionFailed')
            await app.state.rag.llm_response_cache.delete_query_cache()
            job.update(document_id=identifier, message='文档及关联索引已删除，查询缓存已清理')
        return app.state.management.submit('document_delete', work)

    def cases():
        return json.loads((REPO_ROOT / 'data/book_eval_dataset.json').read_text(encoding='utf-8-sig'))['cases']

    @router.get('/evaluation/cases')
    async def evaluation_cases():
        return cases()

    @router.post('/evaluation')
    async def evaluate(data: EvaluationInput):
        from rag_app.evaluate import EvaluationCase, EvaluationStrategy, score_result
        selected = {row['id']: row for row in cases()}
        if any(key not in selected for key in data.case_ids):
            raise HTTPException(422, '评估题目不存在')
        async def work(job):
            rag = app.state.rag
            old_cache = rag.enable_llm_cache
            old_global = rag.llm_response_cache.global_config.get('enable_llm_cache', True)
            rag.enable_llm_cache = False
            rag.llm_response_cache.global_config['enable_llm_cache'] = False
            job['cache_policy'] = 'cold'
            try:
                for key in dict.fromkeys(data.case_ids):
                    row = selected[key]
                    case = EvaluationCase(key, row['question'], tuple(row['expected_keywords']), row['expected_source'])
                    for strategy in ('agent', 'workflow'):
                        job.update(current_case=case.question, current_strategy=strategy, current_trace=[])
                        usage = ACTIVE_USAGE.get()
                        before = usage.get_usage()
                        started = time.monotonic()
                        try:
                            runner = app.state.agent if strategy == 'agent' else ResearchWorkflow(app.state.agent)
                            result = await runner.run(case.question, on_event=lambda trace: job.update(current_trace=list(trace)))
                            scored = score_result(case, EvaluationStrategy(strategy, 'hybrid', True),
                                                  result.evaluation_payload(), time.monotonic() - started,
                                                  {key: value - before.get(key, 0) for key, value in usage.get_usage().items()
                                                   if isinstance(value, int) and not isinstance(value, bool)})
                            scored.update(stop_reason=result.stop_reason, fallback=result.stop_reason == 'fallback')
                            scored['tool_calls'] = sum(e.get('action') in {'search_knowledge','read_source','query_graph'}
                                                      and e.get('status') not in {'duplicate','budget_blocked'} for e in result.trace)
                            scored['loop_blocks'] = sum(e.get('action') == 'loop_guard' or e.get('status') == 'duplicate'
                                                       for e in result.trace)
                            job['results'].append(scored)
                        except Exception as exc:
                            job['results'].append({'case_id': key, 'strategy': strategy, 'status': 'failure',
                                                   'error_type': type(exc).__name__})
                        app.state.management.save(job)
            finally:
                rag.enable_llm_cache = old_cache
                rag.llm_response_cache.global_config['enable_llm_cache'] = old_global
        return app.state.management.submit('evaluation', work)

    return router
