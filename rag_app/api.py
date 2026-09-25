"""Local HTTP API for the existing book index; run with a single worker."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Literal
from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, field_validator

from rag_app.answer_text import plain_answer
from rag_app.agent import RetrievalAgent
from rag_app.agent_types import AgentResult, AgentSettings
from rag_app.context import recent_history
from rag_app.session_store import SessionStore
from rag_app.lifecycle import finalize_rag
from rag_app.telemetry import ACTIVE_USAGE, RequestUsage
from rag_app.tasks import ResearchTasks, task_router
from rag_app.management import Management, management_router
from rag_app.runtime import (
    LightRAG,
    default_llm_model,
    make_embedding_func,
    make_embedding_rerank_func,
    make_llm_func,
    make_local_tokenizer,
    query_rag,
)
from lightrag.utils import logger
from rag_app.config import INDEX_DIR, SESSION_DB, WORKSPACE


CHAT_PAGE = """<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LightRAG 知识库对话</title>
<style>
body { margin:0; background:#f5f7fb; color:#172338; font:16px/1.6 system-ui; }
main { max-width:850px; margin:32px auto; padding:0 20px; }
h1 { font-size:26px; margin-bottom:4px; }
#messages { min-height:240px; max-height:55vh; overflow:auto; margin:20px 0; }
.message { white-space:pre-wrap; overflow-wrap:anywhere; background:white;
  border:1px solid #dce3ed; border-radius:12px; padding:16px; margin:12px 0; }
.user { background:#e9f1ff; }
.assistant { white-space:normal; line-height:1.85; }
.message-label { margin-bottom:8px; }
.answer-body p { margin:0 0 14px; text-indent:2em; }
.answer-body ul { margin:0 0 14px; padding-left:1.6em; list-style-type:disc; }
.answer-body li { margin:6px 0; padding-left:0.2em; }
.answer-body > :last-child { margin-bottom:0; }
textarea { box-sizing:border-box; width:100%; min-height:95px; padding:12px;
  border:1px solid #8a9ab1; border-radius:8px; font:inherit; }
button { padding:10px 18px; margin:8px 8px 0 0; border:0; border-radius:8px;
  background:#2257c7; color:white; font:inherit; cursor:pointer; }
button:disabled { opacity:.5; cursor:wait; }
#end { background:#536175; } #status { min-height:26px; color:#455675; }
select { font:inherit; padding:5px; margin:8px 0; }
* { box-sizing:border-box; }
body { background:#f3f6fa; font-size:15px; }
main { max-width:1000px; margin:28px auto; }
h1 { font-size:28px; margin:22px 0 6px; letter-spacing:-.5px; }
main > a { display:inline-block; color:#315ddd; text-decoration:none;
  background:#eaf0ff; padding:9px 14px; border-radius:8px; font-size:13px; }
#messages { background:#eef2f8; padding:10px 18px; border-radius:16px;
  min-height:300px; max-height:52vh; border:1px solid #e0e7f1; }
.message { border:1px solid #e3e9f2; padding:20px 24px; margin:14px 0; }
.user { background:#e6edff; margin-left:12%; }
.assistant { margin-right:4%; }
.message-label { font-size:12px; font-weight:700; color:#657899; }
#chat-form { background:white; border:1px solid #e3e9f2; border-radius:16px; padding:20px 24px; }
textarea { border-color:#d8e1ee; background:#fbfcfe; margin-top:6px; }
textarea:focus { outline:3px solid #315ddd18; border-color:#315ddd; }
button { background:#315ddd; font-size:14px; }
#end { background:#edf1f7; color:#536175; }
select { border:1px solid #d8e1ee; border-radius:8px; margin:0 0 14px 10px; font-size:13px; }
#status,main > small { font-size:12px; color:#738197; }
.topbar { background:white; border-bottom:1px solid #e3e9f2; padding:18px max(20px,calc((100vw - 960px)/2)); display:flex; justify-content:space-between; align-items:center; gap:16px; }
.brand { font-size:18px; font-weight:750; }
.brand span { display:inline-grid; place-items:center; width:32px; height:32px; background:#315ddd; color:white; border-radius:9px; margin-right:10px; }
nav a { color:#738197; text-decoration:none; padding:8px 12px; border-radius:8px; font-size:14px; }
nav .active { background:#edf2ff; color:#315ddd; }
@media(max-width:600px) { .topbar { flex-wrap:wrap; } main { padding:0 14px; } #messages { padding:6px; } .message { padding:16px; } #chat-form { padding:16px; } }
</style>
<header class="topbar"><div class="brand"><span>K</span>知识资料工作台</div><nav aria-label="主导航"><a class="active" href="/" aria-current="page">知识问答</a><a href="/tasks">专题任务</a><a href="/manage">知识库与评估</a></nav></header>
<main>
<h1>LightRAG 知识库对话</h1>
<a href="/tasks">进入专题资料工作台：制定任务、查看进度、导出资料 →</a>
<p></p>
<div id="messages" role="log" aria-label="聊天记录"></div>
<form id="chat-form">
<label for="engine">问答方式</label>
<select id="engine"><option value="agent">自主检索 Agent</option><option value="rag">普通 RAG</option></select><br>
<label style="display:block;font-size:13px"><input type="checkbox" id="workflow" checked> 自动优化、拆分与汇总检查（复杂问题耗时较长）</label><label for="question">你的问题</label>
<textarea id="question" required maxlength="10000"
 placeholder="请输入问题，Enter 发送，Shift+Enter 换行"></textarea>
<button id="send" type="submit">发送</button>
<button id="end" type="button">结束对话 / 清空记忆</button>
</form>
<p id="status" role="status" aria-live="polite">可以开始提问。</p>
<small>对话与执行记录保存在本机。刷新本标签页可恢复对话；结束对话会删除本次聊天和执行记录。</small>
</main>
<script>
const form = document.getElementById('chat-form');
const question = document.getElementById('question');
const messages = document.getElementById('messages');
const status = document.getElementById('status');
const send = document.getElementById('send');
const end = document.getElementById('end');
const engine = document.getElementById('engine');
let sessionId = sessionStorage.getItem('rag-session-id');
let busy = false;
const offlineMessage = '服务器未运行或连接已断开，请启动服务器后重试。你输入的问题已保留。';
async function checkServer() {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch('/health', {
      cache: 'no-store', signal: controller.signal
    });
    if (!response.ok) throw new Error('服务器尚未就绪，请稍后重试。');
  } finally {
    clearTimeout(timer);
  }
}
function setBusy(value) {
  busy = value;
  send.disabled = end.disabled = question.disabled = engine.disabled = value;
}
function addMessage(role, text) {
  const item = document.createElement('div');
  item.className = 'message ' + role;
  if (role === 'assistant') {
    const label = document.createElement('div');
    label.className = 'message-label';
    label.textContent = '助手：';
    const body = document.createElement('div');
    body.className = 'answer-body';
    let paragraph = [];
    let list = null;
    function flushParagraph() {
      if (!paragraph.length) return;
      const p = document.createElement('p');
      p.textContent = paragraph.join(' ');
      body.appendChild(p);
      paragraph = [];
    }
    for (const rawLine of text.split('\\n')) {
      const line = rawLine.trim();
      if (!line) {
        flushParagraph();
      } else if (line.startsWith('• ')) {
        flushParagraph();
        if (!list) {
          list = document.createElement('ul');
          body.appendChild(list);
        }
        const entry = document.createElement('li');
        entry.textContent = line.slice(2);
        list.appendChild(entry);
      } else {
        list = null;
        paragraph.push(line);
      }
    }
    flushParagraph();
    item.append(label, body);
  } else {
    item.textContent = '你：' + '\\n' + text;
  }
  messages.appendChild(item);
  messages.scrollTop = messages.scrollHeight;
  return item;
}
form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const text = question.value.trim();
  if (busy || !text) return;
  setBusy(true);
  status.textContent = '正在检索资料并生成回答，请稍候……';
  const bubble = addMessage('user', text);
  let monitorTimer = null;
  try {
    await checkServer();
    const payload = {question: text, engine: engine.value, workflow: document.getElementById('workflow').checked && engine.value === 'agent'};
    if (payload.workflow) {
      payload.monitor_id = crypto.randomUUID();
      monitorTimer = setInterval(async () => {
        try {
          const r = await fetch('/activity/' + payload.monitor_id);
          if (!r.ok) return;
          const w = await r.json(), u = w.usage || {};
          if (!busy) return;
          const stages = {partial:'部分完成',planning:'优化拆分',retrieving:'子任务检索',synthesizing:'汇总',reviewing:'质量自评',fallback:'通用知识兜底',completed:'已完成',completed_with_fallback:'兜底完成'};
          const events = (w.steps || []).flatMap(s => s.trace || []);
          const tools = events.filter(e => ['search_knowledge','read_source','query_graph'].includes(e.action) && !['duplicate','budget_blocked'].includes(e.status)).length;
          status.textContent = (stages[w.phase] || w.phase) + ' · LLM Tokens ' + (u.total_tokens || 0) + '/' + (w.llm_token_budget || 50000) + ' · 嵌入 Tokens ' + (u.embedding_tokens || 0) + '/' + (w.embedding_token_budget || 250000) + ' · 工具调用 ' + tools + ' · 循环拦截 ' + events.filter(e => e.status === 'duplicate' || e.action === 'loop_guard').length + '（用量随接口返回更新）';
        } catch (_) {}
      }, 1200);
    }
    if (sessionId) payload.session_id = sessionId;
    const response = await fetch('/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    if (response.status === 404) {
      sessionId = null;
      sessionStorage.removeItem('rag-session-id');
      throw new Error('会话已失效，请重新发送问题以开始新会话。');
    }
    if (!response.ok) throw new Error('请求失败，请重试；对话过长时可先结束对话。');
    const data = await response.json();
    sessionId = data.session_id;
    sessionStorage.setItem('rag-session-id', sessionId);
    addMessage('assistant', data.answer);
    question.value = '';
    status.textContent = data.stop_reason === 'partial' ? '本次部分完成，未完成项已列在回答末尾，可继续追问。' : data.committed
      ? '已完成 ' + data.turns + ' 轮对话，可以继续追问。'
      : '本次未完成，失败内容未加入会话记忆，可以重试。';
  } catch (error) {
    bubble.remove();
    const disconnected = error instanceof TypeError || error.name === 'AbortError';
    status.textContent = disconnected ? offlineMessage : error.message;
    if (disconnected) window.alert(offlineMessage);
  } finally {
    clearInterval(monitorTimer);
    setBusy(false);
    question.focus();
  }
});
question.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});
end.addEventListener('click', async () => {
  if (busy) return;
  setBusy(true);
  try {
    if (sessionId) {
      const response = await fetch('/sessions/' + encodeURIComponent(sessionId), {
        method: 'DELETE'
      });
      if (!response.ok && response.status !== 404) throw new Error('清除失败，请重试。');
    }
    sessionId = null;
    sessionStorage.removeItem('rag-session-id');
    messages.replaceChildren();
    question.value = '';
    status.textContent = '对话已结束，记忆已清除。可以开始新的对话。';
  } catch (error) {
    status.textContent = error.message;
  } finally {
    setBusy(false);
    question.focus();
  }
});
async function restoreSession() {
  if (!sessionId) return;
  setBusy(true);
  try {
    const response = await fetch('/sessions/' + encodeURIComponent(sessionId));
    if (response.status === 404) {
      sessionId = null;
      sessionStorage.removeItem('rag-session-id');
      return;
    }
    if (!response.ok) throw new Error('暂时无法恢复对话，请刷新重试。');
    const data = await response.json();
    for (const message of data.messages) addMessage(message.role, message.content);
    status.textContent = '已恢复 ' + data.turns + ' 轮对话。';
  } catch (error) {
    status.textContent = '对话恢复失败，请刷新页面重试。';
    return;
  } finally {
    setBusy(false);
  }
}
restoreSession();
</script>
</html>"""


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    mode: Literal["naive", "local", "global", "hybrid", "mix"] = "hybrid"
    engine: Literal["agent", "rag"] | None = None
    workflow: bool = False
    monitor_id: str | None = Field(default=None, pattern=r'^[a-f0-9-]{36}$')

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value.strip()


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    turns: int
    engine: str
    run_id: str
    stop_reason: str
    committed: bool


def build_rag() -> LightRAG:
    """Reuse the persisted index and the terminal demo's model configuration."""
    index = INDEX_DIR
    graph = index / WORKSPACE / "graph_chunk_entity_relation.graphml"
    if not graph.is_file():
        raise RuntimeError(f"Build the book index first: {graph}")
    llm_model_func = make_llm_func(default_llm_model())
    embedding_func = make_embedding_func("remote")
    return LightRAG(
        working_dir=str(index),
        workspace=WORKSPACE,
        tokenizer=make_local_tokenizer(),
        llm_model_func=llm_model_func,
        llm_model_name=default_llm_model(),
        llm_model_max_async=2,
        embedding_func=embedding_func,
        embedding_func_max_async=4,
        rerank_model_func=make_embedding_rerank_func(embedding_func),
        min_rerank_score=-1.0,
    )


def create_app(rag_factory=build_rag, db_path=SESSION_DB,
               default_engine="agent", settings=None) -> FastAPI:
    settings = settings or AgentSettings.from_env()
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        app.state.store = SessionStore(db_path)
        # Serialize requests for this local demo, including session deletion.
        app.state.query_lock = asyncio.Lock()
        app.state.activities = {}
        rag = None
        try:
            rag = rag_factory()
            app.state.rag = rag
            await rag.initialize_storages()
            app.state.agent = RetrievalAgent(rag, settings=settings)
            app.state.research = ResearchTasks(app)
            app.state.management = Management(app)
            app.state.ready = True
            yield
        finally:
            app.state.ready = False
            try:
                if hasattr(app.state, "research"):
                    await app.state.research.close()
                if hasattr(app.state, "management"):
                    await app.state.management.close()
                if rag is not None:
                    await finalize_rag(rag)
            finally:
                app.state.store.close()

    app = FastAPI(title="RAG Knowledge Base API", lifespan=lifespan)
    app.include_router(task_router(app))
    app.include_router(management_router(app))

    @app.get('/manage', response_class=HTMLResponse, include_in_schema=False)
    async def manage_page():
        return HTMLResponse(Path(__file__).with_name('management.html').read_text(encoding='utf-8'))

    @app.get("/tasks", response_class=HTMLResponse, include_in_schema=False)
    async def tasks_page():
        return HTMLResponse(Path(__file__).with_name("tasks.html").read_text(encoding="utf-8"))

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def chat_page():
        return HTMLResponse(CHAT_PAGE)

    @app.get("/health")
    async def health():
        if not getattr(app.state, "ready", False):
            raise HTTPException(503, "Index is not ready")
        return {"status": "ready"}

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        if len(request.question.encode("utf-8")) > settings.question_bytes:
            raise HTTPException(422, "Question exceeds the configured input budget")
        async with app.state.query_lock:
            store = app.state.store
            session_id = request.session_id
            if session_id is not None and not store.exists(session_id):
                raise HTTPException(404, "Session not found; omit session_id to start one")
            session_id = session_id or str(uuid4())
            history = store.history(session_id)
            engine_name = request.engine or default_engine
            run_id = store.start_run(session_id, request.question, engine_name)
            usage = RequestUsage()
            usage_context = ACTIVE_USAGE.set(usage)
            args = SimpleNamespace(
                history_turns=0,
                mode=request.mode,
                only_context=False,
                no_query_rewrite=False,
                rerank="embedding",
                rerank_top_k=10,
                max_total_tokens=16000,
            )
            try:
                if engine_name == "agent":
                    runner = app.state.agent
                    run_options = {}
                    if request.workflow:
                        from rag_app.workflow import ResearchWorkflow
                        runner = ResearchWorkflow(runner)
                        if request.monitor_id:
                            if len(app.state.activities) >= 20:
                                app.state.activities.pop(next(iter(app.state.activities)))
                            def update_activity(state):
                                app.state.activities[request.monitor_id] = state
                            run_options['on_state'] = update_activity
                    result = await runner.run(
                        request.question, history, request.mode,
                        on_event=lambda trace: store.progress(
                            session_id, run_id, request.question, trace, usage.get_usage()),
                        **run_options,
                    )
                else:
                    # Recall must read the full stored history; only model context is bounded.
                    from rag_app.runtime import recall_previous_question
                    recalled = recall_previous_question(request.question, history)
                    if recalled is not None:
                        result = AgentResult(recalled, "recall")
                    else:
                        answer = await asyncio.wait_for(query_rag(
                            app.state.rag, args, request.question, app.state.rag.llm_model_func,
                            recent_history(history, settings.history_turns, settings.history_bytes),
                        ), timeout=settings.request_timeout)
                        result = AgentResult(answer, "answered")
                if not isinstance(result.answer, str):
                    raise TypeError("Expected a non-streaming text response")
                result.answer = plain_answer(result.answer)
            except asyncio.CancelledError:
                if request.monitor_id in app.state.activities:
                    app.state.activities[request.monitor_id]['phase'] = 'interrupted'
                interrupted = AgentResult("", "interrupted", usage=usage.get_usage())
                previous_run = store.run(session_id, run_id)
                interrupted.trace = previous_run.get("trace", [])
                store.save_turn(session_id, request.question, interrupted, engine_name,
                                commit_messages=False, run_id=run_id)
                raise
            except Exception as exc:
                logger.error("Book API query failed (%s)", type(exc).__name__)
                failure = AgentResult("", "error", trace=[{"error_type": type(exc).__name__}],
                                      usage=usage.get_usage())
                store.save_turn(session_id, request.question, failure, engine_name,
                                commit_messages=False, run_id=run_id)
                raise HTTPException(
                    502, "Query failed; retry or start a new session if history is too long"
                ) from exc
            finally:
                ACTIVE_USAGE.reset(usage_context)
            result.usage = usage.get_usage()
            committed = result.stop_reason in {"answered", "budget_answer", "recall", "clarify", "insufficient", "fallback", "partial"}
            run_id = store.save_turn(session_id, request.question, result, engine_name, committed, run_id)
            return ChatResponse(
                session_id=session_id, answer=result.answer,
                turns=len(history) // 2 + int(committed), engine=engine_name,
                run_id=run_id, stop_reason=result.stop_reason, committed=committed,
            )

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        async with app.state.query_lock:
            if not app.state.store.exists(session_id):
                raise HTTPException(404, "Session not found")
            history = app.state.store.history(session_id)
            return {"session_id": session_id, "messages": history, "turns": len(history) // 2,
                    "runs": app.state.store.recent_runs(session_id)}

    @app.get('/activity/{monitor_id}')
    async def activity(monitor_id: str):
        state = app.state.activities.get(monitor_id)
        if state is None:
            raise HTTPException(404, '尚未开始或记录已过期')
        return state

    @app.get("/sessions/{session_id}/runs/{run_id}")
    async def get_run(session_id: str, run_id: str):
        async with app.state.query_lock:
            result = app.state.store.run(session_id, run_id)
            if result is None:
                raise HTTPException(404, "Run not found")
            return result

    @app.delete("/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        async with app.state.query_lock:
            if not app.state.store.delete(session_id):
                raise HTTPException(404, "Session not found")
        return Response(status_code=204)

    return app


app = create_app()
