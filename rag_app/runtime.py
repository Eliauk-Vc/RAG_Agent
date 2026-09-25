"""Index a UTF-8 text file with LightRAG and run an interactive query.

This example is intentionally small and explicit so it can be used as a learning
entry point. It supports the OpenAI-compatible services configured in ``.env``
and a dependency-free local embedding fallback for smoke tests.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import json_repair
import numpy as np
from rag_app.config import BOOK_PATH, INDEX_DIR, WORKSPACE, project_path


DEFAULT_BOOK_PATH = BOOK_PATH

# Import after loading .env because LightRAG reads several defaults at import time.
from lightrag import LightRAG, QueryParam  # noqa: E402
from lightrag.llm.openai import (  # noqa: E402
    openai_complete_if_cache,
    openai_embed,
)
from lightrag.utils import EmbeddingFunc, Tokenizer  # noqa: E402


LOCAL_EMBEDDING_DIM = 2048
EXIT_COMMANDS = {"/exit", "/quit", "exit", "quit", "退出"}
QUERY_REWRITE_SYSTEM_PROMPT = """You rewrite conversational questions for retrieval.
Use the conversation history to resolve pronouns and omitted subjects in the latest
question. Preserve the user's language and intent, do not answer the question, and do
not add facts. Keep an already self-contained question unchanged, especially when
the user explicitly names a new topic. Never replace a request to recall the
conversation with an answer to an earlier knowledge question.
Return JSON only in this form: {"query": "standalone question"}.
"""


class UnicodeCodepointCodec:
    """Provide deterministic local tokenization without downloading tiktoken data."""

    def encode(self, content: str) -> list[int]:
        return [ord(character) for character in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def make_local_tokenizer() -> Tokenizer:
    """Build a conservative tokenizer suitable for this Chinese learning demo."""
    return Tokenizer(
        model_name="unicode-codepoint-v1",
        tokenizer=UnicodeCodepointCodec(),
    )


def configure_console() -> None:
    """Keep Chinese output readable in Windows terminals."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def read_book(path: Path, start_char: int, max_chars: int | None) -> str:
    """Read the source strictly as UTF-8 so bad input fails visibly."""
    text = path.read_text(encoding="utf-8-sig")[start_char:]
    if max_chars is not None:
        text = text[:max_chars]
    if not text.strip():
        raise ValueError(f"The input file is empty: {path}")
    return text


def _text_features(text: str) -> list[str]:
    """Create lexical features that work for both Chinese and Latin text."""
    units = re.findall(r"[\u3400-\u9fff]|[a-z0-9]+", text.lower())
    bigrams = [f"{left}{right}" for left, right in zip(units, units[1:])]
    return units + bigrams


async def local_hash_embed(texts: list[str]) -> np.ndarray:
    """Return deterministic character n-gram embeddings without extra models."""
    vectors = np.zeros((len(texts), LOCAL_EMBEDDING_DIM), dtype=np.float32)

    for row, value in enumerate(texts):
        for feature in _text_features(value):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            column = int.from_bytes(digest[:4], "little") % LOCAL_EMBEDDING_DIM
            sign = 1.0 if digest[4] & 1 else -1.0
            vectors[row, column] += sign

        norm = np.linalg.norm(vectors[row])
        if norm:
            vectors[row] /= norm

    return vectors


def make_embedding_func(mode: str) -> EmbeddingFunc:
    if mode == "local":
        return EmbeddingFunc(
            embedding_dim=LOCAL_EMBEDDING_DIM,
            max_token_size=int(os.getenv("EMBEDDING_TOKEN_LIMIT", "8192")),
            model_name="local-char-ngram-hash-v1",
            func=local_hash_embed,
        )

    required = (
        "EMBEDDING_BINDING_API_KEY",
        "EMBEDDING_BINDING_HOST",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIM",
    )
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing remote embedding settings: {', '.join(missing)}")

    async def tracked_embed(texts, **kwargs):
        from rag_app.telemetry import ACTIVE_USAGE, ACTIVE_BUDGET

        guard = ACTIVE_BUDGET.get()
        estimate = sum(len(t.encode('utf-8')) + 32 for t in texts)
        if guard:
            guard.acquire('embedding', estimate)
        usage = ACTIVE_USAGE.get()
        if usage is not None:
            usage.embedding_invocations += 1
            kwargs["token_tracker"] = usage.embedding_usage
        try:
            return await openai_embed.func(
                texts, model=os.environ["EMBEDDING_MODEL"],
                api_key=os.environ["EMBEDDING_BINDING_API_KEY"].strip(),
                base_url=os.environ["EMBEDDING_BINDING_HOST"].strip(), **kwargs,
            )
        finally:
            if guard:
                guard.release('embedding', estimate)

    return EmbeddingFunc(
        embedding_dim=int(os.environ["EMBEDDING_DIM"]),
        max_token_size=int(os.getenv("EMBEDDING_TOKEN_LIMIT", "8192")),
        model_name=os.environ["EMBEDDING_MODEL"],
        supports_asymmetric=True,
        func=tracked_embed,
    )


def make_embedding_rerank_func(embedding_func: EmbeddingFunc):
    """Build a second-stage semantic reranker using the configured embeddings."""

    async def embedding_rerank(
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[dict[str, float | int]]:
        if not documents:
            return []
        from rag_app.embedding_reuse import embed_once

        query_vector = np.asarray(
            await embed_once(embedding_func, [query], context="query"), dtype=np.float32
        )[0]
        document_vectors = np.asarray(
            await embed_once(embedding_func, documents, context="document"), dtype=np.float32
        )

        query_norm = float(np.linalg.norm(query_vector))
        document_norms = np.linalg.norm(document_vectors, axis=1)
        denominator = document_norms * query_norm
        scores = np.divide(
            document_vectors @ query_vector,
            denominator,
            out=np.zeros(len(documents), dtype=np.float32),
            where=denominator > 0,
        )

        ranked_indices = np.argsort(-scores, kind="stable")
        if top_n is not None:
            ranked_indices = ranked_indices[:top_n]

        return [
            {"index": int(index), "relevance_score": float(scores[index])}
            for index in ranked_indices
        ]

    return embedding_rerank


def make_llm_func(model: str, token_tracker=None):
    book_api_key = os.getenv("BOOK_LLM_API_KEY")
    book_base_url = os.getenv("BOOK_LLM_BINDING_HOST")
    anthropic_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
    anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "")

    if book_api_key and book_base_url:
        api_key = book_api_key.strip()
        base_url = book_base_url.strip()
    elif anthropic_token and "api.deepseek.com" in anthropic_base_url.lower():
        # The same DeepSeek token works with its OpenAI-compatible endpoint.
        api_key = anthropic_token.strip()
        base_url = "https://api.deepseek.com"
    else:
        api_key = (
            os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        ).strip()
        base_url = os.getenv("LLM_BINDING_HOST", "").strip()

    if not api_key or not base_url:
        raise RuntimeError(
            "Set BOOK_LLM_API_KEY and BOOK_LLM_BINDING_HOST, or configure the "
            "corresponding LLM_BINDING_* fallback values"
        )

    async def llm_model_func(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict] | None = None,
        keyword_extraction: bool = False,
        **kwargs,
    ) -> str:
        # LightRAG may pass this through llm_model_kwargs. This demo deliberately
        # disables chain-of-thought and must avoid forwarding the argument twice.
        kwargs.pop("enable_cot", None)
        call_token_tracker = kwargs.pop("token_tracker", token_tracker)
        from rag_app.telemetry import ACTIVE_USAGE, ACTIVE_BUDGET

        guard = ACTIVE_BUDGET.get()
        estimate = len((prompt + (system_prompt or '') + json.dumps(history_messages or [], ensure_ascii=False)).encode('utf-8')) + 512 + int(kwargs.get('max_tokens') or 1800)
        if guard:
            guard.acquire('llm', estimate)
            if not kwargs.get('max_tokens'):
                kwargs['max_tokens'] = 1800
        request_usage = ACTIVE_USAGE.get()
        if request_usage is not None:
            request_usage.llm_invocations += 1
            call_token_tracker = request_usage
        is_deepseek = "api.deepseek.com" in base_url.lower()
        if is_deepseek:
            extra_body = dict(kwargs.pop("extra_body", {}) or {})
            extra_body.setdefault("thinking", {"type": "disabled"})
            kwargs["extra_body"] = extra_body
        try:
            return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            # DeepSeek currently rejects the Pydantic response schema used by
            # openai_complete_if_cache. Its keyword prompt already requests JSON,
            # which LightRAG validates with json_repair after this call.
            keyword_extraction=keyword_extraction and not is_deepseek,
            api_key=api_key,
            base_url=base_url,
            timeout=int(os.getenv("BOOK_LLM_TIMEOUT", "90")),
            enable_cot=False,
            token_tracker=call_token_tracker,
            **kwargs,
            )
        finally:
            if guard:
                guard.release('llm', estimate)

    return llm_model_func


def default_llm_model() -> str:
    if os.getenv("BOOK_LLM_MODEL"):
        return os.environ["BOOK_LLM_MODEL"].strip()
    if os.getenv("ANTHROPIC_AUTH_TOKEN") and "api.deepseek.com" in os.getenv(
        "ANTHROPIC_BASE_URL", ""
    ).lower():
        return os.getenv("ANTHROPIC_MODEL", "deepseek-v4-pro").strip()
    return "glm-4-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--book",
        type=Path,
        default=DEFAULT_BOOK_PATH,
        help=f"UTF-8 text file to index (default: {DEFAULT_BOOK_PATH})",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=INDEX_DIR,
        help="Index and cache directory",
    )
    parser.add_argument(
        "--embedding",
        choices=("local", "remote"),
        default="remote",
        help="Local lexical smoke-test vectors or the embedding service from .env",
    )
    parser.add_argument(
        "--llm-model",
        default=default_llm_model(),
        help="OpenAI-compatible chat model used for extraction and answers",
    )
    parser.add_argument(
        "--query",
        default="CCMD-3中精神分裂症的主要诊断标准是什么？",
        help="Question asked after indexing",
    )
    parser.add_argument(
        "--mode",
        choices=("naive", "local", "global", "hybrid", "mix"),
        default="hybrid",
        help="LightRAG retrieval mode",
    )
    parser.add_argument(
        "--start-char",
        type=int,
        default=0,
        help="Start indexing at this character offset; useful for one chapter",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        help="Index only the first N characters; useful for a low-cost smoke test",
    )
    parser.add_argument(
        "--skip-insert",
        action="store_true",
        help="Reuse an existing index and only run the query",
    )
    parser.add_argument(
        "--only-context",
        action="store_true",
        help="Print retrieved evidence without asking the LLM for a final answer",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep the index open and accept questions until /exit or Ctrl+C",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=0,
        help=(
            "Number of recent question-answer turns sent to the LLM in interactive "
            "mode; 0 remembers the entire current session (default: 0)"
        ),
    )
    parser.add_argument(
        "--no-query-rewrite",
        action="store_true",
        help="Disable history-aware standalone question rewriting before retrieval",
    )
    parser.add_argument(
        "--rerank",
        choices=("embedding", "none"),
        default="embedding",
        help="Second-stage chunk reranker (default: embedding)",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=10,
        help="Number of text chunks kept after reranking (default: 10)",
    )
    return parser.parse_args()


async def rewrite_query_for_retrieval(
    llm_model_func,
    question: str,
    conversation_history: list[dict[str, str]],
) -> str:
    """Turn a context-dependent follow-up into a standalone retrieval query."""
    if not conversation_history:
        return question

    try:
        response = await llm_model_func(
            question,
            system_prompt=QUERY_REWRITE_SYSTEM_PROMPT,
            history_messages=conversation_history,
            keyword_extraction=False,
            stream=False,
        )
        parsed = json_repair.loads(response)
        rewritten = parsed.get("query", "") if isinstance(parsed, dict) else ""
        rewritten = rewritten.strip()
        return rewritten or question
    except Exception as exc:
        print(
            f"Query rewrite unavailable ({type(exc).__name__}); "
            "using the original question."
        )
        return question


async def query_rag(
    rag: LightRAG,
    args: argparse.Namespace,
    question: str,
    llm_model_func,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    """Run one retrieval query while sharing configuration across both modes."""
    history = conversation_history or []
    recent_history = (
        list(history)
        if args.history_turns == 0
        else history[-args.history_turns * 2 :]
    )
    if not args.only_context:
        recall = recall_previous_question(question, recent_history)
        if recall is not None:
            return recall
    retrieval_question = question
    if recent_history and not args.no_query_rewrite:
        retrieval_question = await rewrite_query_for_retrieval(
            llm_model_func,
            question,
            recent_history,
        )
        if retrieval_question != question:
            print(f"Retrieval query: {retrieval_question}")

    return await rag.aquery(
        retrieval_question,
        param=QueryParam(
            mode=args.mode,
            only_need_context=args.only_context,
            response_type="Chinese concise answer with evidence",
            user_prompt=answer_instructions(question, recent_history),
            conversation_history=recent_history,
            chunk_top_k=args.rerank_top_k,
            max_total_tokens=getattr(args, "max_total_tokens", 16000),
            max_entity_tokens=1500,
            max_relation_tokens=1500,
            enable_rerank=args.rerank != "none",
        ),
    )


def recall_previous_question(
    question: str, history: list[dict[str, str]]
) -> str | None:
    """Resolve explicit last-question requests from this session, without retrieval."""
    normalized = re.sub(r"[\s，。？！,.?!]", "", question)
    if not re.fullmatch(
        r"(?:请)?(?:告诉我)?我(?:刚才|刚刚|刚|上次|上一轮)"
        r"(?:(?:问|提问|提)(?:的)?(?:问题)?(?:是)?什么(?:问题)?(?:来着)?|"
        r"(?:问|提问)(?:了|过)什么(?:问题)?|(?:的问题|那\s*个问题)是什么)(?:呢|啊|呀|吗)?",
        normalized,
    ):
        return None
    previous = next(
        (item["content"] for item in reversed(history) if item.get("role") == "user"),
        None,
    )
    if previous is None:
        return "当前会话中还没有你之前的问题。"
    return f"你刚才的问题是：“{previous}”"


def answer_instructions(question: str, history: list[dict[str, str]]) -> str:
    """Keep answer intent separate from retrieval and scope the answer cache to history."""
    # LightRAG includes user_prompt in the answer cache key, but omits history.
    history_key = hashlib.sha256(
        json.dumps(history, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return (
        "Answer the latest original user question below. The retrieval query may "
        "have been rewritten only to find evidence; it must not replace the user's "
        "intent. Use history only when relevant, and do not answer a previous "
        "question instead. If the user changes topic, follow the new topic.\n"
        f"Latest original question: {json.dumps(question, ensure_ascii=False)}\n"
        f"Conversation cache version: 2/{history_key} (internal; do not output)."
    )


async def interactive_chat(
    rag: LightRAG,
    args: argparse.Namespace,
    llm_model_func,
) -> None:
    """Read questions repeatedly without rebuilding or reopening the index."""
    conversation_history: list[dict[str, str]] = []
    print("\nInteractive RAG chat is ready.")
    print("Enter a question, or use /exit, /quit, or 退出 to stop.")
    print("Press Ctrl+C to interrupt at any time.")
    if args.history_turns == 0:
        print("Memory: all turns in this run are remembered until you exit.")
    else:
        print(f"Memory: the latest {args.history_turns} turn(s) are remembered.")

    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nStopping interactive chat...")
            return

        if not question:
            continue
        if question.lower() in EXIT_COMMANDS:
            print("Stopping interactive chat...")
            return

        try:
            answer = await query_rag(
                rag,
                args,
                question,
                llm_model_func,
                conversation_history=conversation_history,
            )
        except KeyboardInterrupt:
            print("\nQuery interrupted. Stopping interactive chat...")
            return
        except Exception as exc:
            print(f"\nQuery failed: {type(exc).__name__}: {exc}")
            print("You can ask another question or enter /exit.")
            continue

        print("\nAssistant:\n")
        print(answer)
        if not args.only_context:
            conversation_history.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )


async def run(args: argparse.Namespace) -> None:
    book_path = project_path(args.book)
    working_dir = project_path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    print(f"Book: {book_path}")
    print(f"Index: {working_dir}")
    print(
        f"LLM: {args.llm_model}; embedding: {args.embedding}; "
        f"mode: {args.mode}; rerank: {args.rerank}"
    )

    llm_model_func = make_llm_func(args.llm_model)
    embedding_func = make_embedding_func(args.embedding)
    rerank_model_func = (
        make_embedding_rerank_func(embedding_func)
        if args.rerank == "embedding"
        else None
    )

    rag = LightRAG(
        working_dir=str(working_dir),
        workspace=WORKSPACE,
        tokenizer=make_local_tokenizer(),
        llm_model_func=llm_model_func,
        llm_model_name=args.llm_model,
        llm_model_max_async=2,
        embedding_func=embedding_func,
        embedding_func_max_async=4,
        rerank_model_func=rerank_model_func,
        min_rerank_score=-1.0,
        max_parallel_insert=2,
        entity_extract_max_gleaning=0,
        addon_params={
            "language": "Chinese",
            "entity_types": [
                "疾病",
                "症状",
                "诊断标准",
                "病程",
                "治疗",
                "药物",
                "人群",
                "概念",
            ],
        },
    )

    await rag.initialize_storages()
    try:
        if not args.skip_insert:
            text = read_book(book_path, args.start_char, args.max_chars)
            print(
                f"Indexing {len(text):,} characters "
                f"from offset {args.start_char:,}..."
            )
            track_id = await rag.ainsert(text, file_paths=str(book_path))
            documents = await rag.doc_status.get_docs_by_track_id(track_id)
            failures = [
                document
                for document in documents.values()
                if document.status.value == "failed"
            ]
            if failures:
                details = "; ".join(
                    document.error_msg or "unknown indexing error"
                    for document in failures
                )
                raise RuntimeError(f"Indexing failed for track {track_id}: {details}")
            print(f"Indexing succeeded. Track ID: {track_id}")

        if args.interactive:
            await interactive_chat(rag, args, llm_model_func)
        else:
            print(f"\nQuestion: {args.query}")
            answer = await query_rag(rag, args, args.query, llm_model_func)
            print("\nResult:\n")
            print(answer)
    finally:
        await rag.finalize_storages()


def main() -> None:
    configure_console()
    args = parse_args()
    if args.start_char < 0:
        raise SystemExit("--start-char cannot be negative")
    if args.max_chars is not None and args.max_chars <= 0:
        raise SystemExit("--max-chars must be greater than zero")
    if args.history_turns < 0:
        raise SystemExit("--history-turns cannot be negative")
    if args.rerank_top_k <= 0:
        raise SystemExit("--rerank-top-k must be greater than zero")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
