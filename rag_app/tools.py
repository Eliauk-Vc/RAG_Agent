"""Read-only, validated knowledge tools; the agent never opens arbitrary paths."""

import re
import os
import asyncio

from lightrag import QueryParam
from rag_app.agent_types import GraphArgs, SearchArgs, SourceArgs
from rag_app.context import clip_text
from rag_app.mmr import mmr_indices
from rag_app.embedding_reuse import SEARCH_EMBEDDINGS, SearchEmbeddings, embed_once


class KnowledgeTools:
    def __init__(self, rag):
        self.rag = rag
        self.mmr_enabled = os.getenv('AGENT_MMR_ENABLED', 'true').lower() in {'true', '1', 'yes'}
        self.mmr_lambda = float(os.getenv('AGENT_MMR_LAMBDA', '0.7'))
        self.mmr_timeout = float(os.getenv('AGENT_MMR_TIMEOUT', '8'))
        if not 0 <= self.mmr_lambda <= 1 or not 0 < self.mmr_timeout <= 30:
            raise ValueError('Invalid MMR configuration')

    @staticmethod
    def chunk(row: dict, chunk_id: str | None = None) -> dict | None:
        identifier = chunk_id or row.get("chunk_id") or row.get("_id", "")
        content = row.get("content")
        if not re.fullmatch(r"chunk-[a-f0-9]{32}", identifier) or not isinstance(content, str) or not content.strip():
            return None
        return {"chunk_id": identifier, "content": clip_text(content, 6000),
                "file_path": clip_text(str(row.get("file_path", "")), 500),
                "truncated": len(content.encode("utf-8")) > 6000}

    async def execute(self, name: str, arguments: dict) -> dict:
        if name == 'search_knowledge' and self.mmr_enabled and SEARCH_EMBEDDINGS.get() is None:
            cache = SearchEmbeddings(model_identity=id(self.rag))
            token = SEARCH_EMBEDDINGS.set(cache)
            try:
                result = await self._execute(name, arguments)
                result['selection']['embedding_reused_texts'] = cache.hits
                result['selection']['embedding_computed_texts'] = cache.misses
                return result
            finally:
                SEARCH_EMBEDDINGS.reset(token)
                cache.vectors.clear()
        return await self._execute(name, arguments)

    async def finalize_evidence(self, question, chunks, top_k=6):
        """Run MMR only after the agent explicitly chooses to answer."""
        candidates = list({row['chunk_id']: row for row in chunks}.values())
        meta = {'method': 'mmr', 'stage': 'after_sufficient',
                'candidates': len(candidates), 'input_ids': [r['chunk_id'] for r in candidates]}
        if not self.mmr_enabled or len(candidates) < 2:
            return candidates, {**meta, 'status': 'skipped', 'selected': len(candidates)}
        async def select():
            cache = SEARCH_EMBEDDINGS.get()
            originals = cache.originals if cache is not None else {}
            query = await embed_once(self.rag.embedding_func, [question], context='query')
            vectors = await embed_once(self.rag.embedding_func,
                [originals.get(r['chunk_id'], r['content']) for r in candidates], context='document')
            indices = mmr_indices(query[0], vectors, top_k, self.mmr_lambda)
            return [candidates[i] for i in indices]
        try:
            selected = await asyncio.wait_for(select(), self.mmr_timeout)
            return selected, {**meta, 'status': 'ok', 'selected': len(selected)}
        except Exception as exc:
            return candidates, {**meta, 'status': 'fallback', 'selected': len(candidates),
                                'error_type': type(exc).__name__}

    async def _execute(self, name: str, arguments: dict) -> dict:
        if name == "search_knowledge":
            args = SearchArgs.model_validate(arguments)
            fetch_k = args.top_k
            result = await self.rag.aquery_data(args.query, QueryParam(
                mode=args.mode, chunk_top_k=fetch_k, top_k=30,
                # Candidate count follows the requested top_k, independently of MMR.
                max_total_tokens=16000,
                max_entity_tokens=1500,
                max_relation_tokens=1500, enable_rerank=True,
            ))
            if not isinstance(result, dict):
                raise TypeError("Invalid retrieval result")
            data = result.get("data") or {}
            # Reuse the exact original text embedded by the relevance reranker.
            # The shorter writer excerpt must not trigger a second embedding.
            originals = {item['chunk_id']: row['content'] for row in data.get('chunks', [])[:fetch_k]
                         if (item := self.chunk(row)) is not None}
            candidates = [item for row in data.get("chunks", [])[:fetch_k]
                      if (item := self.chunk(row)) is not None]
            candidates = list({row['chunk_id']: row for row in candidates}.values())
            cache = SEARCH_EMBEDDINGS.get()
            if cache is not None:
                cache.originals.update(originals)
            chunks = candidates
            selection = {'method': 'relevance', 'status': 'mmr_deferred' if self.mmr_enabled else 'disabled',
                         'candidates': len(candidates), 'selected': len(chunks)}
            entities = [clip_text(str(row.get("entity_name", "")), 200)
                        for row in data.get("entities", [])[:12]]
            return {"status": "ok" if chunks else "empty", "chunks": chunks,
                    "entities": entities, 'selection': selection}
        if name == "read_source":
            args = SourceArgs.model_validate(arguments)
            row = await self.rag.text_chunks.get_by_id(args.chunk_id)
            if not row:
                return {"status": "not_found", "chunks": []}
            identifiers = [args.chunk_id]
            if args.surrounding_chunks and row.get("full_doc_id"):
                document = await self.rag.doc_status.get_by_id(row["full_doc_id"])
                ordered = (document or {}).get("chunks_list", [])
                if args.chunk_id in ordered:
                    index = ordered.index(args.chunk_id)
                    identifiers = ordered[max(0, index - args.surrounding_chunks):index + args.surrounding_chunks + 1]
            rows = await self.rag.text_chunks.get_by_ids(identifiers)
            chunks = [item for identifier, row in zip(identifiers, rows)
                      if row and (item := self.chunk(row, identifier)) is not None]
            return {"status": "ok" if chunks else "empty", "chunks": chunks}
        if name == "query_graph":
            args = GraphArgs.model_validate(arguments)
            if "*" in args.entity_name:
                raise ValueError("A specific entity is required")
            graph = await self.rag.get_knowledge_graph(
                args.entity_name, max_depth=args.max_depth, max_nodes=args.max_nodes)
            graph = graph.model_dump() if hasattr(graph, "model_dump") else graph
            nodes, edges, source_ids = [], [], []
            for row in graph.get("nodes", [])[:args.max_nodes]:
                props = row.get("properties", {})
                nodes.append({"name": clip_text(str(row["id"]), 200),
                              "description": clip_text(str(props.get("description", "")), 800)})
                source_ids.extend(str(props.get("source_id", "")).split("<SEP>"))
            for row in graph.get("edges", [])[:30]:
                props = row.get("properties", {})
                edges.append({"source": clip_text(str(row["source"]), 200),
                              "target": clip_text(str(row["target"]), 200),
                              "description": clip_text(str(props.get("description", "")), 500)})
                source_ids.extend(str(props.get("source_id", "")).split("<SEP>"))
            identifiers = list(dict.fromkeys(item for item in source_ids
                               if re.fullmatch(r"chunk-[a-f0-9]{32}", item)))[:6]
            rows = await self.rag.text_chunks.get_by_ids(identifiers) if identifiers else []
            chunks = [item for identifier, row in zip(identifiers, rows)
                      if row and (item := self.chunk(row, identifier)) is not None]
            return {"status": "ok" if nodes else "empty", "chunks": chunks,
                    "nodes": nodes, "edges": edges,
                    "truncated": bool(graph.get("is_truncated"))}
        raise ValueError("Unknown tool")
