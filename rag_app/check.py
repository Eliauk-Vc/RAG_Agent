"""Check imports, source paths and real index initialization without model calls."""
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from rag_app.config import BOOK_PATH, INDEX_DIR, REPO_ROOT, WORKSPACE
from rag_app.api import build_rag, create_app
import lightrag


def main() -> None:
    if not Path(lightrag.__file__).resolve().is_relative_to(REPO_ROOT):
        raise RuntimeError("LightRAG was imported from outside this project")
    if not BOOK_PATH.is_file():
        raise RuntimeError(f"Missing source document: {BOOK_PATH}")
    rag = build_rag()
    for name in ("chunks", "entities", "relationships"):
        path = INDEX_DIR / WORKSPACE / f"vdb_{name}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["embedding_dim"] != rag.embedding_func.embedding_dim:
            raise RuntimeError(f"Embedding dimension mismatch: {path}")
        if not payload["data"]:
            raise RuntimeError(f"Empty vector index: {path}")
        for item in payload["data"]:
            for source in item.get("file_path", "").split("<SEP>"):
                if source and not Path(source).is_file():
                    raise RuntimeError(f"Missing indexed source: {source}")
    with TestClient(create_app(lambda: rag, db_path=":memory:")) as client:
        for route in ("/health", "/", "/openapi.json"):
            response = client.get(route)
            if response.status_code != 200:
                raise RuntimeError(f"Route failed: {route}")
    print(f"Python: {sys.executable}")
    print(f"LightRAG: {lightrag.__file__}")
    print(f"Index: {INDEX_DIR / WORKSPACE}")
    print("PASS: local imports, source references, vector dimensions, storage lifecycle and web routes")
    print("Remote model requests were not performed.")


if __name__ == "__main__":
    main()
