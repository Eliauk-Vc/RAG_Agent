"""Resolve all project paths independently of the caller's directory."""
import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env", override=False)


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


BOOK_PATH = project_path(os.getenv("RAG_BOOK_PATH", "data/book.txt"))
INDEX_DIR = project_path(os.getenv("RAG_INDEX_DIR", "rag_storage/book_project_full"))
OUTPUT_DIR = project_path(os.getenv("RAG_OUTPUT_DIR", "outputs/evaluation"))
WORKSPACE = os.getenv("RAG_WORKSPACE", "demo")
SESSION_DB = project_path(os.getenv("RAG_SESSION_DB", "data/sessions.sqlite3"))
os.environ["LOG_DIR"] = str(project_path(os.getenv("LOG_DIR", "logs")))
