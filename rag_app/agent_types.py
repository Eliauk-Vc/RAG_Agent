"""Validated actions and bounded runtime settings for the retrieval agent."""

import os
from dataclasses import dataclass, field
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
ChunkId = Annotated[str, StringConstraints(pattern=r"^chunk-[a-f0-9]{32}$")]
Mode = Literal["naive", "local", "global", "hybrid", "mix"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchArgs(StrictModel):
    query: Text
    mode: Mode = "hybrid"
    top_k: int = Field(default=6, ge=1, le=10)


class SourceArgs(StrictModel):
    chunk_id: ChunkId
    surrounding_chunks: int = Field(default=1, ge=0, le=2)


class GraphArgs(StrictModel):
    entity_name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    max_depth: int = Field(default=1, ge=1, le=2)
    max_nodes: int = Field(default=12, ge=1, le=30)


class Search(StrictModel):
    action: Literal["search_knowledge"]
    arguments: SearchArgs


class Read(StrictModel):
    action: Literal["read_source"]
    arguments: SourceArgs


class Graph(StrictModel):
    action: Literal["query_graph"]
    arguments: GraphArgs


class Answer(StrictModel):
    action: Literal["answer"]
    evidence_ids: list[ChunkId] = Field(min_length=1, max_length=10)


class Clarify(StrictModel):
    action: Literal["clarify"]
    question: Text


class Insufficient(StrictModel):
    action: Literal["insufficient"]


Action = Annotated[Search | Read | Graph | Answer | Clarify | Insufficient, Field(discriminator="action")]
ACTION_ADAPTER = TypeAdapter(Action)


@dataclass(frozen=True)
class AgentSettings:
    max_searches: int = 3
    max_tools: int = 5
    max_decisions: int = 7
    request_timeout: float = 120
    tool_timeout: float = 40
    model_timeout: float = 45
    history_turns: int = 4
    history_bytes: int = 6000
    evidence_bytes: int = 18000
    prompt_bytes: int = 36000
    question_bytes: int = 12000

    def __post_init__(self):
        if any(value <= 0 for value in vars(self).values()):
            raise ValueError("Agent limits must be positive")
        if self.prompt_bytes < 4000:
            raise ValueError("Agent prompt budget must be at least 4000 bytes")
        if self.prompt_bytes < self.question_bytes + 4000:
            raise ValueError("Agent prompt budget must leave 4000 bytes beyond the question budget")

    @classmethod
    def from_env(cls):
        defaults = cls()
        return cls(**{
            name: type(value)(os.getenv("AGENT_" + name.upper(), str(value)))
            for name, value in vars(defaults).items()
        })


@dataclass
class AgentResult:
    answer: str
    stop_reason: str
    evidence: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    def evaluation_payload(self):
        return {
            "status": "success" if self.stop_reason in {"answered", "budget_answer"} else "failure",
            "message": self.stop_reason,
            "data": {"chunks": self.evidence, "references": [
                {"file_path": row["file_path"], "reference_id": row["chunk_id"]}
                for row in self.evidence
            ]},
            "llm_response": {"content": self.answer},
        }
