"""Request-local provider usage, including keyword extraction inside LightRAG."""

from contextvars import ContextVar

from lightrag.utils import TokenTracker

ACTIVE_USAGE: ContextVar = ContextVar("rag_agent_usage", default=None)
ACTIVE_BUDGET: ContextVar = ContextVar('rag_workflow_budget', default=None)
FINALIZING: ContextVar = ContextVar('rag_finalizing', default=False)


class BudgetExceeded(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class WorkflowBudget:
    """Reserve conservative per-call estimates, including concurrent embedding calls."""
    def __init__(self, llm_limit, embedding_limit, reserve=8000):
        self.llm_limit = llm_limit
        self.embedding_limit = embedding_limit
        self.reserve = min(reserve, llm_limit // 3)
        self.held = {'llm': 0, 'embedding': 0}
        self.blocked = None

    def remaining(self, kind, final=False):
        usage = ACTIVE_USAGE.get()
        counts = usage.get_usage() if usage else {}
        used = counts.get('total_tokens' if kind == 'llm' else 'embedding_tokens', 0)
        limit = self.llm_limit if kind == 'llm' else self.embedding_limit
        if kind == 'llm' and not final:
            limit -= self.reserve
        return max(0, limit - used - self.held[kind])

    def acquire(self, kind, estimate):
        if estimate > self.remaining(kind, FINALIZING.get()):
            reason = kind + '_token_budget'
            if not FINALIZING.get():
                self.blocked = reason
            raise BudgetExceeded(reason)
        self.held[kind] += estimate

    def release(self, kind, estimate):
        self.held[kind] -= estimate


class RequestUsage(TokenTracker):
    def __init__(self):
        super().__init__()
        self.llm_invocations = 0
        self.embedding_invocations = 0
        self.embedding_usage = TokenTracker()

    def get_usage(self):
        return {**super().get_usage(), "llm_invocations": self.llm_invocations,
                "embedding_invocations": self.embedding_invocations,
                "embedding_tokens": self.embedding_usage.get_usage()["total_tokens"],
                "usage_reported": self.call_count > 0}
