import unittest
from rag_app.telemetry import WorkflowBudget, BudgetExceeded, ACTIVE_USAGE, FINALIZING, RequestUsage


class BudgetTests(unittest.TestCase):
    def test_separate_limits_reserve_and_concurrent_reservations(self):
        usage = RequestUsage()
        usage.add_usage({'total_tokens': 40000})
        usage.embedding_usage.add_usage({'total_tokens': 100000})
        token = ACTIVE_USAGE.set(usage)
        try:
            guard = WorkflowBudget(50000, 250000)
            self.assertEqual(guard.remaining('llm'), 2000)
            guard.acquire('embedding', 90000)
            with self.assertRaises(BudgetExceeded):
                guard.acquire('embedding', 90000)
            guard.release('embedding', 90000)
            guard.acquire('llm', 1500)
            with self.assertRaises(BudgetExceeded):
                guard.acquire('llm', 1000)
            guard.release('llm', 1500)
            final = FINALIZING.set(True)
            try:
                guard.acquire('llm', 9000)
                with self.assertRaises(BudgetExceeded):
                    guard.acquire('llm', 2000)
                guard.release('llm', 9000)
            finally:
                FINALIZING.reset(final)
            self.assertEqual(guard.held, {'llm': 0, 'embedding': 0})
        finally:
            ACTIVE_USAGE.reset(token)
