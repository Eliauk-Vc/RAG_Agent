import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag_app import evaluate as EVALUATOR  # noqa: E402


class BookEvaluationTests(unittest.TestCase):
    def test_load_cases_rejects_duplicate_ids(self) -> None:
        payload = {
            "cases": [
                {
                    "id": "same",
                    "question": "first",
                    "expected_keywords": ["fact"],
                    "expected_source": "book.txt",
                },
                {
                    "id": "same",
                    "question": "second",
                    "expected_keywords": ["fact"],
                    "expected_source": "book.txt",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cases.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing or duplicate id"):
                EVALUATOR.load_cases(path)

    def test_parse_all_strategies_includes_baseline_and_rerank(self) -> None:
        strategies = EVALUATOR.parse_strategies("all")
        names = [strategy.name for strategy in strategies]
        self.assertIn("bypass", names)
        self.assertIn("hybrid", names)
        self.assertIn("hybrid+rerank", names)
        self.assertIn("agent", names)

    def test_score_result_measures_answer_evidence_and_source(self) -> None:
        case = EVALUATOR.EvaluationCase(
            case_id="course",
            question="病程是什么？",
            expected_keywords=("1个月", "2周"),
            expected_source="book.txt",
        )
        strategy = EVALUATOR.EvaluationStrategy(
            name="hybrid+rerank",
            mode="hybrid",
            enable_rerank=True,
        )
        result = {
            "status": "success",
            "data": {
                "entities": [{"entity_name": "精神分裂症"}],
                "relationships": [{"src_id": "精神分裂症"}],
                "chunks": [
                    {
                        "content": "病程至少1个月，特定情况下分裂症状持续2周。",
                        "file_path": "D:\\project\\book.txt",
                    }
                ],
                "references": [
                    {"reference_id": "1", "file_path": "D:\\project\\book.txt"}
                ],
            },
            "llm_response": {
                "content": "病程至少 **1 个月**，相关症状还需持续 **2 周**。"
            },
        }

        row = EVALUATOR.score_result(
            case,
            strategy,
            result,
            latency_seconds=1.25,
            usage={
                "call_count": 2,
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        )

        self.assertEqual(row["answer_keyword_coverage"], 1.0)
        self.assertEqual(row["evidence_keyword_coverage"], 1.0)
        self.assertTrue(row["source_hit"])
        self.assertTrue(row["citation_present"])
        self.assertEqual(row["total_tokens"], 120)


if __name__ == "__main__":
    unittest.main()
