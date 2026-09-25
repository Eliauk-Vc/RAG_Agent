import copy
import unittest

from rag_app.context import packed, rrf_evidence


def row(identifier):
    return {'chunk_id': identifier, 'content': '原文', 'file_path': 'book.txt'}


class RRFTests(unittest.TestCase):
    def test_shared_evidence_outranks_single_list_and_formula(self):
        batches = [[row('a'), row('shared')], [row('b'), row('shared')]]
        original = copy.deepcopy(batches)
        result = rrf_evidence(batches, 10000)
        self.assertEqual(list(result), ['shared', 'a', 'b'])
        self.assertAlmostEqual(result['shared']['rrf_score'], 2 / 62)
        self.assertEqual(result['shared']['rrf_support'], 2)
        self.assertEqual(batches, original)

    def test_duplicates_do_not_add_votes_or_shift_unique_rank(self):
        result = rrf_evidence([[row('a'), row('a'), row('b')]], 10000)
        self.assertEqual(result['a']['rrf_support'], 1)
        self.assertAlmostEqual(result['a']['rrf_score'], 1 / 61)
        self.assertAlmostEqual(result['b']['rrf_score'], 1 / 62)

    def test_budget_includes_rank_metadata(self):
        batches = [[row('a')], [row('b'), row('a')]]
        full = rrf_evidence(batches, 10000)
        budget = len(packed([full['a']]).encode('utf-8'))
        result = rrf_evidence(batches, budget)
        self.assertEqual(list(result), ['a'])
        self.assertLessEqual(len(packed(list(result.values())).encode('utf-8')), budget)
        self.assertEqual(rrf_evidence([], 100), {})

    def test_single_subtask_order_and_invalid_parameters(self):
        self.assertEqual(list(rrf_evidence([[row('b'), row('a')]], 10000)), ['b', 'a'])
        with self.assertRaises(ValueError):
            rrf_evidence([], 100, 0)
