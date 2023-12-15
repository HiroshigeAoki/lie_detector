import unittest
from collections import Counter
import sys, os
from src.utils.create_ngram_list import calc_diff_ngrams


def test_calc_diff_ngrams(self):
    label_0_ngram_counts = Counter({"a": 0, "b": 100, "c": 3, "d": 5})
    label_1_ngram_counts = Counter({"a": 1, "b": 2, "c": 3, "d": 4})
    sorted_ngrams = calc_diff_ngrams(label_0_ngram_counts, label_1_ngram_counts)
    expected = [("d", 1), ("c", 0), ("b", 98), ("a", -1)]
    
    # 順序に依存しない比較
    self.assertCountEqual(sorted_ngrams, expected)
        
if __name__ == "__main__":
    unittest.main()