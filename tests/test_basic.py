import unittest
import numpy as np
from context import l3wtransformer

import os


class Testl3wtransformerMethods(unittest.TestCase):

    def tearDown(self):
        path = './tests/temp_l3wt_dict'
        os.remove(path) if os.path.exists(path) else None

    def test_max_ngrams(self):
        l3wt = l3wtransformer.L3wTransformer(max_ngrams=2)
        lookup_table = l3wt.fit_on_texts(['aaa bbb ccc acb'])
        self.assertEqual(len(lookup_table), 2)
        self.assertEqual(l3wt.max_ngrams, 2)

        l3wt = l3wtransformer.L3wTransformer(max_ngrams=None)
        lookup_table = l3wt.fit_on_texts(['abcdef'])
        self.assertEqual(len(lookup_table), 6)
        self.assertEqual(l3wt.max_ngrams, 6)

        l3wt = l3wtransformer.L3wTransformer(max_ngrams=3)
        lookup_table = l3wt.fit_on_texts(['abc'])

        self.assertEqual(lookup_table, {'<ab': 1, 'bc>': 3, 'abc': 2})

        l3wt = l3wtransformer.L3wTransformer(max_ngrams=4)
        lookup_table = l3wt.fit_on_texts(['abc'])

        self.assertEqual(lookup_table, {'<ab': 1, 'bc>': 3, 'abc': 2})

        l3wt = l3wtransformer.L3wTransformer(max_ngrams=2)
        lookup_table = l3wt.fit_on_texts(['abcd'])

        self.assertEqual(lookup_table, {'<ab': 1, 'abc': 2})

        l3wt = l3wtransformer.L3wTransformer(max_ngrams=4)
        lookup_table = l3wt.fit_on_texts(['abc adc'])

        self.assertEqual(
            lookup_table, {'<ab': 1, '<ad': 2, 'abc': 3, 'adc': 4})

    def test_word_to_ngrams(self):

        l3wt = l3wtransformer.L3wTransformer(ngram_size=3)
        self.assertEqual(l3wt.word_to_ngrams(''), [])
        self.assertEqual(l3wt.word_to_ngrams('aa'), ['<aa', 'aa>'])

    def test_scan_paragraphs(self):
        l3wt = l3wtransformer.L3wTransformer()
        self.assertEqual(l3wt.scan_paragraphs([]), {})
        self.assertEqual(l3wt.scan_paragraphs(
            ['a', 'b']), {'<a>': 1, '<b>': 1})
        self.assertEqual(l3wt.scan_paragraphs(['a', 'a']), {'<a>': 2})
        self.assertEqual(l3wt.scan_paragraphs(['a b']), {'<a>': 1, '<b>': 1})

        l3wt = l3wtransformer.L3wTransformer(split_char='ö')
        self.assertEqual(l3wt.scan_paragraphs(['aöb']), {'<a>': 1, '<b>': 1})

        with self.assertRaises(Exception):
            l3wt = l3wtransformer.L3wTransformer()
            l3wt.scan_paragraphs(5)
            l3wt.scan_paragraphs([4])
            l3wt.scan_paragraphs('s')
            l3wt.scan_paragraphs(['a', 5])

    def test_empty(self):
        l3wt = l3wtransformer.L3wTransformer(max_ngrams=3)

        lookup_table = l3wt.fit_on_texts(['', ''])
        self.assertEqual(l3wt.texts_to_sequences(['']), [[]])

        lookup_table = l3wt.fit_on_texts(['', 'addsd'])
        self.assertEqual(l3wt.texts_to_sequences(['']), [[]])

        lookup_table = l3wt.fit_on_texts([''])
        self.assertEqual(l3wt.texts_to_sequences(['', '']), [[], []])

    def test_texts_to_sequences(self):
        l3wt = l3wtransformer.L3wTransformer()

        self.assertEqual(l3wt.texts_to_sequences(
            ['a', 'b']), [[50005], [50005]])

        lookup_table = l3wt.fit_on_texts(['aaa'])
        self.assertEqual(l3wt.texts_to_sequences(['Aaa aAa aaa AAA', 'Abb aba BbB']), [
                         [1, 3, 2, 50001, 1, 3, 2, 50004, 1, 3, 2, 50003, 1, 3, 2, 50002], [50005, 50005, 50005, 50005, 50005, 50005, 50005, 50005, 50005]])

        with self.assertRaises(Exception):
            l3wt = l3wtransformer.L3wTransformer()
            l3wt.texts_to_sequences([[], []])
            l3wt.texts_to_sequences([5, 1])

    # def test_unknown_word(self):
    #     max_ngrams = 50000
    #     l3wt = l3wtransformer.L3wTransformer(max_ngrams=max_ngrams)
    #     unknown_flag = max_ngrams + 5
    #     l3wt.fit_on_texts(['a'])
    #     self.assertEqual(l3wt.texts_to_sequences(['b']), [[unknown_flag]])
    # def test_numpy_input(self):
    #     l3wt = l3wtransformer.L3wTransformer()
    #     self.assertEqual(l3wt.texts_to_sequences(
    #         np.array(['Aaa aAa aaa AAA', 'Abb aba BbB'])), [[], []])
    #
    #     lookup_table = l3wt.fit_on_texts(np.array(['aaa']))
    #     self.assertEqual(l3wt.texts_to_sequences(['Aaa aAa aaa AAA', 'Abb aba BbB']), [
    #                      [1, 3, 2, 50001, 1, 3, 2, 50004, 1, 3, 2, 50003, 1, 3, 2, 50002], []])
    #
    #     with self.assertRaises(Exception):
    #         l3wt = l3wtransformer.L3wTransformer()
    #         l3wt.texts_to_sequences([[], []])
    #         l3wt.texts_to_sequences([5, 1])

    def test_save_and_load(self):

        path = './tests/temp_l3wt_dict'

        test_indexed_lookup_table = {'<ab': 1, 'abc': 2, 'bc>': 3}
        ngram_size = 3
        lower = True
        split_char = None
        max_ngrams = 100
        parallelize = True

        unknown = 105

        l3wt = l3wtransformer.L3wTransformer(
            ngram_size=3, lower=True, split_char=None, max_ngrams=100, parallelize=parallelize)
        l3wt.fit_on_texts(['abc'])

        l3wt.save(path)
        self.assertEqual(os.path.exists(path), True)

        loaded_l3wt = l3wtransformer.L3wTransformer.load(path)
        self.assertEqual(isinstance(
            loaded_l3wt, l3wtransformer.L3wTransformer), True)
        self.assertEqual(ngram_size, loaded_l3wt.ngram_size)
        self.assertEqual(lower, loaded_l3wt.lower)
        self.assertEqual(split_char, loaded_l3wt.split_char)
        self.assertEqual(max_ngrams, loaded_l3wt.max_ngrams)
        self.assertEqual(parallelize, loaded_l3wt.parallelize)
        self.assertEqual(test_indexed_lookup_table,
                         loaded_l3wt.indexed_lookup_table)
        self.assertEqual(loaded_l3wt.texts_to_sequences(
            ['ab']), [[1, unknown, 103]])

    def test_to_hot_vector(self):
        l3wt = l3wtransformer.L3wTransformer()
        l3wt.fit_on_texts(['abc'])
        res = l3wt.texts_to_hot_vectors(['abc'])
        self.assertEquals(res, [[1, 1, 1]])

        res = l3wt.texts_to_hot_vectors(['ab'])
        self.assertEquals(res, [[1, 0, 0]])

        res = l3wt.texts_to_hot_vectors([''])
        self.assertEquals(res, [[0, 0, 0]])

        res = l3wt.texts_to_hot_vectors(['ab abc',])
        self.assertEquals(res, [[2, 1, 1]])

if __name__ == '__main__':
    unittest.main()
