import unittest
from context import l3wtransformer


class Testl3wtransformerMethods(unittest.TestCase):

    def test_max_ngrams(self):
        l3wt = l3wtransformer.L3wTransformer(max_ngrams=2)
        lookup_table = l3wt.fit_on_texts(['aaa bbb ccc acb'])
        self.assertEqual(len(lookup_table), 2)
        self.assertEqual(l3wt.max_ngrams, 2)

        l3wt = l3wtransformer.L3wTransformer(max_ngrams=None)
        lookup_table = l3wt.fit_on_texts(['abcdef'])
        self.assertEqual(len(lookup_table), 6)
        self.assertEqual(l3wt.max_ngrams, 6)

    def test_word_to_ngrams(self):

        l3wt = l3wtransformer.L3wTransformer(ngram_size=3)
        self.assertEqual(l3wt.word_to_ngrams(''), [])
        self.assertEqual(l3wt.word_to_ngrams('aa'), ['#aa', 'aa#'])

        l3wt = l3wtransformer.L3wTransformer(ngram_size=3, mark_char='ö')
        self.assertEqual(l3wt.word_to_ngrams('a'), ['öaö'])

    def test_scan_paragraphs(self):
        l3wt = l3wtransformer.L3wTransformer()
        self.assertEqual(l3wt.scan_paragraphs([]), {})
        self.assertEqual(l3wt.scan_paragraphs(
            ['a', 'b']), {'#a#': 1, '#b#': 1})
        self.assertEqual(l3wt.scan_paragraphs(['a', 'a']), {'#a#': 2})
        self.assertEqual(l3wt.scan_paragraphs(['a b']), {'#a#': 1, '#b#': 1})

        l3wt = l3wtransformer.L3wTransformer(split_char='ö')
        self.assertEqual(l3wt.scan_paragraphs(['aöb']), {'#a#': 1, '#b#': 1})

        with self.assertRaises(Exception):
            l3wt = l3wtransformer.L3wTransformer()
            l3wt.scan_paragraphs(5)
            l3wt.scan_paragraphs([4])
            l3wt.scan_paragraphs('s')
            l3wt.scan_paragraphs(['a', 5])

    def test_texts_to_sequences(self):
        l3wt = l3wtransformer.L3wTransformer()
        self.assertEqual(l3wt.texts_to_sequences(
            ['Aaa aAa aaa AAA', 'Abb aba BbB']), [[], []])

        l3wt.fit_on_texts(['aaa'])
        self.assertEqual(l3wt.texts_to_sequences(['Aaa aAa aaa AAA', 'Abb aba BbB']), [
                         [1, 3, 2, 50001, 1, 3, 2, 50004, 1, 3, 2, 50003, 1, 3, 2, 50002], []])

        with self.assertRaises(Exception):
            l3wt = l3wtransformer.L3wTransformer()
            l3wt.texts_to_sequences([[], []])
            l3wt.texts_to_sequences([5, 1])


if __name__ == '__main__':
    unittest.main()
