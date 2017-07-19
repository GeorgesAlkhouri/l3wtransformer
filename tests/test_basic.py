import unittest
from context import l3wtransformer

class Testl3wtransformerMethods(unittest.TestCase):

    def test_word_to_ngrams(self):

        l3wt = l3wtransformer.L3wTransformer(trigram_size=3)
        self.assertEqual(l3wt.word_to_ngrams(''), [])
        self.assertEqual(l3wt.word_to_ngrams('aa'), ['#aa', 'aa#'])

        l3wt = l3wtransformer.L3wTransformer(trigram_size=3, mark_char='ö')
        self.assertEqual(l3wt.word_to_ngrams('a'), ['öaö'])


    def test_scan_paragraphs(self):
        l3wt = l3wtransformer.L3wTransformer()
        self.assertEqual(l3wt.scan_paragraphs([]), {})
        self.assertEqual(l3wt.scan_paragraphs(['a', 'b']), {'#a#': 1, '#b#': 1})
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

    # def test_texts_to_sequences(self):
    #     #self.assertEqual(l3wtransformer.text_to_sequence())
    #     l3wt = l3wtransformer.L3wTransformer()
    #     print(l3wt.texts_to_sequences(['Das ist eine Test, Iphone 5s', 'Haushaufgaben sind naja.']))




if __name__ == '__main__':
    unittest.main()
