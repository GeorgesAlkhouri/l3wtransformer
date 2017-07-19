import unittest
from context import l3wtransformer

class Testl3wtransformerMethods(unittest.TestCase):

    def test_word_to_ngrams(self):
        self.assertEqual(l3wtransformer.word_to_ngrams('', n=3), [])
        self.assertEqual(l3wtransformer.word_to_ngrams('a', n=3, mark_char='ö'), ['öaö'])
        self.assertEqual(l3wtransformer.word_to_ngrams('aa', n=3), ['#aa', 'aa#'])

    def test_scan_paragraphs(self):
        self.assertEqual(l3wtransformer.scan_paragraphs([]), {})
        self.assertEqual(l3wtransformer.scan_paragraphs(['a', 'b']), {'#a#': 1, '#b#': 1})
        self.assertEqual(l3wtransformer.scan_paragraphs(['a', 'a']), {'#a#': 2})

        self.assertEqual(l3wtransformer.scan_paragraphs(['a b']), {'#a#': 1, '#b#': 1})
        self.assertEqual(l3wtransformer.scan_paragraphs(['aöb'], split_char='ö'), {'#a#': 1, '#b#': 1})

        with self.assertRaises(Exception):
            l3wtransformer.scan_paragraphs(5)
            l3wtransformer.scan_paragraphs([4])
            l3wtransformer.scan_paragraphs('s')
            l3wtransformer.scan_paragraphs(['a', 5])

    # def test_text_to_sequence(self):
    #     self.assertEqual(l3wtransformer.text_to_sequence())
    #
    #     #print(l3wtransformer.flags_from_word('Hallo'))




if __name__ == '__main__':
    unittest.main()
