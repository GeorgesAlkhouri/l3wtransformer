import unittest
import time

from context import l3wtransformer
from sklearn.datasets import fetch_20newsgroups

class Testl3wtransformerBenchmark(unittest.TestCase):

    def test_benchmark(self):
        newsgroups_train = fetch_20newsgroups(subset='test')
        l3wt = l3wtransformer.L3wTransformer(max_ngrams=None)

        print('')
        print('Starting benchmark')

        start = time.time()
        lookup_table = l3wt.fit_on_texts(newsgroups_train.data)
        end = time.time()

        print('Execution time - fit_on_texts:', end - start, 's')

        start = time.time()
        lookup_table = l3wt.texts_to_sequences(newsgroups_train.data)
        end = time.time()

        print('Execution time - texts_to_sequences:', end - start, 's')

    def test_parallel_benchmark(self):
        newsgroups_train = fetch_20newsgroups(subset='test')
        l3wt = l3wtransformer.L3wTransformer(max_ngrams=None, parallelize=True)

        print('')
        print('Starting parallel benchmark')

        start = time.time()
        lookup_table = l3wt.fit_on_texts(newsgroups_train.data)
        end = time.time()

        print('Execution time - fit_on_texts:', end - start, 's')

        start = time.time()
        lookup_table = l3wt.texts_to_sequences(newsgroups_train.data)
        end = time.time()

        print('Execution time - texts_to_sequences:', end - start, 's')


if __name__ == '__main__':
    unittest.main()
