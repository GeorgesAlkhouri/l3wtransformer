import operator
import logging
import pickle
from functools import reduce
from itertools import chain

from pathos.multiprocessing import ProcessingPool as Pool
from nltk.util import ngrams
import numpy as np


# TODO white spaces?
class L3wTransformer:
    """
    Parameters
    ----------
    max_ngrams : The upper bound of the top n most frequent ngrams to be used. If None use all containing ngrams.
    ngram_size : The size of the ngrams.
    lower : Should the ngrams be treated as lower char ngrams.
    split_char : Delimeter for splitting strings into a list of words.
    """

    def __init__(self, max_ngrams=50000, ngram_size=3, lower=True, split_char=None, parallelize=False):
        self.ngram_size = ngram_size
        self.lower = lower
        self.split_char = split_char
        self.max_ngrams = max_ngrams
        self.indexed_lookup_table = {}
        self.parallelize = parallelize

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            dump_dict = pickle.load(f)

        l3wt = L3wTransformer(
            max_ngrams=dump_dict['max_ngrams'],
            ngram_size=dump_dict['ngram_size'],
            lower=dump_dict['lower'],
            split_char=dump_dict['split_char'],
            parallelize=dump_dict['parallelize']
        )

        l3wt.indexed_lookup_table = dump_dict['indexed_lookup_table']
        return l3wt

    ### Helper Start ###

    def __flags_from_word(self, word):

        # flags = []
        #
        # isIc = word[0].isupper()
        # isUp = word.isupper()
        # isLo = word.islower()
        # isMc = not isIc or (not isUp and not isLo)
        #
        # if isIc and not isUp:
        #     flags.append('ic')
        # elif isIc and isUp:
        #     flags.append('up')
        # elif isLo:
        #     flags.append('lo')
        #
        # if not word[1:].islower() and not word[1:].isupper() and word.isalpha():
        #     # 'aB-' does not count as mixed word, only pure words which containing mixed cased chars are counting
        #     flags.append('mc')

        if word.lower() == word:
            return ['lo']
        elif word.upper() == word:
            return ['up']
        elif word[0].upper() == word[0]:
            return ['ic']
        else:
            return ['mc']

    def __flags_to_sequence(self, flags, base_value=0):
        flag_seq = []

        for flag in flags:
            if flag == 'ic':
                flag_seq.append(base_value + 1)
            elif flag == 'up':
                flag_seq.append(base_value + 2)
            elif flag == 'lo':
                flag_seq.append(base_value + 3)
            elif flag == 'mc':
                flag_seq.append(base_value + 4)
            elif flag == 'unk':
                flag_seq.append(base_value + 5)
            else:
                raise Exception('Unknown flag value')

        return flag_seq

    ### Helper End ###

    def save(self, path):
        dump_dict = {
            'ngram_size': self.ngram_size,
            'lower': self.lower,
            'split_char': self.split_char,
            'max_ngrams': self.max_ngrams,
            'indexed_lookup_table': self.indexed_lookup_table,
            'parallelize': self.parallelize
        }
        with open(path, 'wb') as f:
            pickle.dump(dump_dict, f)

    def word_to_ngrams(self, word):
        """Returns a list of all n-gram possibilities of the given word."""
        if self.lower:
            word = word.lower()
        word = '<' + word + '>'
        return list(map(lambda x: ''.join(x), list(ngrams(word, self.ngram_size))))

    def scan_paragraphs(self, paragraphs):
        """Creates a lookup table from the given paragraphs, containing all
        n-gram frequencies."""
        lookup_table = {}
        paras_len = len(paragraphs)

        if self.parallelize:
            with Pool(None) as p:
                paragraphs = p.map(lambda para: para.split(
                    self.split_char), paragraphs)
        else:
            paragraphs = map(lambda para: para.split(
                self.split_char), paragraphs)

        def fill_lookup_table(lookup_table, words):
            for w in words:
                ngrams_w = self.word_to_ngrams(w)
                for n in ngrams_w:
                    if n in lookup_table:
                        lookup_table[n] = lookup_table[n] + 1
                    else:
                        lookup_table[n] = 1
            return lookup_table

        lookup_table = reduce(fill_lookup_table, paragraphs, {})

        return lookup_table

    def text_to_sequence(self, text, indexed_lookup_table):
        """Transforms a list of strings into a list of integer sequences from
        indexed lookup table.
        Relaces unknown trigrams with self.max_ngrams + 5.
        """

        def word_to_ngrams_and_flags(word):
            ngrams_w = self.word_to_ngrams(word)
            flags = self.__flags_to_sequence(
                self.__flags_from_word(word), base_value=self.max_ngrams)
            return ngrams_w, flags

        trigrams = list(map(word_to_ngrams_and_flags,
                            text.split(self.split_char)))

        seq = []
        unknown = self.max_ngrams + 5
        for ngrams_w, flags in trigrams:
            for trigram in ngrams_w:
                if trigram in indexed_lookup_table:
                    seq.append(indexed_lookup_table[trigram])
                else:
                    seq.append(unknown)
                    logging.info(str(trigram) +
                                 ' not in indexed lookup table.')
            if len(list(filter(lambda x: x != unknown, seq))) > 0:
                for flag in flags:
                    seq.append(flag)
        return seq

    def texts_to_hot_vectors(self, texts):

        def text_to_hot_vec(text):
            hot_vec = np.zeros(len(self.indexed_lookup_table), dtype='int')
            trigrams = list(chain.from_iterable(map(self.word_to_ngrams,
                                                    text.split(self.split_char))))
            for tri in trigrams:
                if tri in self.indexed_lookup_table:
                    index = self.indexed_lookup_table[tri] - 1
                    hot_vec[index] = hot_vec[index] + 1
                else:
                    logging.info(str(tri) +
                                 ' not in indexed lookup table.')

            return list(hot_vec)

        if not texts:
            return []

        return list(map(text_to_hot_vec, texts))

    def texts_to_sequences(self, texts):
        """Convenient method to tansform new texts into integer sequences."""
        if not texts:
            return []

        if self.parallelize:
            with Pool(None) as p:
                res = p.map(lambda text: self.text_to_sequence(
                    text, self.indexed_lookup_table), texts)
        else:
            res = list(map(lambda text: self.text_to_sequence(
                text, self.indexed_lookup_table), texts))

        return res

    def fit_on_texts(self, texts):
        """Convenient method for creating a indexed lookup table,
        necessary to transform text into a integer sequence. Always call this before
        texts_to_sequences method to get results.
        Returns the indexed lookup table."""
        if not texts:
            return []

        lookup_table = self.scan_paragraphs(texts)

        if not lookup_table:
            return {}

        if not self.max_ngrams:
            self.max_ngrams = len(lookup_table)

        # before cutted_lookup_table.items() are sorted by their counts, sort cutted_lookup_table.items() alphabetical
        # to preserve same order by items with same count.
        # Example: cutted_lookup_table.items() could by [('aa', 1), ('a', 1)] or [('a', 1), ('aa', 1)]
        sorted_lookup = sorted(lookup_table.items(),
                               key=operator.itemgetter(0), reverse=False)
        cutted_lookup_table = dict(sorted_lookup[:self.max_ngrams])

        indexed_lookup_table = dict(
            zip(list(zip(*sorted_lookup[:self.max_ngrams]))[0],  # get only the max_ngrams frequent tri grams
                list(range(1, self.max_ngrams + 1)))
        )

        self.indexed_lookup_table = indexed_lookup_table
        return self.indexed_lookup_table
