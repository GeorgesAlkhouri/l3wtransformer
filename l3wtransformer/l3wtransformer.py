from functools import reduce
from itertools import chain
import operator
import logging

from nltk.util import ngrams

class L3wTransformer:

    def __init__(self, max_words=50000, trigram_size=3, lower=True, mark_char='#', split_char=' '):
        self.trigram_size = trigram_size
        self.lower = lower
        self.mark_char = mark_char
        self.split_char = split_char
        self.max_words = max_words

    ### Helper Start ###

    def __flags_from_word(self, word):

        flags = []

        isIc = word[0].isupper()
        isUp = word.isupper()
        isLo = word.islower()
        isMc = not isIc or (not isUp and not isLo)

        if isIc and not isUp:
            flags.append('ic')
        elif isIc and isUp:
            flags.append('up')
        elif isLo:
            flags.append('lo')

        if not word[1:].islower() and not word[1:].isupper() and word.isalpha():
            flags.append('mc') # aB- zählt nicht als Mixed nur reihne wörter welche Mixed case enthalten werden auch als dieses gezählt

        return flags

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
            else:
                raise Exception('Unkown flag value')

        return flag_seq

    def __add_flags_to_indexed_lookup_table(self, base_value, indexed_lookup_table):
        #Flags
        #initial capitalization,
        ic = base_value + 1
        #uppercase
        up = base_value + 2
        #lower case
        lo = base_value + 3
        #mixed case
        mc = base_value + 4

        # add flags to indexed_lookup_table

        indexed_lookup_table[ic] = ic
        indexed_lookup_table[up] = up
        indexed_lookup_table[lo] = lo
        indexed_lookup_table[mc] = mc

        return indexed_lookup_table

    ### Helper End ###

    def word_to_ngrams(self, word):
        if self.lower:
            word = word.lower()
        word = self.mark_char + word + self.mark_char
        return list( map(lambda x: ''.join(x), list(ngrams(word, self.trigram_size))) )

    def scan_paragraphs(self, paragraphs):
        lookup_table = {}
        paras_len = len(paragraphs)

        for idx, para in enumerate(paragraphs):

            logging.info(str(idx+1) + ' of ' + str(paras_len))
            # sys.stdout.write(str(idx+1) + ' of ' + str(paras_len)+'\r')
            # sys.stdout.flush()

            words = para.split(self.split_char)

            for w in words:
                ngrams_w = self.word_to_ngrams(w)
                for n in ngrams_w:
                    if n in lookup_table:
                        lookup_table[n] = lookup_table[n] + 1
                    else:
                        lookup_table[n] = 1
        return lookup_table

    def text_to_sequence(self, text, indexed_lookup_table):
        trigrams = []

        for word in text.split(self.split_char):
            ngrams_w = self.word_to_ngrams(word)
            flags = self.__flags_to_sequence(self.__flags_from_word(word), base_value=self.max_words)
            trigrams.append( (ngrams_w, flags) )

        seq = []
        for trigram_tuple in trigrams:
            for trigram in trigram_tuple[0]:
                if trigram in indexed_lookup_table:
                    seq.append(indexed_lookup_table[trigram])
                else:
                    logging.info(str(trigram) + ' not in indexed lookup table.')
            if seq:
                for flag in trigram_tuple[1]:
                    seq.append(flag)
        return seq

    def texts_to_sequences(self, texts):
        lookup_table = self.scan_paragraphs(texts)
        cutted_lookup_table = dict(sorted(lookup_table.items(), key=operator.itemgetter(1), reverse=True)[:self.max_words])
        sorted_lookup = sorted(cutted_lookup_table.items(), key=operator.itemgetter(1), reverse=True)
        indexed_lookup_table = dict(
                            zip(list(zip(*sorted_lookup[:self.max_words]))[0], # get only the max_words frequent tri grams
                                list(range(1, self.max_words + 1)))
                            )

        indexed_lookup_table = self.__add_flags_to_indexed_lookup_table(self.max_words, indexed_lookup_table)
        return list(map(lambda text: self.text_to_sequence(text, indexed_lookup_table), texts))
