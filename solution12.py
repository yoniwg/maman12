## general imports
import operator
import random
import itertools
from collections import defaultdict
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # data splitter
from sklearn.linear_model import LogisticRegression
import re

## project supplied imports
from submission_specs.SubmissionSpec12 import SubmissionSpec12
import re

DELTA = 0.001
START_LABEL = '<S>'
UNKNOWN = ''


SPECIAL_PATTERNS = [
     '\-?[0-9]+(\.[0-9]+)?',                 # numbers
     '\+?[0-9]+(\-[0-9]+)',                  # phone-numbers
     '[0-9]+([:\.\/\\\]+[0-9]+)+',           # date/time
     '\w+([:@\.\/\\\]+\w+)+',                # internet addresses, emails, etc.
     '[^A-Za-z0-9]{2,}',                    # characters
]
class Submission(SubmissionSpec12):
    def __init__(self):
        self.emission_p = defaultdict(lambda: defaultdict(float))
        self.transition_p = defaultdict(lambda: defaultdict(float))

    @staticmethod
    def _lower_first(str):
        """Lower the first letter if the rest of the word is in lower case"""
        rest_of_word = str[1:]
        return str[0].lower() + rest_of_word if rest_of_word.lower() == rest_of_word else str

    def _estimate_emission_probabilites(self, annotated_sentences):
        emission_tag_count = defaultdict(lambda: [0, defaultdict(int)])
        for sentence in annotated_sentences:
            no_first_capital_sentence = [(self._lower_first(sentence[0][0]), sentence[0][1])] + sentence[1:]
            for word, tag in no_first_capital_sentence:
                emission_tag_count[tag][0] += 1
                emission_tag_count[tag][1][self._get_word_or_pattern(word)] += 1
        for tag, data in emission_tag_count.items():
            count, word_dict = data
            word_dict[UNKNOWN] = 0
            count += len(word_dict) * DELTA
            for word in word_dict:
                word_prob = (word_dict[word] + DELTA) / count
                self.emission_p[tag][word] = word_prob

    @staticmethod
    def _get_word_or_pattern(word):
        for patt in SPECIAL_PATTERNS:
            if re.fullmatch(patt, word):
                return patt
        return word

    def _estimate_transition_probabilites(self, annotated_sentences):
        transition_count = defaultdict(lambda: [0, defaultdict(int)])
        for sentence in annotated_sentences:
            prev_tag = START_LABEL
            for word, tag in sentence:
                transition_count[prev_tag][0] += 1
                transition_count[prev_tag][1][tag] += 1
                prev_tag = tag

        for tag, data in transition_count.items():
            count, next_tags_dict = data
            for next_tag in next_tags_dict:
                self.transition_p[tag][next_tag] = next_tags_dict[next_tag] / count

    def train(self, annotated_sentences):
        ''' trains the HMM model (computes the probability distributions) '''

        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))

        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)

        return self

    def predict(self, sentence):
        tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
        viterbi_table = self._init_table(tag_set, sentence)
        self._fill_table(viterbi_table, tag_set, sentence)
        prediction = self._compute_backtrace(viterbi_table, tag_set)
        assert (len(prediction) == len(sentence)), "prediction length is too short"
        return prediction

    def _init_table(self, tag_set, sentence, start_label=START_LABEL):
        initial_table = [[(-1, 0.0) for _ in tag_set] for _ in sentence]
        first_word = self._lower_first(sentence[0])
        for idx, tag in enumerate(tag_set):
            emission_p = self._get_emission_prob(tag, first_word)
            transition_p = self.transition_p[start_label][tag]
            initial_table[0][idx] = (-1, emission_p * transition_p)
        return initial_table

    def _fill_table(self, viterbi_table, tag_set, sentence):
        for w_idx, word in enumerate(sentence):
            if w_idx == 0: continue
            for t_idx, tag in enumerate(tag_set):
                emission_p = self._get_emission_prob(tag, word)
                max_prob = (-1, 0.0)
                for prev_t_idx, prob in enumerate(viterbi_table[w_idx - 1]):
                    transition_p = self.transition_p[tag_set[prev_t_idx]][tag]
                    cur_prob = prob[1] * transition_p
                    if cur_prob > max_prob[1]:
                        max_prob = (prev_t_idx, cur_prob)
                # We multiply the real value by 100 in order to avoid floating point overflow
                # This does not change the final result since all the column grows accordingly
                viterbi_table[w_idx][t_idx] = (max_prob[0], emission_p * max_prob[1] * 100)

    def _get_emission_prob(self, tag, word):
        return self.emission_p[tag][self._get_word_or_pattern(word)] or self.emission_p[tag][UNKNOWN]

    def _compute_backtrace(self, viterbi_table, tag_set):
        backtrace = []
        cur_elem = max(viterbi_table[-1], key=operator.itemgetter(1))
        backtrace.insert(0, tag_set[viterbi_table[-1].index(cur_elem)])
        for col in reversed(viterbi_table[0:-1]):
            if cur_elem[0] == -1 or cur_elem[1] == 0.0: break
            backtrace.insert(0, tag_set[cur_elem[0]])
            cur_elem = col[cur_elem[0]]
        return backtrace
