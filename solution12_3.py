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
    '\-?[0-9]+(\.[0-9]+)?',  # numbers
    '\+?[0-9]+(\-[0-9]+)',  # phone-numbers
    '[0-9]+([:\.\/\\\]+[0-9]+)+',  # date/time
    '\w+([:@\.\/\\\]+\w+)+',  # internet addresses, emails, etc.
    '[^A-Za-z0-9]{2,}',  # characters
]


class Submission(SubmissionSpec12):
    def __init__(self):
        self.emission_p = defaultdict(lambda: defaultdict(float))
        self.transition_p = defaultdict(lambda: defaultdict(float))
        self.transition_trigram_p = defaultdict(lambda: defaultdict(float))
       # self.transition_p = defaultdict(lambda: defaultdict(lambda: defaultdict(float))) TTTTT

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
        transition_trigram_count = defaultdict(lambda: [0, defaultdict(int)])
        for sentence in annotated_sentences:
            prev_tag = START_LABEL
            prev_prev_tag = START_LABEL
            for _, tag in sentence:
                transition_count[prev_tag][0] += 1
                transition_count[prev_tag][1][tag] += 1
                transition_trigram_count[(prev_prev_tag, prev_tag)][0] += 1
                transition_trigram_count[(prev_prev_tag, prev_tag)][1][tag] += 1
                prev_prev_tag = prev_tag
                prev_tag = tag

        for tag, data in transition_count.items():
            count, next_tags_dict = data
            for next_tag in next_tags_dict:
                self.transition_p[tag][next_tag] = next_tags_dict[next_tag] / count

        for tags, data in transition_trigram_count.items():
            count, next_tags_dict = data
            for next_tag in next_tags_dict:
                self.transition_trigram_p[tags][next_tag] = (0.9 * next_tags_dict[next_tag] / count +
                                                             0.1 * self.emission_p[tags[1]][next_tag])

    def train(self, annotated_sentences):
        ''' trains the HMM model (computes the probability distributions) '''

        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)

        return self

    def predict(self, sentence, top_n=1):
        tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
        viterbi_table = self._init_table(tag_set, sentence)
        self._fill_table(viterbi_table, tag_set, sentence)
        prediction = self._compute_backtrace(viterbi_table, tag_set)
        assert (len(prediction) == len(sentence)), "prediction length is too short"
        return prediction

    def _init_table(self, tag_set, sentence):
        initial_table = [[[((-1, -1), 0.0) for _ in tag_set] for _ in tag_set] for _ in sentence]
        first_word = self._lower_first(sentence[0])
        for idx1, tag in enumerate(tag_set):
            emission_p = self._get_emission_prob(tag, first_word)
            transition_p = self.transition_trigram_p[(START_LABEL, START_LABEL)][tag]
            initial_table[0][0][idx1] = ((-1, -1), emission_p * transition_p * 100)

        if len(sentence) > 1:
            second_word = sentence[1]
            for idx1, tag in enumerate(tag_set):
                for prev_t_idx, prev_tag in enumerate(tag_set):
                    emission_p = self._get_emission_prob(tag, second_word)
                    transition_p = self.transition_trigram_p[(START_LABEL, prev_tag)][tag]
                    _, prev_cell_prob = initial_table[0][0][prev_t_idx]
                    initial_table[1][prev_t_idx][idx1] = ((0, prev_t_idx), emission_p * transition_p * prev_cell_prob * 100)
        return initial_table

    def _fill_table(self, viterbi_table, tag_set, sentence):
        for w_idx, word in enumerate(sentence):
            if w_idx < 2:
                continue
            for t_idx, tag in enumerate(tag_set):
                emission_p = self._get_emission_prob(tag, word)
                for prev_t_idx, prev_tag in enumerate(tag_set):
                    max_prob = 0
                    max_cell = (-1, -1)
                    for prev_prev_t_idx, prev_prev_tag in enumerate(tag_set):
                        relevant_cell, relevant_prob = viterbi_table[w_idx - 1][prev_prev_t_idx][prev_t_idx]
                        cur_prob = relevant_prob * self.transition_trigram_p[(prev_prev_tag, prev_tag)][tag]
                        if cur_prob > max_prob:
                            max_prob = cur_prob
                            max_cell = (prev_prev_t_idx, prev_t_idx)
                    viterbi_table[w_idx][prev_t_idx][t_idx] = (max_cell, max_prob * emission_p * 100)

    def _get_emission_prob(self, tag, word):
        return self.emission_p[tag][self._get_word_or_pattern(word)] or self.emission_p[tag][UNKNOWN]

    def _compute_backtrace(self, viterbi_table, tag_set):
        backtrace = [None]
        max_prob = 0
        cur_elem = None
        for t_idx, tag in enumerate(tag_set):
            for prev_t_idx, _ in enumerate(tag_set):
                prev_cell, cur_prob = viterbi_table[-1][prev_t_idx][t_idx]
                if cur_prob > max_prob:
                    max_prob = cur_prob
                    cur_elem = prev_cell
                    backtrace[0] = tag_set[t_idx]
        for col in reversed(viterbi_table[0:-1]):
            if cur_elem == (-1, -1): break
            backtrace.insert(0, tag_set[cur_elem[1]])
            cur_elem, _ = col[cur_elem[0]][cur_elem[1]]
        return backtrace