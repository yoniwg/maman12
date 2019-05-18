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
                self.transition_trigram_p[tags][next_tag] = next_tags_dict[next_tag] / count

    def train(self, annotated_sentences):
        ''' trains the HMM model (computes the probability distributions) '''

        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))
        print("TTT training")
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)

        return self

    def predict(self, sentence, top_n=1):
        tag_set = 'ADJ ADP PUNCT ADV AUX SYM INTJ CCONJ X NOUN DET PROPN NUM VERB PART PRON SCONJ'.split()
        viterbi_table = self._init_table(tag_set, sentence, top_n)
        self._fill_table(viterbi_table, tag_set, sentence, top_n)
        top_n_prediction = self._compute_backtrace(viterbi_table, tag_set, top_n)
        ttt = len(sentence)
        for prediction in top_n_prediction:
            assert (len(prediction) == len(sentence)), "prediction length is too short"
        return top_n_prediction

    def _init_table(self, tag_set, sentence, top_n, start_label=START_LABEL):
        initial_table = [[[(-1, -1, 0.0)] * top_n for _ in tag_set] for _ in sentence]
        first_word = self._lower_first(sentence[0])
        for idx, tag in enumerate(tag_set):
            emission_p = self._get_emission_prob(tag, first_word)
            transition_p = self.transition_trigram_p[(start_label,start_label)][tag]
            initial_table[0][idx] = [((-2, -2), (-1, -1), emission_p * transition_p)]

        if len(sentence) > 1:
            second_word = self._lower_first(sentence[1])
            for t_idx, tag in enumerate(tag_set):
                emission_p = self._get_emission_prob(tag, second_word)
                all_probs = {}
                for prev_t_idx, cell in enumerate(initial_table[0]):
                    _, _, prev_prob = cell[0]
                    transition_p = self.transition_trigram_p[(start_label,tag_set[prev_t_idx])][tag]
                    all_probs[transition_p * prev_prob] = (prev_t_idx, 0)

                sorted_probs = sorted(all_probs.items(), reverse=True)
                best_prob_item = sorted_probs[0]
                best_prob, father = best_prob_item
                prev_row, prew_colom = father
                father_cell = initial_table[prew_colom][prev_row]
                _, _, prev_and_trans_prob = father_cell[0]
                # We multiply the real value by 100 in order to avoid floating point overflow
                # This does not change the final result since all the column grows by constant
                initial_table[1][t_idx] = [((-1, -1), (prev_row, prew_colom), emission_p * prev_and_trans_prob * 100)]

        return initial_table

    def _fill_table(self, viterbi_table, tag_set, sentence, top_n):
        for w_idx, word in enumerate(sentence):
            if w_idx == 0 or w_idx == 1:
                prev_word = word
                continue
            for t_idx, tag in enumerate(tag_set):
                emission_p = self._get_emission_prob(tag, word)
                all_probs = {}
                for prev_prev_t_idx, prev_prev_cell in enumerate(viterbi_table[w_idx - 2]):
                    _, _, prev_prev_prob = prev_prev_cell[0]
                    for prev_t_idx, prev_cell in enumerate(viterbi_table[w_idx - 1]):
                        _, _, prev_prob = prev_cell[0]
                        transition_p = self.transition_trigram_p[(tag_set[prev_prev_t_idx],tag_set[prev_t_idx])][tag]
                        prev_emission_p = self._get_emission_prob(tag_set[prev_t_idx], prev_word)
                        all_probs[transition_p * prev_prev_prob * prev_emission_p] = ((prev_prev_t_idx, w_idx - 2), (prev_t_idx, w_idx - 1))
                    # We multiply the real value by 100 in order to avoid floating point overflow
                    # This does not change the final result since all the column grows by constant

                sorted_probs = sorted(all_probs.items(), reverse=True)
                best_prob_item = sorted_probs[0]
                best_prob, fathers = best_prob_item
                grandfather, fathers = fathers
                # We multiply the real value by 100 in order to avoid floating point overflow
                # This does not change the final result since all the column grows by constant
                viterbi_table[w_idx][t_idx] = [(grandfather, fathers, emission_p * best_prob * 100)]

            prev_word = word

    def _get_emission_prob(self, tag, word):
        return self.emission_p[tag][self._get_word_or_pattern(word)] or self.emission_p[tag][UNKNOWN]

    def _compute_backtrace(self, viterbi_table, tag_set, top_n):

        sentense_len = len(viterbi_table)
        last = viterbi_table[sentense_len-1]
        int_cell = viterbi_table[sentense_len-1][0]
        prev_prev_cell, prev_cell, max_prob = int_cell[0]
        index = 0
        for prev_t_idx, cell in enumerate(last):
            tested_prev_prev_cell, tested_prev_cell, tested_max_prob = cell[0]
            if tested_max_prob > max_prob:
                index = prev_t_idx
                max_prob = tested_max_prob
                prev_prev_cell = tested_prev_prev_cell
                prev_cell = tested_prev_cell

        backtrace = []
        backtrace.append(tag_set[index])

        prev_prev_row, prev_prev_colom = prev_prev_cell
        prev_row, prev_colom = prev_cell

        while prev_prev_row > -1 and prev_prev_colom > -1:
            backtrace.append(tag_set[prev_row])
            backtrace.append(tag_set[prev_prev_row])
            prev_cell = viterbi_table[prev_row][prev_colom]

            tested_cell = viterbi_table[prev_prev_row][prev_prev_colom]
            prev_prev_cell, prev_cell, max_prob = tested_cell[0]
            prev_prev_row, prev_prev_colom = prev_prev_cell
            prev_row, prev_colom = prev_cell

        if prev_row > -1 and prev_colom > -1:
            backtrace.append(tag_set[prev_row])

        reversed_backtrace = reversed(backtrace)

        return reversed_backtrace
