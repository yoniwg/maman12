## general imports
import operator
from collections import defaultdict

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

## project supplied imports
from submission_specs.SubmissionSpec12 import SubmissionSpec12

START_LABEL = '<S>'
END_LABEL = '<E>'
DELTA = 0.001

class Submission(SubmissionSpec12):
    def __init__(self):
        self.transition_p = defaultdict(lambda: defaultdict(float))
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        self.hasher = DictVectorizer()
        self.tags = []
        self.proba_table = list([])

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
            count += len(next_tags_dict) * DELTA
            for next_tag in next_tags_dict:
                self.transition_p[tag][next_tag] = (next_tags_dict[next_tag] + DELTA) / count

    def _vectorize(self, sentence, w_idx):
        cur_word = sentence[w_idx]
        if w_idx > 0:
            prev_word = sentence[w_idx - 1]
        else:
            prev_word = START_LABEL
        if w_idx < len(sentence) - 1:
            next_word = sentence[w_idx + 1]
        else:
            next_word = END_LABEL
        return {
            'word': cur_word,
            'index': w_idx,
            'is_first': w_idx == 0,
            'is_last': w_idx == len(sentence) - 1,
            'prev_word': prev_word,
            'next_word': next_word,
            'curr_is_lower': cur_word.islower(),
            'prev_is_lower': prev_word.islower(),
            'next_is_lower': next_word.islower(),
            'curr_is_upper': cur_word.isupper(),
            'prev_is_upper': prev_word.isupper(),
            'next_is_upper': next_word.isupper(),
            'first_upper': cur_word[0].isupper(),
            'prev_first_upper': prev_word[0].isupper(),
            'next_first_upper': next_word[0].isupper(),
            'is_digit': cur_word.isdigit(),
            'prev_is_digit': prev_word.isdigit(),
            'next_is_digit': next_word.isdigit(),
            'has_no_sign': cur_word.isalnum(),
            'prev_is_sign': all(map(lambda c: not c.isalnum(), prev_word)),
            'next_is_sign': all(map(lambda c: not c.isalnum(), next_word)),
            'prefix-1': cur_word[0],
            'prefix-2': cur_word[:2],
            'prefix-3': cur_word[:3],
            'suffix-1': cur_word[-1],
            'suffix-2': cur_word[-2:],
            'suffix-3': cur_word[-3:],
            'prev_prefix-1': prev_word[0],
            'prev_prefix-2': prev_word[:2],
            'prev_prefix-3': prev_word[:3],
            'prev_suffix-1': prev_word[-1],
            'prev_suffix-2': prev_word[-2:],
            'prev_suffix-3': prev_word[-3:],
            'next_prefix-1': next_word[0],
            'next_prefix-2': next_word[:2],
            'next_prefix-3': next_word[:3],
            'next_suffix-1': next_word[-1],
            'next_suffix-2': next_word[-2:],
            'next_suffix-3': next_word[-3:],
        }

    def _estimate_emission_probabilites(self, annotated_sentences):
        ''' trains the MEMM model (computes the probability distributions) '''
        words_only = [[entry[0] for entry in sentence] for sentence in annotated_sentences]
        X = self.hasher.fit_transform(
            [self._vectorize(sentence, idx)
             for sentence in words_only
             for idx in range(len(sentence))]
        )

        y = [entry[1] for sentence in annotated_sentences for entry in sentence]
        self.model.fit(X, y)

    def train(self, annotated_sentences):
        print('training function received {} annotated sentences as training data'.format(len(annotated_sentences)))
        self._estimate_emission_probabilites(annotated_sentences)
        self._estimate_transition_probabilites(annotated_sentences)
        return self

    def predict(self, sentence):
        self.tags = self.model.classes_
        self.proba_table = self.model.predict_proba(self.hasher.transform(
            [self._vectorize(sentence, idx) for idx in range(len(sentence))]
        ))
        viterbi_table = self._init_table(sentence)
        self._fill_table(viterbi_table, sentence)
        prediction = self._compute_backtrace(viterbi_table)
        assert (len(prediction) == len(sentence)), "prediction length is too short"
        return prediction

    def _init_table(self, sentence):
        initial_table = [[(-1, 0.0) for _ in self.tags] for _ in sentence]
        for idx, tag in enumerate(self.tags):
            emission_p = self.proba_table[0][idx]
            transition_p = self.transition_p[START_LABEL][tag]
            initial_table[0][idx] = (-1, emission_p * transition_p)
        return initial_table

    def _fill_table(self, viterbi_table, sentence):
        for w_idx, word in enumerate(sentence):
            if w_idx == 0: continue
            for t_idx, tag in enumerate(self.tags):
                emission_p = self.proba_table[w_idx][t_idx]
                max_prob = (-1, 0.0)
                for prev_t_idx, prob in enumerate(viterbi_table[w_idx - 1]):
                    transition_p = self.transition_p[self.tags[prev_t_idx]][tag]
                    cur_prob = prob[1] * transition_p
                    if cur_prob > max_prob[1]:
                        max_prob = (prev_t_idx, cur_prob)
                # We multiply the real value by 100 in order to avoid floating point overflow
                # This does not change the final result since all the column grows accordingly
                viterbi_table[w_idx][t_idx] = (max_prob[0], emission_p * max_prob[1] * 100)

    def _compute_backtrace(self, viterbi_table):
        backtrace = []
        cur_elem = max(viterbi_table[-1], key=operator.itemgetter(1))
        backtrace.insert(0, self.tags[viterbi_table[-1].index(cur_elem)])
        for col in reversed(viterbi_table[0:-1]):
            if cur_elem[0] == -1 or cur_elem[1] == 0.0: break
            backtrace.insert(0, self.tags[cur_elem[0]])
            cur_elem = col[cur_elem[0]]
        return backtrace
