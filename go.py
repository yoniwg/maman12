from sys import argv

import solution12_3
from verification.exercise_data_building.pos import build_POS_tagging_data
from verification.model_driver_12 import model_driver_12
import solution12


def goall():
    go12()

def go12():
    model_driver_12(
        solution12_3.Submission,
        build_POS_tagging_data(
            source_treebank_name = "UD_English-EWT",  
            git_hash = "7be629932192bf1ceb35081fb29b8ecb0bd6d767"),
        passes = 20)

def print_usage():
    print()
    print("Usage:")
    print('python go.py')
    print()

if len(argv) > 1:
    print_usage()
else:
    go12()
        