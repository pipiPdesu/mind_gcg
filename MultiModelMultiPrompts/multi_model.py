import argparse
import random
import gc
import pandas as pd
import time
import logging
import sys

sys.path.append('../')
from utils.opt_utils import AttackManager
from utils.str_utils import SuffixManager
import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn

parser = argparse.ArgumentParser(
    description=': Generating malicious responses using pre-made triggers/'
                'on-the-spot generated triggers.'
)

parser.add_argument(
    '--attack_model', type=str, default='llama2',
    help='The model to attack.'
)

parser.add_argument(
    '--model_path', type=str,  default='/root/code/llama2',
    help='The path of the model that needs to be loaded.'
)

parser.add_argument(
    '--trigger', type=str,  default='''! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !''',
    help='Transferable triggers or initial triggers.'
)

parser.add_argument(
    '--trigger_type',  default=False, action='store_true',
    help='Transferable triggers or initial triggers, default to initial triggers.'
)

parser.add_argument(
    '--train_epoch', type=int,  default=1000,
    help='The number of epochs to train the trigger.'
)

parser.add_argument(
    '--batch_size', type=int,  default=512,
    help='The batch size to train the trigger.To keep the search enough big it is recommanded not to change.'
)

parser.add_argument(
    '--search_size', type=int,  default=128,
    help='The search size of every batch. Decrease if OOM.'
)

parser.add_argument(
    '--topk', type=int,  default=256,
    help='The number of top-k tokens to be sampled.'
)

parser.add_argument(
    '--allow_non_ascii', default=False, action='store_true',
    help='Allow non-ascii tokens.'
)

parser.add_argument(
    '--n_train_data', default=25,
    help='data to train suffix'
)

parser.add_argument(
    '--n_test_data', default=25,
    help='data to test suffix'
)

parser.add_argument(
    '--train_data', default='/root/code/data/harmful_behaviors.csv',
    help='path to train data'
)

parser.add_argument(
    '--test_step', default=25,
    help='how many step to test'
)

def get_goals_and_targets(params):
    #这里修改到只有一点假设就是train和test在同一目录下
    train_goals = getattr(params, 'goals', [])
    train_targets = getattr(params, 'targets', [])
    test_goals = getattr(params, 'test_goals', [])
    test_targets = getattr(params, 'test_targets', [])
    offset = getattr(params, 'data_offset', 0)

    if params.train_data:
        train_data = pd.read_csv(params.train_data)
        train_targets = train_data['target'].tolist()[offset:offset+params.n_train_data]
        if 'goal' in train_data.columns:
            train_goals = train_data['goal'].tolist()[offset:offset+params.n_train_data]
        else:
            train_goals = [""] * len(train_targets)
        
        if params.n_test_data > 0:
            test_targets = train_data['target'].tolist()[offset+params.n_train_data:offset+params.n_train_data+params.n_test_data]
            if 'goal' in train_data.columns:
                test_goals = train_data['goal'].tolist()[offset+params.n_train_data:offset+params.n_train_data+params.n_test_data]
            else:
                test_goals = [""] * len(test_targets)

    assert len(train_goals) == len(train_targets)
    assert len(test_goals) == len(test_targets)
    logging.info('Loaded {} train goals'.format(len(train_goals)))
    logging.info('Loaded {} test goals'.format(len(test_goals)))

    return train_goals, train_targets, test_goals, test_targets


