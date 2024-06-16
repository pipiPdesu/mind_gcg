import argparse
import random
import gc
from tqdm import tqdm
import pandas as pd

from opt_utils import AttackManager
from str_utils import SuffixManager
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
    '--trigger', type=str,  default="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    help='Transferable triggers or initial triggers.'
)

parser.add_argument(
    '--trigger_type',  default=False, action='store_true',
    help='Transferable triggers or initial triggers, default to initial triggers.'
)

parser.add_argument(
    '--train_epoch', type=int,  default=100,
    help='The number of epochs to train the trigger.'
)

parser.add_argument(
    '--batch_size', type=int,  default=40,
    help='The batch size to train the trigger.Decrease if OOM.'
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
    '--n_train_data', default=2,
    help='data to train suffix'
)

parser.add_argument(
    '--n_test_data', default=2,
    help='data to test suffix'
)

parser.add_argument(
    '--train_data', default='/root/code/data/harmful_behaviors.csv',
    help='path to train data'
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
    print('Loaded {} train goals'.format(len(train_goals)))
    print('Loaded {} test goals'.format(len(test_goals)))

    return train_goals, train_targets, test_goals, test_targets




if __name__ == "__main__":
    args = parser.parse_args()
    
    train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(args)

    #随机修改一下开头的这个，我也不知道为什么
    process_fn = lambda s: s.replace('Sure, h', 'H')
    process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
    train_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in train_targets]
    test_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in test_targets]
    
    attack = AttackManager(args.model_path, args.batch_size, args.topk)

    train_manager, test_manager = [], []
    for i in range(len(train_goals)):
        train_manager.append(SuffixManager(attack.tokenizer, train_goals[i], train_targets[i], args.trigger))
    for i in range(len(test_goals)):
        test_manager.append(SuffixManager(attack.tokenizer, test_goals[i], test_targets[i], args.trigger))
    

    loss = np.infty
    num_goals = 1
    control = args.trigger
    best_control = control
    best_loss = np.infty

    not_allowed_tokens = None if args.allow_non_ascii else attack.get_nonascii_toks()
    for i in range(args.train_epoch):
            
        coordinate_grad = sum(attack.token_gradients(train_manager[j].get_input_ids(control),
                                            train_manager[j].control_slice,
                                            train_manager[j].target_slice,
                                            train_manager[j].loss_slice)
                                            for j in range(len(train_goals)))

        adv_suffix_tokens = train_manager[-1].get_input_ids(control)[train_manager[-1].control_slice]
        new_adv_suffix_toks = attack.sample_control(adv_suffix_tokens,
                                                        coordinate_grad,
                                                        not_allowed_tokens)
        
        new_adv_suffix = attack.get_filtered_cands(new_adv_suffix_toks,
                                                       filter_cand=True,
                                                       curr_control=control)
        
        
    

