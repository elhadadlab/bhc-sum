from glob import glob
import os
from pathlib import Path
import random
import subprocess

import numpy as np
import torch


def get_free_gpus():
    try:
        gpu_stats = subprocess.check_output(
            ['nvidia-smi', '--format=csv,noheader', '--query-gpu=memory.used'], encoding='UTF-8')
        used = list(filter(lambda x: len(x) > 0, gpu_stats.split('\n')))
        return [idx for idx, x in enumerate(used) if int(x.strip().rstrip(' [MiB]')) <= 500]
    except:
        return []


def set_same_seed(seed):
    # Set same random seed for each run
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f'Set random, numpy and torch seeds to {seed}')


def split_into_notes(html_str):
    tps = html_str.split('<SEP>')
    notes = []
    curr_note = []
    for tp in tps:
        curr_note.append(tp)
        if tp == '</d>':
            notes.append('<SEP>'.join(curr_note))
            curr_note = []
    return notes


def get_path_from_exp(weights_dir, experiment, last=False):
    dir = os.path.join(weights_dir, experiment)
    paths = list(map(str, list(Path(dir).rglob('pytorch_model.bin'))))
    if len(paths) == 0:
        paths = list(map(str, list(Path(dir).rglob('*.ckpt'))))
    if last:
        return [p for p in paths if 'last' in p][0]
    paths = [p for p in paths if 'last' not in p]
    if len(paths) == 0:
        raise Exception(f'No weights found in {dir}')
    elif len(paths) == 1:
        return str(paths[0])
    else:
        print('\n'.join([str(x) for x in paths]))
        raise Exception('Multiple possible weights found.  Please remove one or specify the path with --restore_path')


class Seq2SeqCollate:
    def __init__(self, tokenizer, add_global_att, max_input_length=16348, max_output_length=512, add_cols=None):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        assert self.max_input_length <= tokenizer.model_max_length
        self.max_output_length = max_output_length
        self.pad_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.add_global_att = add_global_att
        self.add_cols = [] if add_cols is None else add_cols

    def __call__(self, batch_list):
        # tokenize the inputs and labels
        inputs = self.tokenizer(
            [x['source'] for x in batch_list],
            padding='longest',
            truncation=True,
            max_length=self.max_input_length,
            return_tensors='pt'
        )

        outputs = self.tokenizer(
            [x['target'] for x in batch_list],
            padding='longest',
            truncation=True,
            max_length=self.max_output_length,
            return_tensors='pt'
        )

        batch = {}
        batch['input_ids'] = inputs.input_ids
        batch['attention_mask'] = inputs.attention_mask
        if self.add_global_att:
            # create 0 global_attention_mask lists
            batch['global_attention_mask'] = torch.FloatTensor(len(batch['input_ids']) * [
                [0 for _ in range(len(batch['input_ids'][0]))]
            ])

            # the 1st element of each sequence in batch should be flipped to 1
            batch['global_attention_mask'][:, 0] = 1

        batch['labels'] = outputs.input_ids
        # We have to make sure that the PAD token is ignored
        batch['labels'][torch.where(batch['labels'] == 1)] = -100
        for col in self.add_cols + ['curr_note_idx']:
            batch[col] = [x[col] for x in batch_list]
        return batch
