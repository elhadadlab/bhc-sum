import os
import regex as re

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from utils import Seq2SeqCollate
from data.utils import extract_sorted_notes_from_html, transform_text


DATA_FN = 'bhc_data_10000_filt.csv'
MINI_DATA_FN = 'bhc_mini_data_10000_filt.csv'


class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, args, note_meta_df, tokenizer, max_val_num=None):
        super().__init__()

        if args.debug:
            data_fn = os.path.join(args.data_dir, MINI_DATA_FN)
        else:
            data_fn = os.path.join(args.data_dir, DATA_FN)
        print(f'Reading in dataset from {data_fn}')
        self.data_df = pd.read_csv(data_fn)
        self.max_val_num = max_val_num
        self.note_meta_df = note_meta_df
        self.tokenizer = tokenizer
        self.data_dir = args.data_dir
        self.debug = args.debug
        self.tokenizer = tokenizer
        self.hf_name = args.hf_name
        self.num_workers = 0 if self.debug else 8
        self.max_input_length = tokenizer.model_max_length if args.max_input_length is None else args.max_input_length
        if self.max_input_length > tokenizer.model_max_length:
            print(f'Warning! Setting maximum input length to be maximum model length of {tokenizer.model_max_length}')
            self.max_input_length = tokenizer.model_max_length
        self.max_output_length = args.max_output_length
        self.batch_size = 1

    def train_dataloader(self):
        train_df = self.data_df[self.data_df['split'] == 'train']
        records = train_df.to_dict('records')

        train_split = SummarizationDataset(self.note_meta_df, records, 'train', self.max_input_length)

        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            add_global_att='allenai/led' in self.hf_name,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
        )
        kwargs = {
            'batch_size': self.batch_size,
            'shuffle': True,
            'num_workers': 1 if self.debug else self.num_workers,
            'collate_fn': collate_fn
        }
        return DataLoader(train_split, **kwargs)

    def predict_dataloader(self):
        records = self.data_df[self.data_df['split'] == 'test'].to_dict('records')
        test_split = SummarizationDataset(self.note_meta_df, records, 'test', self.max_input_length)
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            add_global_att='allenai/led' in self.hf_name,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            add_cols=['example_id']
        )
        kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': 1 if self.debug else self.num_workers,
            'collate_fn': collate_fn
        }
        return DataLoader(test_split, **kwargs)

    def test_dataloader(self, add_cols=None, max_examples=None, split='test'):
        test_df = self.data_df[self.data_df['split'] == split]
        if max_examples is not None and max_examples < len(test_df):
            print(f'Subsampling {max_examples} examples from {len(test_df)}')
            test_df = test_df.sample(n=max_examples, replace=False, random_state=1992)
        records = test_df.to_dict('records')
        test_split = SummarizationDataset(self.note_meta_df, records, split, self.max_input_length, add_cols=add_cols)
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            add_global_att='allenai/led' in self.hf_name,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            add_cols=add_cols
        )
        kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': 1 if self.debug else self.num_workers,
            'collate_fn': collate_fn
        }
        return DataLoader(test_split, **kwargs)

    def val_dataloader(self, max_n=None, add_cols=None):
        val_df = self.data_df[self.data_df['split'] == 'validation']
        max_n = min(filter(None, [max_n, self.max_val_num]))
        if max_n is not None and max_n < len(val_df):
            print(f'Sampling {max_n} examples out of {len(val_df)}')
            val_df = val_df.sample(n=max_n, replace=False, random_state=1992)
        records = val_df.to_dict('records')
        val_split = SummarizationDataset(self.note_meta_df, records, 'validation', self.max_input_length)
        collate_fn = Seq2SeqCollate(
            self.tokenizer,
            add_global_att='allenai/led' in self.hf_name,
            max_input_length=self.max_input_length,
            max_output_length=self.max_output_length,
            add_cols=add_cols
        )
        kwargs = {
            'batch_size': self.batch_size,
            'shuffle': False,
            'num_workers': 1 if self.debug else self.num_workers,
            'collate_fn': collate_fn
        }
        return DataLoader(val_split, **kwargs)


class SummarizationDataset(Dataset):
    def __init__(self, note_meta_df, examples, split, max_input_length, add_cols=None):
        super(SummarizationDataset, self).__init__()
        self.examples = examples
        self.split = split
        self.max_input_length = max_input_length
        self.add_cols = [] if add_cols is None else add_cols
        self.note_meta_df = note_meta_df

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        source_note_ids = re.findall(r'note_id=([^ ]+)', example['source'])
        source_note_meta = self.note_meta_df[
            self.note_meta_df['note_id'].isin(set(source_note_ids))].sort_values(by='created_time').to_dict('records')
        source_html = extract_sorted_notes_from_html(example['source'], source_note_meta)

        source_str = transform_text(source_html, include_header=True, include_title=True)
        target_str = transform_text(example['reference'], include_header=False, include_title=False)
        row = {
            'source': source_str,
            'target': target_str,
        }

        for col in self.add_cols:
            row[col] = example[col]
        return row
