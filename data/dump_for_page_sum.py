import os
import regex as re

import pandas as pd
from p_tqdm import p_uimap
import ujson

from data.utils import extract_sorted_notes_from_html, remove_tags_from_sent, sents_from_html


def dump(example, split_dir, note_meta_df):
    source_note_ids = re.findall(r'note_id=([^ ]+)', example['source'])
    source_note_meta = note_meta_df[
        note_meta_df['note_id'].isin(set(source_note_ids))].sort_values(by='created_time').to_dict('records')
    source_html = extract_sorted_notes_from_html(example['source'], source_note_meta)

    source_str = list(map(remove_tags_from_sent, sents_from_html(source_html)))
    target_str = list(map(remove_tags_from_sent, sents_from_html(example['reference'])))
    row = {
        'article': source_str,
        'abstract': target_str,
    }

    example_idx = example['idx']
    out_fn = os.path.join(split_dir, f'{example_idx}.json')
    if os.path.exists(out_fn):
        print('Already done. Skipping.')
        return 0
    with open(out_fn, 'w') as fd:
        ujson.dump(row, fd)
    return 1


if __name__ == '__main__':
    fn = os.path.expanduser(os.path.join('~', 'bhc_data_10000_filt.csv'))
    df = pd.read_csv(fn)
    note_meta_fn = os.path.join('/nlp/projects/summarization/kabupra/cumc/note_meta.csv')
    note_meta_df = pd.read_csv(note_meta_fn)

    for split in ['train', 'validation', 'test']:
        records = df[df['split'] == split].to_dict('records')
        for i in range(len(records)):
            records[i]['idx'] = i
        bhc_split = split
        if bhc_split == 'validation':
            bhc_split = 'val'
        split_dir = os.path.expanduser(os.path.join('~', 'bhc', 'base', bhc_split))
        statuses = list(p_uimap(lambda example: dump(example, split_dir, note_meta_df), records))
        print(sum(statuses))
    print('Fini!')
