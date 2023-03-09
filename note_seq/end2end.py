import os
from pathlib import Path
import ujson
import regex as re
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
from evaluate import load
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from model.summarizer import Summarizer
from note_seq.dataset import DATA_FN, MINI_DATA_FN
from model.utils import split_into_notes, get_path_from_exp
from data.utils import (
    extract_sorted_notes_from_html, remove_tags_from_sent, transform_text, split_into_sections, sents_from_html
)


def generate_baseline(args, model, tokenizer, gen_kwargs, source_html, reference, predicted_top_section=None):
    source = transform_text(source_html, include_header=True, include_title=True)

    if not args.empty_init:
        source = prepend_partial(predicted_top_section, source)

    inputs = tokenizer(
        [source],
        padding='do_not_pad',
        truncation=True,
        max_length=16384,
        return_tensors='pt'
    )

    inputs['input_ids'] = inputs['input_ids'].to(model.device)
    inputs['attention_mask'] = inputs['attention_mask'].to(model.device)

    inputs['global_attention_mask'] = torch.FloatTensor(len(inputs['input_ids']) * [
        [0 for _ in range(len(inputs['input_ids'][0]))]
    ]).to(model.device)

    # the 1st element of each sequence in batch should be flipped to 1
    inputs['global_attention_mask'][:, 0] = 1

    inputs.update(**gen_kwargs)
    with torch.no_grad():
        pred_ids = model.model.generate(**inputs)
        generated_str = tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)[0]
        output = {'prediction': generated_str}

        score_obj = rouge.compute(
            references=[reference], predictions=[generated_str], rouge_types=['rouge1', 'rouge2']
        )
        score_obj['mean'] = (score_obj['rouge1'] + score_obj['rouge2']) / 2.0
        output.update(score_obj)
        return output


def generate_notewise(args, model, tokenizer, gen_kwargs, source_html, reference, predicted_top_section=None):
    notes = split_into_notes(source_html)
    n = len(notes)

    if args.empty_init:
        partial_sum = ''
    else:
        if args.oracle_init:
            source_sections, section_rouges = score_sections(rouge, source_html, reference)
            top_sec_idx = int(np.argmax([x['mean'] for x in section_rouges]))
            top_sec = source_sections[top_sec_idx]
            scores_by_time = [section_rouges[top_sec_idx]]
            partial_sum = ' '.join(map(remove_tags_from_sent, top_sec['sents']))
        else:
            sec_sents = sents_from_html(predicted_top_section)
            sec_tok = ' '.join(list(map(remove_tags_from_sent, sec_sents)))
            score_obj = rouge.compute(references=[reference], predictions=[sec_tok], rouge_types=['rouge1', 'rouge2'])
            score_obj['mean'] = (score_obj['rouge1'] + score_obj['rouge2']) / 2.0
            scores_by_time = [score_obj]
            partial_sum = sec_tok

    for start in range(0, n, args.note_window):
        end = min(n, start + args.note_window)
        curr_notes = notes[start:end]
        curr_note_str = transform_text('<SEP>'.join(curr_notes), include_header=True, include_title=True)
        if len(partial_sum) > 0:
            updated_source = prepend_partial(partial_sum, curr_note_str)
        else:
            updated_source = curr_note_str

        inputs = tokenizer(
            [updated_source],
            padding='do_not_pad',
            truncation=True,
            max_length=16384,
            return_tensors='pt'
        )

        inputs['input_ids'] = inputs['input_ids'].to(model.device)
        inputs['attention_mask'] = inputs['attention_mask'].to(model.device)

        inputs['global_attention_mask'] = torch.FloatTensor(len(inputs['input_ids']) * [
            [0 for _ in range(len(inputs['input_ids'][0]))]
        ]).to(model.device)

        # the 1st element of each sequence in batch should be flipped to 1
        inputs['global_attention_mask'][:, 0] = 1

        inputs.update(**gen_kwargs)

        with torch.no_grad():
            pred_ids = model.model.generate(**inputs)
            generated_str = tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)[0]

            score_obj = rouge.compute(
                references=[reference], predictions=[generated_str], rouge_types=['rouge1', 'rouge2']
            )
            score_obj['mean'] = (score_obj['rouge1'] + score_obj['rouge2']) / 2.0

            # Oracle Rejection
            if args.rejection:
                # TODO replace with model
                if score_obj['mean'] > max([x['mean'] for x in scores_by_time]):
                    partial_sum = generated_str
                    # print('Improvement')
                # else:
                #     print('No improvement')
            else:  # Always take the most recent
                partial_sum = generated_str
            scores_by_time.append(score_obj)

    init_r1 = scores_by_time[0]['rouge1']
    init_r2 = scores_by_time[0]['rouge2']
    r1 = scores_by_time[-1]['rouge1']
    r2 = scores_by_time[-1]['rouge2']
    max_r1 = max([x['rouge1'] for x in scores_by_time])
    max_r2 = max([x['rouge2'] for x in scores_by_time])
    print(f'ROUGE 1 (T0, MAX): {round(r1, 3)} ({round(init_r1, 3)}, {round(max_r1, 3)})')

    return {
        'prediction': partial_sum,
        'init_rouge1': init_r1,
        'init_rouge2': init_r2,
        'rouge1': r1,
        'rouge2': r2,
        'max_rouge1': max_r1,
        'max_rouge2': max_r2,
        'rouge_history': ujson.dumps(scores_by_time)
    }


def score_sections(rouge, source_html, reference):
    sections = split_into_sections(source_html)

    scores = []
    for section in sections:
        sec_tok = ' '.join(list(map(remove_tags_from_sent, section['sents'])))
        score_obj = rouge.compute(references=[reference], predictions=[sec_tok], rouge_types=['rouge1', 'rouge2'])
        score_obj['mean'] = (score_obj['rouge1'] + score_obj['rouge2']) / 2.0
        scores.append(score_obj)
    return sections, scores


def prepend_partial(partial, source_str):
    return f'<doc-sep>Title: Discharge Summary\nBRIEF HOSPITAL COURSE:\n{partial}\n{source_str}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Note By Note Generator.')
    parser.add_argument('--data_dir', default=os.path.expanduser('~'))
    parser.add_argument('--experiment', default='led_v2')
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--max_examples', default=None, type=int)
    parser.add_argument('--split', default='test')
    parser.add_argument('--device', default=0, type=int)

    parser.add_argument('--max_input_length', type=int, default=16384)
    parser.add_argument('--max_output_length', type=int, default=512)
    parser.add_argument('--min_length', default=64, type=int)
    parser.add_argument('--note_window', default=1, type=int)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--length_penalty', default=2.0, type=float)
    parser.add_argument('--num_beams', default=4, type=int)
    parser.add_argument('--hf_name', default='allenai/led-large-16384', choices=[
        'allenai/led-large-16384',
        'tau/bart-large-sled',
        'google/long-t5-tglobal-base',
        'google/pegasus-x-large',
        # PageSum
    ])

    parser.add_argument('-oracle_data', default=False, action='store_true')

    parser.add_argument('-rejection', default=False, action='store_true')
    parser.add_argument('-empty_init', default=False, action='store_true')
    parser.add_argument('-oracle_init', default=False, action='store_true')
    parser.add_argument('-baseline', default=False, action='store_true')

    args = parser.parse_args()

    rouge = load('rouge')

    gen_kwargs = {
        'length_penalty': args.length_penalty,
        'min_length': args.min_length,
        'max_length': args.max_length,
        'num_beams': args.num_beams,
        'no_repeat_ngram_size': 3,
        'early_stopping': True,
        'use_cache': True,
    }

    weight_dir = os.path.join(args.data_dir, 'bhc_weights')
    ckpt_path = get_path_from_exp(weight_dir, args.experiment)

    results_dir = os.path.join(weight_dir, args.experiment, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print(f'Loading tokenizer from {args.hf_name}...')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_name)

    print('Adding <doc-sep> as an additional special token...')
    add_tokens = ['<doc-sep>']
    special_tokens_dict = {'additional_special_tokens': add_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    print(f'Loading model from {ckpt_path}...')
    if ckpt_path.endswith('ckpt'):
        model = Summarizer.load_from_checkpoint(
            checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_name=args.hf_name
        ).eval()
    else:
        model = Summarizer(args, tokenizer=tokenizer, hf_name=args.hf_name).eval()
        weights = torch.load(ckpt_path)
        weights = {k.replace('_forward_module.', ''): v for k, v in weights.items()}
        model.load_state_dict(weights, strict=False)

    model = model.to(args.device)
    model.on_predict_start()

    note_meta_fn = os.path.join('/nlp/projects/summarization/kabupra/cumc/note_meta.csv')
    print(f'Loading in note meta information from {note_meta_fn}')
    note_meta_df = pd.read_csv(note_meta_fn)

    if args.oracle_data:
        if args.debug:
            data_fn = os.path.join(args.data_dir, MINI_DATA_FN)
        else:
            data_fn = os.path.join(args.data_dir, DATA_FN)
        print(f'Reading in dataset from {data_fn}')
        data_df = pd.read_csv(data_fn)
        test_df = data_df[data_df['split'] == 'test']
    else:
        data_fn = os.path.join(args.data_dir, f'clinsum_test_10000_tokens.csv')
        print(f'Reading in dataset from {data_fn}')
        test_df = pd.read_csv(data_fn)

        if args.debug:
            oracle_mini = pd.read_csv(os.path.join(args.data_dir, MINI_DATA_FN))
            oracle_mini['example_id'] = (
                    oracle_mini['patient_id'].astype(str) + '_' + oracle_mini['visit_id'].astype(str)
            )
            eid = set(oracle_mini[oracle_mini['split'] == 'test']['example_id'])
            test_df['example_id'] = test_df['patient_id'].astype(str) + '_' + test_df['visit_id'].astype(str)
            test_df = test_df[test_df['example_id'].isin(eid)].reset_index(drop=True)

    if args.max_examples is not None:
        test_df = test_df.sample(n=args.max_examples, random_state=1992).reset_index(drop=True)

    records = test_df.to_dict('records')

    for example in tqdm(records):
        reference = transform_text(example['reference'], include_header=False, include_title=False)

        source_note_ids = re.findall(r'note_id=([^ ]+)', example['source'])
        source_note_meta = note_meta_df[
            note_meta_df['note_id'].isin(set(source_note_ids))
        ].sort_values(by='created_time').to_dict('records')
        source_html = extract_sorted_notes_from_html(example['source'], source_note_meta)
        final_predictions = []

        gen_func = generate_baseline if args.baseline else generate_notewise
        outputs = gen_func(
            args, model, tokenizer, gen_kwargs, source_html, reference,
            predicted_top_section=example.get('predicted_top_section', None)
        )
        example.update(outputs)

    out_df = pd.DataFrame(records)
    out_df['example_id'] = out_df['patient_id'].astype(str) + '_' + out_df['visit_id'].astype(str)

    mini_str = '_mini' if args.debug else ''

    baseline_str = '_baseline' if args.baseline else ''
    empty_str = '_empty' if args.empty_init else ''
    oracle_str = '_oracle' if args.oracle_init else ''
    reject_str = '_rejection' if args.rejection else ''
    out_fn = os.path.join(
        results_dir, f'note_wise{baseline_str}{empty_str}{oracle_str}{reject_str}{mini_str}.csv'
    )

    print(f'Saving {len(out_df)} ROUGE scores and predictions to {out_fn}')
    out_df.to_csv(out_fn, index=False)
    num_col = out_df.select_dtypes('number')
    for col in list(num_col.columns):
        if 'rouge' in col:
            print(f'{col}: {num_col[col].dropna().mean()}')
