import os
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import pandas as pd
import spacy
from nltk import word_tokenize
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from baselines.abstractive.gen_transformers.constants import *
from baselines.abstractive.gen_transformers.dataset import SummaryDataModule
from baselines.abstractive.gen_transformers.model import Summarizer
from global_utils import get_free_gpus
from preprocess.nyp.nyp_constants import OUT_DIR as NYP_DIR
from preprocess.shared.tagger import tag_text
from eval.entity import dump_results
from preprocess.shared.sec_tag.main import section_tagger_init
from preprocess.mimic.mimic_constants import OUT_DIR as MIMIC_DIR
from preprocess.shared.fragment_utils import frags


def get_path_from_exp(weights_dir, experiment, last=False):
    dir = os.path.join(weights_dir, experiment)
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


def compute_frags(record, source_col='source', target_col='prediction'):
    source_tokens = word_tokenize(record[source_col])
    target_tokens = word_tokenize(record[target_col])

    # Check coverage
    source_tokens_lower = [x.lower() for x in source_tokens]
    target_tokens_lower = [x.lower() for x in target_tokens]
    frag_obj = frags(source_tokens_lower, target_tokens_lower)
    coverage = frag_obj['coverage']
    density = frag_obj['density']
    compression = frag_obj['compression']

    stats = {
        'coverage': coverage,
        'density': density,
        'compression': compression
    }

    record.update(stats)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LongFormer/BigBird Generator & Evaluator.')
    parser.add_argument('--dataset', default='nyp', choices=['nyp', 'mimic'])
    parser.add_argument('--wandb_name', default=None, required=True)
    parser.add_argument('--experiment', default=None)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('-human_only', default=False, action='store_true')
    parser.add_argument('-oracle_extraction', default=False, action='store_true')
    parser.add_argument('--max_examples', default=None, type=int)
    parser.add_argument('--split', default='test')

    parser = Summarizer.add_model_specific_args(parser)

    args = parser.parse_args()

    mini_str = '_mini' if args.debug else ''
    if args.experiment is None:
        args.experiment = args.wandb_name

    data_dir = MIMIC_DIR if args.dataset == 'mimic' else NYP_DIR


    weight_dir = os.path.join(data_dir, 'clinsum', 'weights')
    ckpt_path = get_path_from_exp(weight_dir, args.wandb_name)

    note_meta_fn = os.path.join(data_dir, 'note_meta.csv')
    note_meta_df = pd.read_csv(note_meta_fn)

    human_split = None
    if args.human_only:
        data_dir = os.path.join(data_dir, 'human')
        human_split = args.split
    results_dir = os.path.join(data_dir, 'gen_transformers', 'results', args.experiment)
    os.makedirs(results_dir, exist_ok=True)

    free_gpus = get_free_gpus()
    gpu = free_gpus[0] if args.gpu_device is None else args.gpu_device
    if args.gpu_device is not None and args.gpu_device not in free_gpus:
        print(f'Warning! Youve selected a GPU that is not available.  Putting the model on {free_gpus[0]} instead.')
        gpu = free_gpus[0]

    print(f'Loading tokenizer from {args.hf_model}...')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_model)

    print(f'Loading model from {ckpt_path}...')
    model = Summarizer.load_from_checkpoint(
        checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_model=args.hf_model, strict=False).to(gpu).eval()

    setattr(args, 'data_dir', data_dir)
    print(f'Loading in note meta information from {note_meta_fn}')
    note_meta_df = pd.read_csv(note_meta_fn)
    datamodule = SummaryDataModule(
        args, note_meta_df, tokenizer, oracle_extraction=args.oracle_extraction, human_split=human_split
    )
    model.on_predict_start()
    dataloader = datamodule.test_dataloader(
        add_cols=['patient_id', 'visit_id'], split=args.split, max_examples=args.max_examples
    )
    outputs = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(gpu) if type(v) == torch.Tensor else v for k, v in batch.items()}
        with torch.no_grad():
            batch_stats = model.predict_step(batch)
        if type(batch_stats) == list:
            outputs += batch_stats
        else:
            outputs.append(batch_stats)

    outputs = pd.DataFrame(outputs)
    outputs['example_id'] = outputs['patient_id'].astype(str) + '_' + outputs['visit_id'].astype(str)

    print('Computing extractive statistics for predictions')
    outputs = pd.DataFrame(list(map(compute_frags, outputs.to_dict('records'))))

    section_tagger_init()
    sentencizer = spacy.load('en_core_sci_lg', disable=['ner', 'parser', 'lemmatizer'])
    sentencizer.add_pipe('sentencizer')
    outputs = outputs.assign(
        prediction_tagged=outputs['prediction'].apply(lambda x: '<SEP>'.join(tag_text(
            x, sentencizer, remove_empty_sub_sections=False, prepend_sub_sections=True
        )[0])),
        pred_num_toks=outputs['prediction'].apply(lambda x: len(x.split(' '))),
        target_num_toks=outputs['target'].apply(lambda x: len(x.split(' '))),
    )
    max_n_suffix = '' if args.max_examples is None else '_' + str(args.max_examples)
    if args.human_only and args.split != 'test':
        out_fn = os.path.join(results_dir, f'outputs_{args.split}.csv')
    else:
        out_fn = os.path.join(results_dir, f'outputs{mini_str}{max_n_suffix}.csv')
    print(f'Saving {len(outputs)} ROUGE scores and predictions to {out_fn}')
    outputs.to_csv(out_fn, index=False)
    num_col = outputs.select_dtypes('number')
    for col in list(num_col.columns):
        print(f'{col}: {num_col[col].dropna().mean()}')

    if args.max_examples is None:
        predictions = outputs.to_dict('records')
        dump_results(results_dir, predictions, prepend_sub_sections=False)
