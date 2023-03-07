import os
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

from model.dataset import SummaryDataModule
from model.summarizer import Summarizer


def get_path_from_exp(weights_dir, experiment, last=False):
    dir = os.path.join(weights_dir, experiment)
    paths = list(map(str, list(Path(dir).rglob('pytorch_model.bin'))))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LongFormer/BigBird Generator & Evaluator.')
    parser.add_argument('--data_dir', default=os.path.expanduser('~'))
    parser.add_argument('--experiment', default=None)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--gpu_device', default=None, type=int)
    parser.add_argument('--max_examples', default=None, type=int)
    parser.add_argument('--split', default='test')
    parser.add_argument('--device', default=0, type=int)

    parser.add_argument('--max_input_length', type=int, default=16384)
    parser.add_argument('--max_output_length', type=int, default=512)
    parser.add_argument('--min_length', default=64, type=int)
    parser.add_argument('--max_length', default=512, type=int)
    parser.add_argument('--length_penalty', default=2.0, type=float)
    parser.add_argument('--num_beams', default=4.0, type=float)
    parser.add_argument('--hf_name', default='allenai/led-large-16384', choices=[
        'allenai/led-large-16384',
        'tau/bart-large-sled',
        'google/long-t5-tglobal-base',
        'google/pegasus-x-large',
        # PageSum
    ])

    args = parser.parse_args()

    gen_kwargs = {
        'length_penalty': args.length_penalty,
        'min_length': args.min_length,
        'max_length': args.max_length,
        'num_beams': args.num_beams,
    }

    mini_str = '_mini' if args.debug else ''
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
    # model = Summarizer.load_from_checkpoint(
    #     checkpoint_path=ckpt_path, tokenizer=tokenizer, hf_name=args.hf_name, strict=True).to(args.device).eval()
    model = Summarizer(args, tokenizer=tokenizer, hf_name=args.hf_name).eval()

    weights = torch.load(ckpt_path)
    weights = {k.replace('_forward_module.', ''): v for k, v in weights.items()}

    model = model.to(args.device)
    model.load_state_dict(weights, strict=False)

    note_meta_fn = os.path.join('/nlp/projects/summarization/kabupra/cumc/note_meta.csv')
    print(f'Loading in note meta information from {note_meta_fn}')
    note_meta_df = pd.read_csv(note_meta_fn)
    datamodule = SummaryDataModule(args, note_meta_df, tokenizer=tokenizer)

    model.on_predict_start()
    dataloader = datamodule.test_dataloader(
        add_cols=['patient_id', 'visit_id'], split=args.split, max_examples=args.max_examples
    )
    outputs = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch = {k: v.to(model.device) if type(v) == torch.Tensor else v for k, v in batch.items()}
        with torch.no_grad():
            batch_stats = model.predict_step(batch, **gen_kwargs)
        if type(batch_stats) == list:
            outputs += batch_stats
        else:
            outputs.append(batch_stats)

    outputs = pd.DataFrame(outputs)
    outputs['example_id'] = outputs['patient_id'].astype(str) + '_' + outputs['visit_id'].astype(str)

    max_n_suffix = '' if args.max_examples is None else '_' + str(args.max_examples)
    if args.human_only and args.split != 'test':
        out_fn = os.path.join(results_dir, f'predictions_{args.split}.csv')
    else:
        out_fn = os.path.join(results_dir, f'predictions{mini_str}{max_n_suffix}.csv')
    print(f'Saving {len(outputs)} ROUGE scores and predictions to {out_fn}')
    outputs.to_csv(out_fn, index=False)
    num_col = outputs.select_dtypes('number')
    for col in list(num_col.columns):
        print(f'{col}: {num_col[col].dropna().mean()}')
