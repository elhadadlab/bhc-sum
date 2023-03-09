import os
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import argparse
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers
import torch
from transformers import AutoTokenizer
from sled import SledTokenizer

from dataset import SummaryDataModule
from summarizer import Summarizer
from model.utils import get_path_from_exp, set_same_seed

torch.set_float32_matmul_precision('medium')  # | 'high')


def run(args):
    if 'sled' in args.hf_name:
        tokenizer = SledTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.hf_name)

    print('Adding <doc-sep> as an additional special token...')
    add_tokens = ['<doc-sep>']
    special_tokens_dict = {'additional_special_tokens': add_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)

    model = Summarizer(args, tokenizer=tokenizer, hf_name=args.hf_name)

    weight_dir = os.path.join(args.data_dir, 'bhc_weights')
    ckpt_path = get_path_from_exp(weight_dir, args.partial_experiment)
    print(f'Loading model from {ckpt_path}...')
    weights = torch.load(ckpt_path)
    weights = {k.replace('_forward_module.', ''): v for k, v in weights.items()}
    model.load_state_dict(weights, strict=False)

    note_meta_fn = os.path.join('/nlp/projects/summarization/kabupra/cumc/note_meta.csv')
    note_meta_df = pd.read_csv(note_meta_fn)
    datamodule = SummaryDataModule(args, note_meta_df, tokenizer=tokenizer, max_val_num=args.max_val_num)

    experiment_dir = os.path.join(args.weight_dir, args.experiment)
    os.makedirs(os.path.join(experiment_dir, 'wandb'), exist_ok=True)  # Only way to make sure it's writable
    logger = pl_loggers.WandbLogger(
        name=args.experiment,
        save_dir=experiment_dir,
        offline=args.debug or args.offline,
        project='bhc_sum',
        entity='clinsum',
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val/rouge_mean',
        save_top_k=1,
        save_last=False,
        mode='max'
    )
    callbacks = [checkpoint_callback]
    if not args.debug:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=logger,
        strategy=None,
        precision='bf16' if 'long-t5' in args.hf_name else 32 if args.cpu else 16,
        accelerator='cpu' if args.cpu else 'gpu',
        devices=1,
        default_root_dir=experiment_dir,
        gradient_clip_val=0.1,
        accumulate_grad_batches=args.grad_accum,
        val_check_interval=1.0 if args.debug else 1000,
        num_sanity_val_steps=2,
        log_every_n_steps=5,
        max_steps=args.max_steps,
    )

    if args.ckpt_path is None:
        print('Starting training...')
    else:
        print(f'Starting training from {args.ckpt_path}...')
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)
    print(f'Best weights saved --> {checkpoint_callback.best_model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('BHC Summarization trainer.')
    parser.add_argument('--data_dir', default=os.path.expanduser('~'))
    parser.add_argument('--experiment', default='default')
    parser.add_argument('--partial_experiment', default='led_v2')
    parser.add_argument('--ckpt_path', default=None)
    parser.add_argument('--seed', default=1992, type=int)
    parser.add_argument('--max_steps', default=10000, type=int)
    parser.add_argument('--warmup', default=2000, type=int)
    parser.add_argument('-debug', default=False, action='store_true')
    parser.add_argument('-find_lr', default=False, action='store_true')
    parser.add_argument('-offline', default=False, action='store_true')
    parser.add_argument('--num_dataloaders', default=16, type=int)
    parser.add_argument('--max_val_num', default=256, type=int)
    parser.add_argument('-cpu', default=False, action='store_true')
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--max_output_length', type=int, default=1024)
    parser.add_argument('--max_input_length', type=int, default=16384)
    parser.add_argument('--grad_accum', default=16, type=int)
    parser.add_argument('--hf_name', default='allenai/led-large-16384', choices=[
        'allenai/led-large-16384',
        'tau/bart-large-sled',
        'google/long-t5-tglobal-base',
        'google/pegasus-x-large',
        # PageSum
    ])

    args = parser.parse_args()
    args.weight_dir = os.path.join(args.data_dir, 'bhc_weights')
    os.makedirs(args.weight_dir, exist_ok=True)

    # Set same random seed for each run
    set_same_seed(args.seed)
    run(args)
