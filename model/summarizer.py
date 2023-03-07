from evaluate import load
import pytorch_lightning as pl

from deepspeed.ops.adam import DeepSpeedCPUAdam
import numpy as np
from sled import SledForConditionalGeneration
import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM, LongT5ForConditionalGeneration
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.trainer_pt_utils import LabelSmoother


class Summarizer(pl.LightningModule):
    def __init__(self, args, tokenizer, hf_name):
        """
        bart_model -> can load in pre-trained bart weights outside of this function (from reviser checkpoint)
        """
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer
        # assert self.hparams.max_input_length <= self.tokenizer.model_max_length
        print(f'Loading {hf_name}')
        if 'allenai/led' in hf_name:
            kwargs = {'gradient_checkpointing': True}
            self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_name, **kwargs)
        elif 'sled' in hf_name:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
        elif 't5' in hf_name:
            self.model = LongT5ForConditionalGeneration.from_pretrained(hf_name)
        else:
            raise Exception(f'Unrecognized HF model -> {hf_name}')
        # If we restore from reviser checkpoint or perturber checkpoint there will be extra tokens not in bart-base
        # see perturber.main and ref_reviser.main for additional tokens
        self.model.resize_token_embeddings(len(tokenizer))
        self.train_size = None
        self.rouge = load('rouge')
        self.label_smoother = LabelSmoother(epsilon=0.1)

    def training_step(self, batch, batch_idx):
        output = self.model(**batch, use_cache=False)
        loss = output.loss
        self.log('train/loss', loss, on_epoch=False, on_step=True, prog_bar=True, sync_dist=True)
        smooth_loss = self.label_smoother(output, batch['labels'])
        return smooth_loss

    def rouge_metrics(self, generated, gold):
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
        stats = self.rouge.compute(predictions=generated, references=gold, rouge_types=rouge_types)
        stats['rouge_mean'] = np.array(list(stats.values())).mean()
        return stats

    def shared_generate(self, batch, num_beams=1, length_penalty=4.0):
        kwargs = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'use_cache': True,
            'num_beams': num_beams,
            'min_length': 64,
            'max_length': self.hparams.max_output_length,
            'no_repeat_ngram_size': 3,
            'early_stopping': True,
            'length_penalty': length_penalty
        }

        if 'global_attention_mask' in batch:
            kwargs['global_attention_mask'] = batch['global_attention_mask']
        generated_ids = self.model.generate(**kwargs)
        generated_str = self.tokenizer.batch_decode(generated_ids.tolist(), skip_special_tokens=True)
        output_ids = batch['labels']
        output_ids[torch.where(batch['labels'] == -100)] = 1
        gold_str = self.tokenizer.batch_decode(output_ids.tolist(), skip_special_tokens=True)
        return generated_str, gold_str

    def validation_step(self, batch, batch_idx):
        output = self.model(**batch)
        loss = output.loss

        generated_str, gold_str = self.shared_generate(batch)
        metrics = self.rouge_metrics(generated_str, gold_str)
        for k, v in metrics.items():
            if v is None:
                continue
            self.log(f'val/{k}', v, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        self.log('val/loss', loss, on_epoch=True, on_step=False, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        nps = list(self.named_parameters())
        grouped_parameters = [
            {
                'params': [p for n, p in nps if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.hparams.weight_decay,
            },
            {
                'params': [p for n, p in nps if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(grouped_parameters, lr=self.hparams.lr)
        # optimizer = DeepSpeedCPUAdam(grouped_parameters, lr=self.hparams.lr)
        if self.hparams.debug:
            return optimizer

        warmup = min(self.hparams.warmup, self.hparams.max_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup, num_training_steps=self.hparams.max_steps)

        lr_scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
        }

        return [optimizer], [lr_scheduler]

    def predict_step(self, batch, batch_idx=None, **gen_kwargs):
        visit_id = batch.pop('visit_id')[0]
        patient_id = batch.pop('patient_id')[0]
        generated_str, gold_str = self.shared_generate(batch, **gen_kwargs)
        outputs = {'patient_id': patient_id, 'visit_id': visit_id}
        outputs.update(self.rouge_metrics(generated_str, gold_str))
        source = self.tokenizer.batch_decode(batch['input_ids'].tolist(), skip_special_tokens=True)[0]
        outputs['source'] = source
        outputs['prediction'] = generated_str[0]
        outputs['target'] = gold_str[0]
        return outputs

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop('v_num', None)
        return items
