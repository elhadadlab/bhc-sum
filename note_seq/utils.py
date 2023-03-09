import torch


def prepend_partial(partial, source_str):
    return f'<doc-sep>Title: Discharge Summary\nBRIEF HOSPITAL COURSE:\n{partial}\n{source_str}'


class Note2NoteCollate:
    def __init__(self, tokenizer, add_global_att, max_input_length=16348, max_output_length=512, add_cols=None):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        assert self.max_input_length <= tokenizer.model_max_length
        self.max_output_length = max_output_length
        self.pad_id = tokenizer.pad_token_id
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

        # partial_inputs = self.tokenizer(
        #     [x['partial'] for x in batch_list],
        #     padding='longest',
        #     truncation=True,
        #     max_length=self.max_output_length,
        #     return_tensors='pt'
        # )

        outputs = self.tokenizer(
            [x['target'] for x in batch_list],
            padding='longest',
            truncation=True,
            max_length=self.max_output_length,
            return_tensors='pt'
        )

        batch = {}
        batch['input_ids'] = inputs.input_ids
        # batch['attention_mask'] = inputs.attention_mask

        # batch['partial_input_ids'] = partial_inputs.input_ids
        # batch['partial_attention_mask'] = partial_inputs.attention_mask

        if self.add_global_att:
            # create 0 global_attention_mask lists
            batch['global_attention_mask'] = torch.FloatTensor(len(batch['input_ids']) * [
                [0 for _ in range(len(batch['input_ids'][0]))]
            ])

            # the 1st element of each sequence in batch should be flipped to 1
            batch['global_attention_mask'][:, 0] = 1

            # # create 0 global_attention_mask lists
            # batch['partial_global_attention_mask'] = torch.FloatTensor(len(batch['partial_input_ids']) * [
            #     [0 for _ in range(len(batch['partial_input_ids'][0]))]
            # ])
            #
            # # the 1st element of each sequence in batch should be flipped to 1
            # batch['partial_global_attention_mask'][:, 0] = 1

        batch['labels'] = outputs.input_ids
        # We have to make sure that the PAD token is ignored
        batch['labels'][torch.where(batch['labels'] == 1)] = -100
        for col in self.add_cols + ['curr_note_idx']:
            batch[col] = [x[col] for x in batch_list]
        return batch
