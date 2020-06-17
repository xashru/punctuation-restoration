import torch
from config import *
import numpy as np


def parse_data(file_path, tokenizer, sequence_len, token_style):
    """

    :param file_path: text file path that contains tokens and punctuations separated by tab in lines
    :param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
    :param sequence_len: length of each sequence
    :param token_style: For getting index of special tokens in config.TOKEN_IDX
    :return: list of [tokens_index, punctuation_index, attention_masks], each having sequence_len length
    """
    data_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f.read().split('\n') if line.strip()]
        idx = 0
        # loop until end of the entire text
        while idx < len(lines):
            x = [TOKEN_IDX[token_style]['START_SEQ']]
            y = [0]

            # loop until we have required sequence length
            # -1 because we will have a special end of sequence token at the end
            while len(x) < sequence_len - 1 and idx < len(lines):
                word, punc = lines[idx].split('\t')
                tokens = tokenizer.tokenize(word)
                # if taking these tokens exceeds sequence length we finish current sequence with padding
                # then start next sequence from this token
                if len(tokens) + len(x) >= sequence_len:
                    x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - 1 - len(x))]
                    y = y + [-1 for _ in range(sequence_len - 1 - len(y))]
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                        y.append(0)
                    if len(tokens) > 0:
                        x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                    else:
                        x.append(TOKEN_IDX[token_style]['UNK'])
                    y.append(punctuation_dict[punc])
                    idx += 1
            # This might happen when we reach the end of the text
            if len(x) < sequence_len - 1:
                x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - 1 - len(x))]
                y = y + [-1 for _ in range(sequence_len - 1 - len(y))]
            x.append(TOKEN_IDX[token_style]['END_SEQ'])
            y.append(0)
            attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]
            data_items.append([x, y, attn_mask])
    print('Dataset len:', len(data_items))
    return data_items


class Dataset(torch.utils.data.Dataset):
    def __init__(self, files, tokenizer, sequence_len, token_style, is_train=False, augment_rate=0.1, augment_type='all'):
        """

        :param files: single file or list of text files containing tokens and punctuations separated by tab in lines
        :param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
        :param sequence_len: length of each sequence
        :param token_style: For getting index of special tokens in config.TOKEN_IDX
        :param augment_rate: token change rate when preparing data
        :param is_train: if false do not apply augmentation
        """
        if isinstance(files, list):
            self.data = []
            for file in files:
                self.data += parse_data(file, tokenizer, sequence_len, token_style)
        else:
            self.data = parse_data(files, tokenizer, sequence_len, token_style)
        self.sequence_len = sequence_len
        self.augment_rate = augment_rate
        self.token_style = token_style
        self.is_train = is_train
        # which augmentations to use
        # TODO: Implement this
        self.augment_type = augment_type

    def __len__(self):
        return len(self.data)

    def _augment(self, x, y):
        x_aug = []
        y_aug = []
        for i in range(len(x)):
            r = np.random.rand()
            if r < self.augment_rate:
                # 0->replace, 1->insert, 2->delete
                aug_type = np.random.randint(0, 3)
                if aug_type == 0:
                    x_aug.append(TOKEN_IDX[self.token_style]['UNK'])
                    y_aug.append(y[i])
                elif aug_type == 1:
                    x_aug.append(TOKEN_IDX[self.token_style]['UNK'])
                    y_aug.append(0)
                    x_aug.append(x[i])
                    y_aug.append(y[i])
                elif aug_type == 2:
                    # delete if there is no punctuation mark in this position
                    if y[i] != 0:
                        x_aug.append(x[i])
                        y_aug.append(y[i])
            else:
                x_aug.append(x[i])
                y_aug.append(y[i])

        if len(x_aug) < self.sequence_len - 1:
            # len decreased due to delete
            x_aug = x_aug + [TOKEN_IDX[self.token_style]['PAD'] for _ in range(self.sequence_len - 1 - len(x_aug))]
            y_aug = y_aug + [-1 for _ in range(self.sequence_len - 1 - len(y_aug))]
        elif len(x_aug) >= self.sequence_len:
            # len increased due to insert
            x_aug = x_aug[0:self.sequence_len - 1]
            y_aug = y_aug[0:self.sequence_len - 1]

        x_aug.append(TOKEN_IDX[self.token_style]['END_SEQ'])
        y_aug.append(0)
        attn_mask = [1 if token != TOKEN_IDX[self.token_style]['PAD'] else 0 for token in x]
        return x_aug, y_aug, attn_mask

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        attn_mask = self.data[index][2]

        # if self.is_train and self.augment_rate > 0:
        #     x, y, attn_mask = self._augment(x, y)

        x = torch.tensor(x)
        y = torch.tensor(y)
        attn_mask = torch.tensor(attn_mask)
        if self.is_train:
            r = torch.rand(x.shape) < self.augment_rate
            x[r] = TOKEN_IDX[self.token_style]['UNK']
        return x, y, attn_mask
