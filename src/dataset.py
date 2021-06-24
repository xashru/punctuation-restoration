import torch
from config import *
from augmentation import *
import numpy as np
from argparser import parse_arguments
import os


def parse_data(file_path, tokenizer, sequence_len, token_style):
    """

    :param file_path: text file path that contains tokens and punctuations separated by tab in lines
    :param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
    :param sequence_len: maximum length of each sequence
    :param token_style: For getting index of special tokens in config.TOKEN_IDX
    :return: list of [tokens_index, punctuation_index, attention_masks, punctuation_mask], each having sequence_len
    punctuation_mask is used to ignore special indices like padding and intermediate sub-word token during evaluation
    """
    data_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f.read().split('\n') if line.strip()]
        idx = 0
        # loop until end of the entire text
        while idx < len(lines):
            x = [TOKEN_IDX[token_style]['START_SEQ']]
            y = [0]
            y_mask = [1]  # which positions we need to consider while evaluating i.e., ignore pad or sub tokens

            # loop until we have required sequence length
            # -1 because we will have a special end of sequence token at the end
            while len(x) < sequence_len - 1 and idx < len(lines):
                # print(lines[idx])
                word, punc = lines[idx].split('\t')
                tokens = tokenizer.tokenize(word.lower())
                # if taking these tokens exceeds sequence length we finish current sequence with padding
                # then start next sequence from this token
                if len(tokens) + len(x) >= sequence_len:
                    break
                else:
                    for i in range(len(tokens) - 1):
                        x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                        y.append(0)
                        y_mask.append(0)
                    if len(tokens) > 0:
                        x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                    else:
                        x.append(TOKEN_IDX[token_style]['UNK'])
                    y.append(punctuation_dict[punc])
                    y_mask.append(1)
                    idx += 1
            x.append(TOKEN_IDX[token_style]['END_SEQ'])
            y.append(0)
            y_mask.append(1)
            if len(x) < sequence_len:
                x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
                y = y + [0 for _ in range(sequence_len - len(y))]
                y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
            attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]
            data_items.append([x, y, attn_mask, y_mask])
    return data_items

def parse_data_by_block(file_path, tokenizer, sequence_len, token_style, stride_size):
    """
    :param file_path: text file path that contains tokens and punctuations separated by tab in lines
    :param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
    :param sequence_len: maximum length of each sequence
    :param token_style: For getting index of special tokens in config.TOKEN_IDX
    :return: list of [tokens_index, punctuation_index, attention_masks, punctuation_mask], each having sequence_len
    punctuation_mask is used to ignore special indices like padding and intermediate sub-word token during evaluation
    """
    
    data_items = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f.read().split('\n') if line.strip()]

    if (file_path.find('LJ_Speech') == -1):
        return sliding_window(lines, tokenizer, sequence_len, token_style, stride_size)

    process_at_index = 0
    while (len(lines[process_at_index:]) > 0):
        data_items += sliding_window(lines[process_at_index:process_at_index + WORD_BATCH], tokenizer, sequence_len, token_style, stride_size)
        process_at_index += WORD_BATCH
    
    # print('\nsequences:')
    # for i in data_items:
    #     print(i[0])
    return data_items

def sliding_window(line_block, tokenizer, sequence_len, token_style, stride_size):
    
    stride = stride_size
    sequence_size = sequence_len - 2
    stride_size_tokens = round(stride * sequence_size)
    if stride > 0.5:
        raise Exception("Maximum stride length is 0.5!")
    elif stride_size_tokens <= 0:
        raise Exception("Stride size too small, please increase stride length!")

    data_items = []
    lines = line_block
    
    words = list(map(split_word, lines))
    words = list(map(lambda x: [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x[0].lower())), x[1]], words))   # tokenize word inputs
    words = list(map(lambda x: [x[0], [0] * (len(x[0]) - 1) + [punctuation_dict[x[1]]]], words))                    # get y list
    words = list(map(lambda x: [fill_unknown(x[0], token_style), x[1]], words))                                     # fill empty tokens
    words = list(map(lambda x: [x[0], x[1], [0] * (len(x[0]) - 1) + [1]], words))                                   # fill y_mask

    full_tokens = []
    full_y = []
    full_y_mask = []
    for tokens, y, y_mask in words:
        full_tokens += tokens
        full_y += y
        full_y_mask += y_mask

    # print('\nfull tokens:', full_tokens, sep='\n')
    
    process_at_index = 0
    
    while (len(full_tokens[process_at_index:]) >= sequence_size):
        
        x = [TOKEN_IDX[token_style]['START_SEQ']]
        y = [0]
        y_mask = [1] 
        x += full_tokens[process_at_index:process_at_index + sequence_size]
        y += full_y[process_at_index:process_at_index + sequence_size]
        y_mask += full_y_mask[process_at_index:process_at_index + sequence_size]
        x.append(TOKEN_IDX[token_style]['END_SEQ'])
        y.append(0)
        y_mask.append(1)
        attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]
        data_items.append([x, y, attn_mask, y_mask, 0])

        process_at_index += stride_size_tokens
    
    if (len(full_tokens[process_at_index:]) > 0):
        x = [TOKEN_IDX[token_style]['START_SEQ']]
        y = [0]
        y_mask = [1] 
        x += full_tokens[process_at_index:]
        y += full_y[process_at_index:]
        y_mask += full_y_mask[process_at_index:]
        x.append(TOKEN_IDX[token_style]['END_SEQ'])
        y.append(0)
        y_mask.append(1)
        if len(x) < sequence_len:
            x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
            y = y + [0 for _ in range(sequence_len - len(y))]
            y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
        attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]
        data_items.append([x, y, attn_mask, y_mask, 1])
        
    data_items[-1][-1] = 1
    # data_items = list(map(lambda x: x + [1] if x == data_items[-1] else x + [0], data_items))
    # print('\nsequences:')
    # for i in data_items:
    #     print(i)
    
    # print(data_items)
    return data_items

def split_word(x):
    return x.split('\t')

def fill_unknown(lst, token_style):
    if len(lst) == 0:
        return [(TOKEN_IDX[token_style]['UNK'])]
    return lst

class Dataset(torch.utils.data.Dataset):
    def __init__(self, files, tokenizer, sequence_len, token_style, is_sliding_window, stride_size, 
            is_train=False, augment_rate=0.1, augment_type='substitute'):
        """

        :param files: single file or list of text files containing tokens and punctuations separated by tab in lines
        :param tokenizer: tokenizer that will be used to further tokenize word for BERT like models
        :param sequence_len: length of each sequence
        :param token_style: For getting index of special tokens in config.TOKEN_IDX
        :param augment_rate: token augmentation rate when preparing data
        :param is_train: if false do not apply augmentation
        """
        if isinstance(files, list):
            self.data = []
            for file in files:
                if (is_sliding_window):
                    self.data += parse_data_by_block(file, tokenizer, sequence_len, token_style, stride_size)
                else:
                    self.data += parse_data(file, tokenizer, sequence_len, token_style)
        else:
            if (is_sliding_window):
                self.data = parse_data_by_block(files, tokenizer, sequence_len, token_style, stride_size)
            else:
                self.data = parse_data(files, tokenizer, sequence_len, token_style)
        self.sequence_len = sequence_len
        self.augment_rate = augment_rate
        self.token_style = token_style
        self.is_train = is_train
        self.augment_type = augment_type
        self.is_sliding_window = is_sliding_window

    def __len__(self):
        return len(self.data)

    def _augment(self, x, y, y_mask):
        x_aug = []
        y_aug = []
        y_mask_aug = []
        for i in range(len(x)):
            r = np.random.rand()
            if r < self.augment_rate:
                AUGMENTATIONS[self.augment_type](x, y, y_mask, x_aug, y_aug, y_mask_aug, i, self.token_style)
            else:
                x_aug.append(x[i])
                y_aug.append(y[i])
                y_mask_aug.append(y_mask[i])

        if len(x_aug) > self.sequence_len:
            # len increased due to insert
            x_aug = x_aug[0:self.sequence_len]
            y_aug = y_aug[0:self.sequence_len]
            y_mask_aug = y_mask_aug[0:self.sequence_len]
        elif len(x_aug) < self.sequence_len:
            # len decreased due to delete
            x_aug = x_aug + [TOKEN_IDX[self.token_style]['PAD'] for _ in range(self.sequence_len - len(x_aug))]
            y_aug = y_aug + [0 for _ in range(self.sequence_len - len(y_aug))]
            y_mask_aug = y_mask_aug + [0 for _ in range(self.sequence_len - len(y_mask_aug))]

        attn_mask = [1 if token != TOKEN_IDX[self.token_style]['PAD'] else 0 for token in x]
        return x_aug, y_aug, attn_mask, y_mask_aug

    def __getitem__(self, index):
        x = self.data[index][0]
        y = self.data[index][1]
        attn_mask = self.data[index][2]
        y_mask = self.data[index][3]

        if self.is_train and self.augment_rate > 0:
            x, y, attn_mask, y_mask = self._augment(x, y, y_mask)

        x = torch.tensor(x)
        y = torch.tensor(y)
        attn_mask = torch.tensor(attn_mask)
        y_mask = torch.tensor(y_mask)
        

        if self.is_sliding_window:
            seq_count_in_block = self.data[index][4]
            seq_count_in_block = torch.tensor(seq_count_in_block)
            return x, y, attn_mask, y_mask, seq_count_in_block
            
        
        return x, y, attn_mask, y_mask

# For inspection and debugging dataset.py
# args = parse_arguments()
# tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)
# token_style = MODELS[args.pretrained_model][3]
# ar = 0.40
# sequence_len = 20
# aug_type = args.augment_type
# use_window = True
# stride_size = 0.5

# Dataset('data/en/test2011 copy', 
#        tokenizer=tokenizer, sequence_len=sequence_len, token_style=token_style, is_train=True, 
#        augment_rate=ar, augment_type=aug_type, is_sliding_window=use_window, stride_size=stride_size)
# Dataset('data/en/no_sentence_test', 
#        tokenizer=tokenizer, sequence_len=sequence_len, token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type)
# Dataset('data/LJ_Speech/non_fiction_20%', 
#        tokenizer=tokenizer, sequence_len=sequence_len, token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type)
