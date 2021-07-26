import re
import torch
import numpy as np

import argparse
from model import DeepPunctuation
from config import *

parser = argparse.ArgumentParser(description='Punctuation restoration inference on text file')
parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
parser.add_argument('--pretrained-model', default='xlm-roberta-large', type=str, help='pretrained language model')
parser.add_argument('--lstm-dim', default=-1, type=int,
                    help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
parser.add_argument('--use-crf', default=False, type=lambda x: (str(x).lower() == 'true'),
                    help='whether to use CRF layer or not')
parser.add_argument('--language', default='en', type=str, help='language English (en) oe Bangla (bn)')
parser.add_argument('--in-file', default='data/test_en.txt', type=str, help='path to inference file')
parser.add_argument('--weight-path', default='xlm-roberta-large.pt', type=str, help='model weight path')
parser.add_argument('--sequence-length', default=256, type=int,
                    help='sequence length to use when preparing dataset (default 256)')
parser.add_argument('--out-file', default='data/test_en_out.txt', type=str, help='output file location')
parser.add_argument('--sliding-window', default=True, type=lambda x: (str(x).lower() == 'true'), help='use sliding window implementation')
parser.add_argument('--stride_size', default=0.5, type=float, help='new sequence at every stride')

args = parser.parse_args()

stride_size = args.stride_size
# tokenizer
tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)
token_style = MODELS[args.pretrained_model][3]

# logs
model_save_path = args.weight_path

# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
deep_punctuation.to(device)


def inference_original():
    deep_punctuation.load_state_dict(torch.load(model_save_path))
    deep_punctuation.eval()

    with open(args.in_file, 'r', encoding='utf-8') as f:
        text = f.read()
    text = re.sub(r"[,:\-–.!;?]", '', text)
    words_original_case = text.split()
    words = text.lower().split()

    word_pos = 0
    sequence_len = args.sequence_length
    result = ""
    decode_idx = 0
    punctuation_map = {0: '', 1: ',', 2: '.', 3: '?'}
    if args.language != 'en':
        punctuation_map[2] = '।'

    while word_pos < len(words):
        x = [TOKEN_IDX[token_style]['START_SEQ']]
        y_mask = [0]

        while len(x) < sequence_len and word_pos < len(words):
            tokens = tokenizer.tokenize(words[word_pos])
            if len(tokens) + len(x) >= sequence_len:
                break
            else:
                for i in range(len(tokens) - 1):
                    x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
                    y_mask.append(0)
                x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
                y_mask.append(1)
                word_pos += 1
        x.append(TOKEN_IDX[token_style]['END_SEQ'])
        y_mask.append(0)
        if len(x) < sequence_len:
            x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
            y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
        attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]

        x = torch.tensor(x).reshape(1,-1)
        y_mask = torch.tensor(y_mask)
        attn_mask = torch.tensor(attn_mask).reshape(1,-1)
        x, attn_mask, y_mask = x.to(device), attn_mask.to(device), y_mask.to(device)

        with torch.no_grad():
            if args.use_crf:
                y = torch.zeros(x.shape[0])
                y_predict = deep_punctuation(x, attn_mask, y)
                y_predict = y_predict.view(-1)
            else:
                y_predict = deep_punctuation(x, attn_mask)
                y_predict = y_predict.view(-1, y_predict.shape[2])
                y_predict = torch.argmax(y_predict, dim=1).view(-1)
        for i in range(y_mask.shape[0]):
            if y_mask[i] == 1:
                result += words_original_case[decode_idx] + punctuation_map[y_predict[i].item()] + ' '
                decode_idx += 1
    print('Punctuated text')
    print(result)
    with open(args.out_file, 'w', encoding='utf-8') as f:
        f.write(result)


def fill_unknown(lst, token_style):
    if len(lst) == 0:
        return [(TOKEN_IDX[token_style]['UNK'])]
    return lst

def sum_overlapping(x, seq_count_in_block):
    # same function as in train.py and test.py but there is no need for buffer sequence since there are no batches during inference
    # and we can infer the whole corpus at once  
    #################### sum up overlapping sequences ####################

    # truncate all first and last token
    x = x[:, 1:-1]
    assert x.shape[1] % 2 == 0, 'sequence length should be an even number'
    
    # split into first half and second half of sequence 
    x = x.reshape(x.shape[0], 2, -1)
    
    
    # add all second half to next first half (exclude end of block and last seq)
    # get all relevant second halves
    sum_of_seq = x
    exclude_first_half = torch.ones(x.shape[0]).to(device)
    second_half_mask = torch.clone(seq_count_in_block)
    to_restore = False
    if 1 in second_half_mask[:-1]:
        end_block_index = (second_half_mask[:-1] == 1).nonzero().item()
        to_restore = True

    second_half_mask[-1] = 1
    second_half_mask = torch.column_stack((exclude_first_half, second_half_mask))
    second_half_mask = second_half_mask.unsqueeze(-1)
    sum_of_seq = torch.masked_select(sum_of_seq, second_half_mask == 0).to(device)
    sum_of_seq = sum_of_seq.reshape(-1, 1, x.shape[2])

    
    # append zeros at the start to account for removed tensors
    sum_of_seq = torch.cat((sum_of_seq, torch.zeros(sum_of_seq.shape[0], 1, sum_of_seq.shape[2]).to(device)), 1)
    if to_restore:
        for i in end_block_index:
            sum_of_seq = torch.cat((sum_of_seq[:i[0]], torch.zeros(1, sum_of_seq.shape[1], sum_of_seq.shape[2]).to(device), sum_of_seq[i[0]:]))
    
    sum_of_seq = torch.cat((torch.zeros(1, sum_of_seq.shape[1], sum_of_seq.shape[2]).to(device), sum_of_seq))

    x = x + sum_of_seq

    # get all first half + end of word batches as final output
    include_first_half = torch.ones(x.shape[0]).to(device)
    seq_mask = torch.clone(seq_count_in_block)
    seq_mask = torch.column_stack((include_first_half, seq_mask))
    seq_mask = seq_mask.unsqueeze(-1)
    x = torch.masked_select(x, seq_mask == 1)

    return x

def get_merged_values(y, seq_count_in_block):
    # same function as in train.py and test.py
    y = y[:, 1:-1]
    y = y.reshape(y.shape[0], 2, -1)

    # get all first halves + end of word batches as final output
    include_first_half = torch.ones(y.shape[0]).to(device)
    seq_mask = torch.clone(seq_count_in_block)
    seq_mask = torch.column_stack((include_first_half, seq_mask))
    seq_mask = seq_mask.unsqueeze(-1)
    y = torch.masked_select(y, seq_mask == 1)
    return y

def inference_window():
    deep_punctuation.load_state_dict(torch.load(model_save_path))
    deep_punctuation.eval()

    with open(args.in_file, 'r', encoding='utf-8') as f:
        text = f.read()
    text = re.sub(r"[,:\-–.!;?]", '', text) 
    words_original_case = text.split()
    words = text.lower().split()            

    word_pos = 0
    sequence_len = args.sequence_length
    result = ""
    decode_idx = 0
    punctuation_map = {0: '', 1: ',', 2: '.', 3: '?'}
    if args.language != 'en':
        punctuation_map[2] = '।'

    ########### apply sliding window implementation to duplicate inputs -- reference to dataset.py ##############################
    stride = stride_size
    sequence_size = sequence_len - 2                        # account for start and end token
    stride_size_tokens = round(stride * sequence_size)      # number of word tokens to take in half a sequence
    if stride > 0.5:
        raise Exception("Maximum stride length is 0.5!")
    elif stride_size_tokens <= 0:
        raise Exception("Stride size too small, please increase stride length!")

    words = list(map(lambda x: tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x.lower())), words))  # tokenize word inputs
    words = list(map(lambda x: fill_unknown(x, token_style), words))                                    # fill empty tokens
    words = list(map(lambda x: [x, [0] * (len(x) - 1) + [1]], words))                                   # fill y_mask
    
    # obtain full list of tokens, and y_mask
    full_tokens = []
    full_y_mask = []
    for tokens, y_mask in words:
        full_tokens += tokens
        full_y_mask += y_mask
    
    process_at_index = 0
    data_items = []

    while (len(full_tokens[process_at_index:]) >= sequence_size):   # loop until no more tokens to process
        # initialize sequence with start token
        x = [TOKEN_IDX[token_style]['START_SEQ']]
        y_mask = [0] 

        # add sequence until required length 
        x += full_tokens[process_at_index:process_at_index + sequence_size]
        y_mask += full_y_mask[process_at_index:process_at_index + sequence_size]
        x.append(TOKEN_IDX[token_style]['END_SEQ'])
        y_mask.append(0)

        attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x] # initialize attn_mask for full sequence
        data_items.append([x, attn_mask, y_mask])                                       # add sequence
        process_at_index += stride_size_tokens                                          # advance by stride size(0.5)
    
    # account for remaining tokens after while loop
    if (len(full_tokens[process_at_index:]) > 0):
        x = [TOKEN_IDX[token_style]['START_SEQ']]
        y_mask = [0] 
        x += full_tokens[process_at_index:]
        y_mask += full_y_mask[process_at_index:]
        x.append(TOKEN_IDX[token_style]['END_SEQ'])
        y_mask.append(0)
        if len(x) < sequence_len:
            x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
            y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
        attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]
        data_items.append([x, attn_mask, y_mask])
    ################################## end of sliding window ###############################################

    with torch.no_grad():
        x_combined = torch.empty(0).to(device)
        y_predict_combined = torch.empty(0).to(device)
        y_mask_combined = torch.empty(0).to(device)
        for seq in data_items:
            x = torch.tensor(seq[0]).reshape(1,-1)
            attn_mask = torch.tensor(seq[1]).reshape(1,-1)
            y_mask = torch.tensor(seq[2]).reshape(1,-1)
            x, attn_mask, y_mask = x.to(device), attn_mask.to(device), y_mask.to(device)
            y_predict = deep_punctuation(x, attn_mask)

            # reduce end weights of sequences: sliding window triangle 0 to 1
            if (sequence_len % 2 == 0):
                middle_index = sequence_len // 2
                left_weights = np.linspace(0.5, 1, middle_index)
                weights = np.append(left_weights, np.flip(left_weights))
                weights = weights.reshape(-1, 1)
            elif (sequence_len % 2 != 0):
                middle_index = (sequence_len // 2) + 1
                left_weights = np.linspace(0.5, 1, middle_index)
                weights = np.append(left_weights, np.flip(left_weights[:-1]))
                weights = weights.reshape(-1, 1)
            weighted_window = torch.from_numpy(weights).to(device)
            y_predict = y_predict * weighted_window

            x_combined = torch.cat((x_combined, x))
            y_predict_combined = torch.cat((y_predict_combined, y_predict))
            y_mask_combined = torch.cat((y_mask_combined, y_mask))
        seq_count_in_block = torch.zeros(x_combined.shape[0]).to(device)
        seq_count_in_block[-1] = 1

        # concatenate everything into single tensor for sum overlapping function
        # since we are trying to predict the values, we can combine the whole corpus into a tensor, there is no need to overlap between batches compared to in validation while training
        x = sum_overlapping(x_combined, seq_count_in_block)
        y_predict = sum_overlapping(y_predict_combined, seq_count_in_block)
        y_mask = get_merged_values(y_mask_combined, seq_count_in_block)
        
        y_predict = y_predict.view(-1, 4)
        y_predict = torch.argmax(y_predict, dim=1).view(-1)
            
        for i in range(y_mask.shape[0]):
            if y_mask[i] == 1:
                result += words_original_case[decode_idx] + punctuation_map[y_predict[i].item()] + ' '
                decode_idx += 1

    print('Punctuated text')
    print(result)
    with open(args.out_file, 'w', encoding='utf-8') as f:
        f.write(result)


if __name__ == '__main__':
    if args.sliding_window:
        inference_window()
    else:
        inference_original()
