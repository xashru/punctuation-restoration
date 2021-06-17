import os
import torch
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
import sklearn.preprocessing as preprocessing

import argparse
from dataset import Dataset
from model import DeepPunctuation
from config import *


parser = argparse.ArgumentParser(description='Punctuation restoration test')
parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
parser.add_argument('--pretrained-model', default='roberta-large', type=str, help='pretrained language model')
parser.add_argument('--lstm-dim', default=-1, type=int,
                    help='hidden dimension in LSTM layer, if -1 is set equal to hidden dimension in language model')
parser.add_argument('--data-path', default='data/test', type=str, help='path to test datasets')
parser.add_argument('--weight-path', type=str, help='model weight path')
parser.add_argument('--sequence-length', default=256, type=int,
                    help='sequence length to use when preparing dataset (default 256)')
parser.add_argument('--batch-size', default=8, type=int, help='batch size (default: 8)')
parser.add_argument('--save-path', default='results/', type=str, help='model and log save directory')
parser.add_argument('--sliding-window', default=True, type=lambda x: (str(x).lower() == 'true'), help='use sliding window implementation')
parser.add_argument('--stride_size', default=0.5, type=float, help='new sequence at every stride')

args = parser.parse_args()


# tokenizer
tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)
token_style = MODELS[args.pretrained_model][3]

test_files = os.listdir(args.data_path)
test_set = []

for file in test_files:
    test_set.append(Dataset(os.path.join(args.data_path, file), tokenizer=tokenizer, sequence_len=args.sequence_length,
                            token_style=token_style, is_train=False, is_sliding_window=args.sliding_window, stride_size=args.stride_size))

# Data Loaders
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': False,
    'num_workers': 0
}

test_loaders = [torch.utils.data.DataLoader(x, **data_loader_params) for x in test_set]

# logs
model_save_path = os.path.join(args.weight_path.strip(), args.pretrained_model + '.pt')
log_path = os.path.join(args.save_path.strip(), args.pretrained_model + '_test_logs.txt')

# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=False, lstm_dim=args.lstm_dim)
deep_punctuation.to(device)

def sum_overlapping(x, buffer_sequence, to_be_processed, seq_count_in_block):
    #################### sum up overlapping sequences ####################

    # truncate all first and last token
    x = x[:, 1:-1]
    assert x.shape[1] % 2 == 0, 'sequence length should be an even number'
    
    # split into first half and second half of sequence 
    x = x.reshape(x.shape[0], 2, -1)
    
    # handle sequence in buffer from previous batch
    if to_be_processed:
        # print("Added buffer sequence from previous\n")
        x[0][0] += buffer_sequence

    # buffer the second half of last sequence for next batch
    if seq_count_in_block[-1] == 0:     # needs to be added to next batch
        buffer_sequence = x[-1][-1]
        # print('buffered: ', x[-1][-1])
        to_be_processed = True
    elif seq_count_in_block[-1] == 1:   # last sequence is end of word block
        to_be_processed = False

    
    # add all second half to next first half (exclude end of block and last seq)

    # get all relevant second halves
    sum_of_seq = x
    exclude_first_half = torch.ones(x.shape[0]).to(device)
    second_half_mask = torch.clone(seq_count_in_block)
    to_restore = False
    if 1 in second_half_mask[:-1]:
        end_block_index = (second_half_mask == 1).nonzero().item()
        to_restore = True

    second_half_mask[-1] = 1
    second_half_mask = torch.column_stack((exclude_first_half, second_half_mask))
    second_half_mask = second_half_mask.unsqueeze(-1)
    sum_of_seq = torch.masked_select(sum_of_seq, second_half_mask == 0).to(device)
    sum_of_seq = sum_of_seq.reshape(-1, 1, x.shape[2])

    
    # append zeros at the start to account for removed tensors
    sum_of_seq = torch.cat((sum_of_seq, torch.zeros(sum_of_seq.shape[0], 1, sum_of_seq.shape[2]).to(device)), 1)
    if to_restore:
        sum_of_seq = torch.cat((sum_of_seq[:end_block_index], torch.zeros(1, sum_of_seq.shape[1], sum_of_seq.shape[2]), sum_of_seq[end_block_index:]))
    
    sum_of_seq = torch.cat((torch.zeros(1, sum_of_seq.shape[1], sum_of_seq.shape[2]).to(device), sum_of_seq))

    x = x + sum_of_seq

    # get all first half + end of word batches as final output
    include_first_half = torch.ones(x.shape[0]).to(device)
    seq_mask = torch.clone(seq_count_in_block)
    seq_mask = torch.column_stack((include_first_half, seq_mask))
    seq_mask = seq_mask.unsqueeze(-1)
    x = torch.masked_select(x, seq_mask == 1)

    return x, buffer_sequence, to_be_processed

def get_merged_values(y, seq_count_in_block):
    y = y[:, 1:-1]
    y = y.reshape(y.shape[0], 2, -1)

    # get all first half + end of word batches as final output
    include_first_half = torch.ones(y.shape[0]).to(device)
    seq_mask = torch.clone(seq_count_in_block)
    seq_mask = torch.column_stack((include_first_half, seq_mask))
    seq_mask = seq_mask.unsqueeze(-1)
    y = torch.masked_select(y, seq_mask == 1)

    return y

def test(data_loader):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    """
    num_iteration = 0
    deep_punctuation.eval()
    correct = 0
    total = 0

    tp = np.zeros(1+len(punctuation_dict), dtype=np.int)
    fp = np.zeros(1+len(punctuation_dict), dtype=np.int)
    fn = np.zeros(1+len(punctuation_dict), dtype=np.int)
    cm = np.zeros((len(punctuation_dict), len(punctuation_dict)), dtype=np.int)
    
    with torch.no_grad():
        # save buffer sequence from previous batch
        x_buffer = torch.empty(0)
        y_predict_buffer = torch.empty(0)
        y_mask_buffer = torch.empty(0)
        x_to_be_processed = False
        y_predict_to_be_processed = False
        y_mask_to_be_processed = False

        for x, y, att, y_mask, seq_count_in_block in tqdm(data_loader, desc='test'):
            x, y, att, y_mask, seq_count_in_block = x.to(device), y.to(device), att.to(device), y_mask.to(device), seq_count_in_block.to(device)       
            y_predict = deep_punctuation(x, att)

            # reduce end weights of sequences
            if args.sliding_window:
                sequence_len = args.sequence_length
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

            x, x_buffer, x_to_be_processed = sum_overlapping(x, x_buffer, x_to_be_processed, seq_count_in_block)
            y_predict, y_predict_buffer, y_predict_to_be_processed = sum_overlapping(y_predict, y_predict_buffer, y_predict_to_be_processed, seq_count_in_block)
            y = get_merged_values(y, seq_count_in_block)
            y_mask, y_mask_buffer, y_mask_to_be_processed = sum_overlapping(y_mask, y_mask_buffer, y_mask_to_be_processed, seq_count_in_block)

            y_predict = y_predict.view(-1, 4)
            y = y.long()
            
            # remove start, end, pad tokens for loss function
            start_token = TOKEN_IDX[token_style]['START_SEQ']
            end_token = TOKEN_IDX[token_style]['END_SEQ']
            pad_token = TOKEN_IDX[token_style]['PAD']
            combined_x_values = x.reshape(-1,1)
            for i in range(combined_x_values.shape[0] - 1, -1, -1):
                if (combined_x_values[i] == start_token or combined_x_values[i] == end_token or combined_x_values[i] == pad_token):
                    y_predict = torch.cat((y_predict[:i], y_predict[i + 1:]))
                    y = torch.cat((y[:i], y[i + 1:]))
                    y_mask = torch.cat((y_mask[:i], y_mask[i + 1:]))

            y_predict = torch.argmax(y_predict, dim=1).view(-1)

            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
            for i in range(y.shape[0]):
                if y_mask[i] == 0:
                    # we can ignore this because we know there won't be any punctuation in this position
                    # since we created this position due to padding or sub-word tokenization
                    continue
                cor = y[i]
                prd = y_predict[i]
                if cor == prd:
                    tp[cor] += 1
                else:
                    fn[cor] += 1
                    fp[prd] += 1
                cm[cor][prd] += 1
    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, correct/total, cm


def run():
    deep_punctuation.load_state_dict(torch.load(model_save_path))
    for i in range(len(test_loaders)):
        precision, recall, f1, accuracy, cm = test(test_loaders[i])
        log = test_files[i] + '\n' + 'Precision: ' + str(precision) + '\n' + 'Recall: ' + str(recall) + '\n' + \
            'F1 score: ' + str(f1) + '\n' + 'Accuracy:' + str(accuracy) + '\n' + 'Confusion Matrix' + str(cm) + '\n'
        print(log)
        with open(log_path, 'a') as f:
            f.write(log)

run()
