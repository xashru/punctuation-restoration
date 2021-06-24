import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import torch.multiprocessing
from tqdm import tqdm
from scipy.stats import norm
import sklearn.preprocessing as preprocessing

from argparser import parse_arguments
from dataset import Dataset
from model import DeepPunctuation
from config import *
import augmentation

torch.multiprocessing.set_sharing_strategy('file_system')   # https://github.com/pytorch/pytorch/issues/11201

args = parse_arguments()

# for reproducibility
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# tokenizer
tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)

augmentation.tokenizer = tokenizer
augmentation.sub_style = args.sub_style 
augmentation.alpha_sub = args.alpha_sub
augmentation.alpha_del = args.alpha_del

token_style = MODELS[args.pretrained_model][3]
ar = args.augment_rate
sequence_len = args.sequence_length
aug_type = args.augment_type
pre_trained_model = args.trained_model_path
use_window = args.sliding_window
stride_size = args.stride_size
print("use window", use_window)
# Datasets
if args.language == 'english':
    train_set = Dataset(os.path.join(args.data_path, 'en/train2012'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type, is_sliding_window=use_window, stride_size=stride_size)
    val_set = Dataset(os.path.join(args.data_path, 'en/dev2012'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_ref = Dataset(os.path.join(args.data_path, 'en/test2011'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_asr = Dataset(os.path.join(args.data_path, 'en/test2011asr'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set = [val_set, test_set_ref, test_set_asr]
elif args.language == 'utt_with_ted_talk_no_asr':
    train_set = Dataset([os.path.join(args.data_path, 'utt/train_utt'), os.path.join(args.data_path, 'en/test2011')], tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type, is_sliding_window=use_window, stride_size=stride_size)
    val_set = Dataset(os.path.join(args.data_path, 'en/dev2012'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_ted_talk = Dataset(os.path.join(args.data_path, 'en/train2012'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set = [val_set]
    
elif args.language == 'utt_with_ted_talk_asr':
    train_set = Dataset([os.path.join(args.data_path, 'utt/train_utt'), os.path.join(args.data_path, 'en/test2011asr')], tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type, is_sliding_window=use_window, stride_size=stride_size)
    val_set = Dataset(os.path.join(args.data_path, 'en/dev2012'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_ted_talk = Dataset(os.path.join(args.data_path, 'en/train2012'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set = [val_set]
elif args.language == 'ted_talk_with_utt':
    train_set = Dataset([os.path.join(args.data_path, 'en/train2012'), os.path.join(args.data_path, 'utt_new_test/train_utt')], tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type, is_sliding_window=use_window, stride_size=stride_size)
    val_set = Dataset(os.path.join(args.data_path, 'utt_new_test/dev_utt'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    new_utt_test = Dataset(os.path.join(args.data_path, 'utt_new_test/utt_test'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set = [val_set, new_utt_test]
elif args.language == 'utt_with_LJ_speech':
    train_set = Dataset([os.path.join(args.data_path, 'utt/train_utt'), os.path.join(args.data_path, 'LJ_Speech/train_LJ_Speech_20%')], tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type, is_sliding_window=use_window, stride_size=stride_size)
    val_set = Dataset(os.path.join(args.data_path, 'LJ_Speech/dev_LJ_Speech_10%'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_LJ = Dataset(os.path.join(args.data_path, 'LJ_Speech/test_LJ_Speech_70%'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set = [val_set]
elif args.language == 'utt':
    train_set = Dataset(os.path.join(args.data_path, 'utt/train_utt'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type, is_sliding_window=use_window, stride_size=stride_size)
    val_set = Dataset(os.path.join(args.data_path, 'utt/dev_utt'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_ref = Dataset(os.path.join(args.data_path, 'utt/test_utt'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_asr = Dataset(os.path.join(args.data_path, 'en/test2011asr'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set = [val_set, test_set_ref, test_set_asr]
elif args.language == 'en_utt':
    train_set = Dataset([os.path.join(args.data_path, 'en/train2012'), os.path.join(args.data_path, 'utt/train_utt')], tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type, is_sliding_window=use_window, stride_size=stride_size)
    val_set = Dataset([os.path.join(args.data_path, 'en/dev2012'), os.path.join(args.data_path, 'utt/dev_utt')], tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_ref = Dataset(os.path.join(args.data_path, 'en/test2011'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_asr = Dataset(os.path.join(args.data_path, 'en/test2011asr'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_utt = Dataset(os.path.join(args.data_path, 'utt/test_utt'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set = [val_set, test_set_ref, test_set_asr, test_set_utt]
elif args.language == 'test':
    train_set = Dataset(os.path.join(args.data_path, 'en/test2011 copy'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type, is_sliding_window=use_window, stride_size=stride_size)
    val_set = Dataset(os.path.join(args.data_path, 'en/dev2012 copy'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_asr = Dataset(os.path.join(args.data_path, 'en/test2011asr copy'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set = [val_set, test_set_asr]
else:
    raise ValueError('Incorrect language argument for Dataset')

# Data Loaders
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 1
}

test_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': False,
    'num_workers': 1
}
train_loader = torch.utils.data.DataLoader(train_set, **data_loader_params)
val_loader = torch.utils.data.DataLoader(val_set, **test_loader_params)
test_loaders = [torch.utils.data.DataLoader(x, **test_loader_params) for x in test_set]

# logs
os.makedirs(args.save_path, exist_ok=True)
model_save_path = os.path.join(args.save_path.strip(), args.pretrained_model + '.pt')
log_path = os.path.join(args.save_path.strip(), args.pretrained_model + '.txt')


# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
deep_punctuation = DeepPunctuation(args.pretrained_model, freeze_bert=args.freeze_bert, lstm_dim=args.lstm_dim)
deep_punctuation.to(device)

#load previous model
if (args.trained_model_path != False):
    pretrained_model = os.path.join(args.trained_model_path.strip(), args.pretrained_model + '.pt')
    print('loading from ', pretrained_model)
    deep_punctuation.load_state_dict(torch.load(pretrained_model))

# print(deep_punctuation)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(deep_punctuation.parameters(), lr=args.lr, weight_decay=args.decay)

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
        sum_of_seq = torch.cat((sum_of_seq[:end_block_index], torch.zeros(1, sum_of_seq.shape[1], sum_of_seq.shape[2]).to(device), sum_of_seq[end_block_index:]))
    
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

def validate_original(data_loader):
    """
    :return: validation accuracy, validation loss
    """
    num_iteration = 0
    deep_punctuation.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc='eval'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            y_predict = deep_punctuation(x, att)
            y = y.view(-1)
            y_predict = y_predict.view(-1, y_predict.shape[2])
            
            #added to test if there is improvement from removing start end seq ---------- change(1)
            # # remove start, end, pad tokens for loss function
            start_token = TOKEN_IDX[token_style]['START_SEQ']
            end_token = TOKEN_IDX[token_style]['END_SEQ']
            pad_token = TOKEN_IDX[token_style]['PAD']
            combined_x_values = x.reshape(-1,1)
            for i in range(combined_x_values.shape[0] - 1, -1, -1):
                if (combined_x_values[i] == start_token or combined_x_values[i] == end_token or combined_x_values[i] == pad_token):
                    y_predict = torch.cat((y_predict[:i], y_predict[i + 1:]))
                    y = torch.cat((y[:i], y[i + 1:]))
                    y_mask = torch.cat((y_mask[:i], y_mask[i + 1:]))

            loss = criterion(y_predict, y)
            y_predict = torch.argmax(y_predict, dim=1).view(-1)
            val_loss += loss.item()
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
    return correct/total, val_loss/num_iteration

def validate_window(data_loader):
    """
    :return: validation accuracy, validation loss
    """
    num_iteration = 0
    deep_punctuation.eval()
    correct = 0
    total = 0
    val_loss = 0
    with torch.no_grad():
        # save buffer sequence from previous batch
        x_buffer = torch.empty(0)
        y_predict_buffer = torch.empty(0)
        y_mask_buffer = torch.empty(0)
        x_to_be_processed = False
        y_predict_to_be_processed = False
        y_mask_to_be_processed = False

        for x, y, att, y_mask, seq_count_in_block in tqdm(data_loader, desc='eval'):
            # print(x.shape, y.shape, att.shape, y_mask.shape)
            x, y, att, y_mask, seq_count_in_block = x.to(device), y.to(device), att.to(device), y_mask.to(device), seq_count_in_block.to(device)     
            y_predict = deep_punctuation(x, att)
            # print(x)
            # reduce end weights of sequences
            if (sequence_len % 2 == 0):
                middle_index = sequence_len // 2
                left_weights = np.linspace(0, 1, middle_index)
                weights = np.append(left_weights, np.flip(left_weights))
                weights = weights.reshape(-1, 1)
            elif (sequence_len % 2 != 0):
                middle_index = (sequence_len // 2) + 1
                left_weights = np.linspace(0, 1, middle_index)
                weights = np.append(left_weights, np.flip(left_weights[:-1]))
                weights = weights.reshape(-1, 1)
            weighted_window = torch.from_numpy(weights).to(device)
            y_predict = y_predict * weighted_window
            # print(y_mask.shape)    
            x, x_buffer, x_to_be_processed = sum_overlapping(x, x_buffer, x_to_be_processed, seq_count_in_block)
            y_predict, y_predict_buffer, y_predict_to_be_processed = sum_overlapping(y_predict, y_predict_buffer, y_predict_to_be_processed, seq_count_in_block)
            y = get_merged_values(y, seq_count_in_block)
            y_mask = get_merged_values(y_mask, seq_count_in_block)
            # print(x)
            y_predict = y_predict.view(-1, 4)
            y = y.long()
            # print(y_mask)

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
            
            loss = criterion(y_predict, y)
            y_predict = torch.argmax(y_predict, dim=1).view(-1)

            val_loss += loss.item()
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
    return correct/total, val_loss/num_iteration
    
def test_original(data_loader):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    """
    num_iteration = 0
    deep_punctuation.eval()
    # +1 for overall result
    tp = np.zeros(1+len(punctuation_dict), dtype=np.int)
    fp = np.zeros(1+len(punctuation_dict), dtype=np.int)
    fn = np.zeros(1+len(punctuation_dict), dtype=np.int)
    cm = np.zeros((len(punctuation_dict), len(punctuation_dict)), dtype=np.int)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, att, y_mask in tqdm(data_loader, desc='test'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            y_predict = deep_punctuation(x, att)
            y = y.view(-1)
            y_predict = y_predict.view(-1, y_predict.shape[2])

            # added to test if there is change from removing start end seq --------- change(1)
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

def test_window(data_loader):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    """
    num_iteration = 0
    deep_punctuation.eval()
    correct = 0
    total = 0

    # initialize true positive, false positive arrays... for [O, COMMA, PERIOD, QUESTION, OVERALL]
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
            sequence_len = args.sequence_length
            if (sequence_len % 2 == 0):
                middle_index = sequence_len // 2
                left_weights = np.linspace(0, 1, middle_index)
                weights = np.append(left_weights, np.flip(left_weights))
                weights = weights.reshape(-1, 1)
            elif (sequence_len % 2 != 0):
                middle_index = (sequence_len // 2) + 1
                left_weights = np.linspace(0, 1, middle_index)
                weights = np.append(left_weights, np.flip(left_weights[:-1]))
                weights = weights.reshape(-1, 1)
            weighted_window = torch.from_numpy(weights).to(device)
            y_predict = y_predict * weighted_window

            x, x_buffer, x_to_be_processed = sum_overlapping(x, x_buffer, x_to_be_processed, seq_count_in_block)
            y_predict, y_predict_buffer, y_predict_to_be_processed = sum_overlapping(y_predict, y_predict_buffer, y_predict_to_be_processed, seq_count_in_block)
            y = get_merged_values(y, seq_count_in_block)
            y_mask = get_merged_values(y_mask, seq_count_in_block)

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
            # print('y_shape: ', y.shape)
            for i in range(y.shape[0]):
                if y_mask[i] == 0:
                    # we can ignore this because we know there won't be any punctuation in this position
                    # since we created this position due to padding or sub-word tokenization
                    continue
                cor = y[i]
                prd = y_predict[i]
                if cor == prd:
                    tp[cor] += 1    #increase true positive count
                else:
                    fn[cor] += 1    #increase false negative for ground truth
                    fp[prd] += 1    #increase false positive for prediction
                cm[cor][prd] += 1   #increase confusion matrix

    # ignore first index which is for no punctuation
    # calculate overall punctuations
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, correct/total, cm

def train_window():
    with open(log_path, 'a') as f:
        f.write(str(args)+'\n')
    best_val_acc = 0

    for epoch in range(args.epoch):
    # for epoch in range(1):  # for inspecting
        # print('epoch: ', epoch)
        train_loss = 0.0
        train_iteration = 0
        correct = 0
        total = 0
        deep_punctuation.train()
        for x, y, att, y_mask, seq_count_in_block in tqdm(train_loader, desc='train'):
            x, y, att, y_mask, seq_count_in_block = x.to(device), y.to(device), att.to(device), y_mask.to(device), seq_count_in_block.to(device)  
            y_mask = y_mask.view(-1)
            y_predict = deep_punctuation(x, att)

            # reduce end weights of sequences: triangle function
            if (sequence_len % 2 == 0):
                middle_index = sequence_len // 2
                left_weights = np.linspace(0, 1, middle_index)
                weights = np.append(left_weights, np.flip(left_weights))
                weights = weights.reshape(-1, 1)
            elif (sequence_len % 2 != 0):
                middle_index = (sequence_len // 2) + 1
                left_weights = np.linspace(0, 1, middle_index)
                weights = np.append(left_weights, np.flip(left_weights[:-1]))
                weights = weights.reshape(-1, 1)

            # reduce end weights of sequences: sin^2 + cos^2 = 1 (Proof of concept)
            # if (sequence_len % 2 == 0):
            #     middle_index = sequence_len // 2
            #     zero_half_pi = np.linspace(0, np.pi/2, middle_index)
            #     sin_curve = np.sin(zero_half_pi)
            #     left_weights = np.square(sin_curve)
            #     cos_curve = np.cos(zero_half_pi)
            #     right_weights = np.square(cos_curve)
            #     weights = np.append(left_weights, right_weights)
            #     weights = weights.reshape(-1, 1)
            # elif (sequence_len % 2 != 0):
            #     middle_index = sequence_len // 2
            #     zero_half_pi = np.linspace(0, np.pi/2, middle_index)
            #     sin_curve = np.sin(zero_half_pi)
            #     left_weights = np.square(sin_curve)
            #     cos_curve = np.cos(zero_half_pi)
            #     right_weights = np.square(cos_curve)
            #     weights = np.append(left_weights, [1])
            #     weights = np.append(weights, right_weights)
            #     weights = weights.reshape(-1, 1)

            # print(weights)
            # print(weights[:len(weights)//2] + weights[len(weights)//2:])
            weighted_window = torch.from_numpy(weights).to(device)
            y_predict = y_predict * weighted_window

            y_predict = y_predict.view(-1, y_predict.shape[2])
            y = y.view(-1)
            
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
                    
            loss = criterion(y_predict, y)
            y_predict = torch.argmax(y_predict, dim=1).view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()

            optimizer.zero_grad()
            train_loss += loss.item()
            train_iteration += 1
            loss.backward()

            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(deep_punctuation.parameters(), args.gradient_clip)
            optimizer.step()

            y_mask = y_mask.view(-1)

            total += torch.sum(y_mask).item()

        train_loss /= train_iteration
        log = 'epoch: {}, Train loss: {}, Train accuracy: {}'.format(epoch, train_loss, correct / total)
        with open(log_path, 'a') as f:
            f.write(log + '\n')
        print(log)

        val_acc, val_loss = validate_window(val_loader)
        log = 'epoch: {}, Val loss: {}, Val accuracy: {}'.format(epoch, val_loss, val_acc)
        with open(log_path, 'a') as f:
            f.write(log + '\n')
        print(log)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(deep_punctuation.state_dict(), model_save_path)

    print('Best validation Acc:', best_val_acc)
    deep_punctuation.load_state_dict(torch.load(model_save_path))
    for loader in test_loaders:
        precision, recall, f1, accuracy, cm = test_window(loader)
        log = 'Precision: ' + str(precision) + '\n' + 'Recall: ' + str(recall) + '\n' + 'F1 score: ' + str(f1) + \
              '\n' + 'Accuracy:' + str(accuracy) + '\n' + 'Confusion Matrix' + str(cm) + '\n'
        print(log)
        with open(log_path, 'a') as f:
            f.write(log)
        log_text = ''
        for i in range(1, 5):
            log_text += str(precision[i] * 100) + ' ' + str(recall[i] * 100) + ' ' + str(f1[i] * 100) + ' '
        with open(log_path, 'a') as f:
            f.write(log_text[:-1] + '\n\n')

def train_original():
    with open(log_path, 'a') as f:
        f.write(str(args)+'\n')
    best_val_acc = 0

    for epoch in range(args.epoch):
    # for epoch in range(1):  # for inspecting
        # print('epoch: ', epoch)
        train_loss = 0.0
        train_iteration = 0
        correct = 0
        total = 0
        deep_punctuation.train()
        for x, y, att, y_mask in tqdm(train_loader, desc='train'):
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            y_predict = deep_punctuation(x, att)
            y_predict = y_predict.view(-1, y_predict.shape[2])
            y = y.view(-1)
            
            # remove start, end, pad tokens for loss function --------- change(1)
            start_token = TOKEN_IDX[token_style]['START_SEQ']
            end_token = TOKEN_IDX[token_style]['END_SEQ']
            pad_token = TOKEN_IDX[token_style]['PAD']
            combined_x_values = x.reshape(-1,1)
            for i in range(combined_x_values.shape[0] - 1, -1, -1):
                if (combined_x_values[i] == start_token or combined_x_values[i] == end_token or combined_x_values[i] == pad_token):
                    y_predict = torch.cat((y_predict[:i], y_predict[i + 1:]))
                    y = torch.cat((y[:i], y[i + 1:]))
                    y_mask = torch.cat((y_mask[:i], y_mask[i + 1:]))
                    
            loss = criterion(y_predict, y)
            y_predict = torch.argmax(y_predict, dim=1).view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()

            optimizer.zero_grad()
            train_loss += loss.item()
            train_iteration += 1
            loss.backward()

            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(deep_punctuation.parameters(), args.gradient_clip)
            optimizer.step()

            y_mask = y_mask.view(-1)

            total += torch.sum(y_mask).item()

        train_loss /= train_iteration
        log = 'epoch: {}, Train loss: {}, Train accuracy: {}'.format(epoch, train_loss, correct / total)
        with open(log_path, 'a') as f:
            f.write(log + '\n')
        print(log)

        val_acc, val_loss = validate_original(val_loader)
        log = 'epoch: {}, Val loss: {}, Val accuracy: {}'.format(epoch, val_loss, val_acc)
        with open(log_path, 'a') as f:
            f.write(log + '\n')
        print(log)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(deep_punctuation.state_dict(), model_save_path)

    print('Best validation Acc:', best_val_acc)
    deep_punctuation.load_state_dict(torch.load(model_save_path))
    for loader in test_loaders:
        precision, recall, f1, accuracy, cm = test_original(loader)
        log = 'Precision: ' + str(precision) + '\n' + 'Recall: ' + str(recall) + '\n' + 'F1 score: ' + str(f1) + \
              '\n' + 'Accuracy:' + str(accuracy) + '\n' + 'Confusion Matrix' + str(cm) + '\n'
        print(log)
        with open(log_path, 'a') as f:
            f.write(log)
        log_text = ''
        for i in range(1, 5):
            log_text += str(precision[i] * 100) + ' ' + str(recall[i] * 100) + ' ' + str(f1[i] * 100) + ' '
        with open(log_path, 'a') as f:
            f.write(log_text[:-1] + '\n\n')

if __name__ == '__main__':
    if use_window:
        print('window')
        train_window()
    else:
        print('non window')
        train_original()
