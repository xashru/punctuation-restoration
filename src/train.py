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
elif args.language == 'mini-english':
    train_set = Dataset([os.path.join(args.data_path, 'utt/train_utt'), os.path.join(args.data_path, 'en/train_ted_talk_20%')], tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type, is_sliding_window=use_window, stride_size=stride_size)
    val_set = Dataset(os.path.join(args.data_path, 'en/dev2012'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_ref = Dataset(os.path.join(args.data_path, 'en/test2011'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_asr = Dataset(os.path.join(args.data_path, 'en/test2011asr'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set = [val_set, test_set_ref, test_set_asr]
elif args.language == 'mixed-LJ-speech':
    train_set = Dataset([os.path.join(args.data_path, 'utt/train_utt'), os.path.join(args.data_path, 'LJ_Speech/non_fiction_20%')], tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar, augment_type=aug_type, is_sliding_window=use_window, stride_size=stride_size)
    val_set = Dataset(os.path.join(args.data_path, 'LJ_Speech/non_fiction_80%'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_ref = Dataset(os.path.join(args.data_path, 'en/test2011'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set_asr = Dataset(os.path.join(args.data_path, 'en/test2011asr'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    test_set = [val_set, test_set_ref, test_set_asr]
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
    # val_set = Dataset(os.path.join(args.data_path, 'en/dev2012 copy'), tokenizer=tokenizer, sequence_len=sequence_len,
    #                   token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    # test_set_asr = Dataset(os.path.join(args.data_path, 'en/test2011asr copy'), tokenizer=tokenizer, sequence_len=sequence_len,
    #                        token_style=token_style, is_train=False, is_sliding_window=use_window, stride_size=stride_size)
    # test_set = [val_set, test_set_asr]
else:
    raise ValueError('Incorrect language argument for Dataset')

# Data Loaders
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 1
}
train_loader = torch.utils.data.DataLoader(train_set, **data_loader_params)
val_loader = torch.utils.data.DataLoader(val_set, **data_loader_params)
test_loaders = [torch.utils.data.DataLoader(x, **data_loader_params) for x in test_set]

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


def validate(data_loader):
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
            
            # reduce end weights of sequences
            if use_window:
                x_values = np.linspace(-3, 3, y_predict.shape[1])                   # get x values uniformly 
                pdf_values = norm.pdf(x_values)                                     # get pdf values of a normal distribution
                pdf_values = pdf_values.reshape(-1,1)            
                min_max_scaler = preprocessing.MinMaxScaler((0.2,1))                # get weights to multiply to tensor between 0.2 to 1
                bellcurve_weights = min_max_scaler.fit_transform(pdf_values)
                bellcurve_weights = torch.from_numpy(bellcurve_weights).to(device)
                y_predict = y_predict * bellcurve_weights

            y_predict = y_predict.view(-1, y_predict.shape[2])
            loss = criterion(y_predict, y)
            y_predict = torch.argmax(y_predict, dim=1).view(-1)

            val_loss += loss.item()
            num_iteration += 1
            y_mask = y_mask.view(-1)
            correct += torch.sum(y_mask * (y_predict == y).long()).item()
            total += torch.sum(y_mask).item()
    return correct/total, val_loss/num_iteration
    
def test(data_loader):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy, confusion matrix
    """
    num_iteration = 0
    deep_punctuation.eval()
    # initialize true positive, false positive arrays... for [O, COMMA, PERIOD, QUESTION, OVERALL]
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
            y_predict = deep_punctuation(x, att)                                    #make prediction using model
            y = y.view(-1)

            # reduce end weights of sequences
            if use_window:
                x_values = np.linspace(-3, 3, y_predict.shape[1])                   # get x values uniformly 
                pdf_values = norm.pdf(x_values)                                     # get pdf values of a normal distribution
                pdf_values = pdf_values.reshape(-1,1)        
                min_max_scaler = preprocessing.MinMaxScaler((0.2,1))                # get weights to multiply to tensor between 0.2 to 1
                bellcurve_weights = min_max_scaler.fit_transform(pdf_values)
                bellcurve_weights = torch.from_numpy(bellcurve_weights).to(device)
                y_predict = y_predict * bellcurve_weights

            y_predict = y_predict.view(-1, y_predict.shape[2])
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
    # print(tp, fp, fn)
    precision = tp/(tp+fp)
    # print(precision)
    recall = tp/(tp+fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, correct/total, cm

def train():
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
            print("\nx:", x)
            x, y, att, y_mask = x.to(device), y.to(device), att.to(device), y_mask.to(device)
            y_mask = y_mask.view(-1)
            y_predict = deep_punctuation(x, att)
            # print("\ny predict:", y_predict)

            # reduce end weights of sequences
            if use_window:
                x_values = np.linspace(-3, 3, y_predict.shape[1])                   # get x values uniformly 
                # print('\nbellcurve x values:\n', x_values)
                pdf_values = norm.pdf(x_values)                                     # get pdf values of a normal distribution
                pdf_values = pdf_values.reshape(-1,1)       
                # print('\nbellcurve y values:\n', pdf_values)
                min_max_scaler = preprocessing.MinMaxScaler((0.2,1))                # get weights to multiply to tensor between 0.2 to 1
                bellcurve_weights = min_max_scaler.fit_transform(pdf_values)
                bellcurve_weights = torch.from_numpy(bellcurve_weights).to(device)
                # print('\nbellcurve y values normalized:\n', bellcurve_weights)
                y_predict = y_predict * bellcurve_weights
                # print('\ny_predict:\n', y_predict)
            y_predict = y_predict.view(-1, y_predict.shape[2])
            y = y.view(-1)

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

        val_acc, val_loss = validate(val_loader)
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
        precision, recall, f1, accuracy, cm = test(loader)
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
    train()
