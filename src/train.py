import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
import torch.multiprocessing
from tqdm import tqdm

from argparser import parse_arguments
from dataset import Dataset
from model import DeepPunctuation
from config import *


torch.multiprocessing.set_sharing_strategy('file_system')   # https://github.com/pytorch/pytorch/issues/11201

args = parse_arguments()

# for reproducibility
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# tokenizer
tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)
token_style = MODELS[args.pretrained_model][3]
ar = args.augment_rate
sequence_len = args.sequence_length

# Datasets
if args.language == 'english':
    train_set = Dataset(os.path.join(args.data_path, 'train2012'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar)
    val_set = Dataset(os.path.join(args.data_path, 'dev2012'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False)
    test_set_ref = Dataset(os.path.join(args.data_path, 'test2011'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set_asr = Dataset(os.path.join(args.data_path, 'test2011asr'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set = [val_set, test_set_ref, test_set_asr]
elif args.language == 'bangla':
    train_set = Dataset(os.path.join(args.data_path, 'train_bn'), tokenizer=tokenizer, sequence_len=sequence_len,
                        token_style=token_style, is_train=True, augment_rate=ar)
    val_set = Dataset(os.path.join(args.data_path, 'dev_bn'), tokenizer=tokenizer, sequence_len=sequence_len,
                      token_style=token_style, is_train=False)
    test_set_news = Dataset(os.path.join(args.data_path, 'test_bn_news'), tokenizer=tokenizer, sequence_len=sequence_len,
                            token_style=token_style, is_train=False)
    test_set_ted = Dataset(os.path.join(args.data_path, 'test_bn_ted'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set = [val_set, test_set_news, test_set_ted]
elif args.language == 'english-bangla':
    train_set = Dataset([os.path.join(args.data_path, 'train2012'), os.path.join(args.data_path, 'train_bn')],
                        tokenizer=tokenizer, sequence_len=sequence_len, token_style=token_style, is_train=True, augment_rate=ar)
    val_set = Dataset([os.path.join(args.data_path, 'dev2012'), os.path.join(args.data_path, 'dev_bn')],
                      tokenizer=tokenizer, sequence_len=sequence_len, token_style=token_style, is_train=False)
    test_set_ref = Dataset(os.path.join(args.data_path, 'test2011'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set_asr = Dataset(os.path.join(args.data_path, 'test2011asr'), tokenizer=tokenizer, sequence_len=sequence_len,
                           token_style=token_style, is_train=False)
    test_set_bn = Dataset(os.path.join(args.data_path, 'test_bn_news'), tokenizer=tokenizer, sequence_len=sequence_len,
                          token_style=token_style, is_train=False)
    test_set_bn_ted = Dataset(os.path.join(args.data_path, 'test_bn_ted'), tokenizer=tokenizer, sequence_len=sequence_len,
                              token_style=token_style, is_train=False)
    test_set = [val_set, test_set_ref, test_set_asr, test_set_bn, test_set_bn_ted]
else:
    raise ValueError('Incorrect language argument for Dataset')

# Data Loaders
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': True,
    'num_workers': 4
}
train_loader = torch.utils.data.DataLoader(train_set, **data_loader_params)
val_loader = torch.utils.data.DataLoader(val_set, **data_loader_params)
test_loaders = [torch.utils.data.DataLoader(x, **data_loader_params) for x in test_set]

# logs
model_save_path = os.path.join(args.save_path, 'weights.pt')
log_path = os.path.join(args.save_path, args.name + '_logs.txt')

# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
deep_punctuation = DeepPunctuation(args.pretrained_model, args.freeze_bert)
deep_punctuation.to(device)
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
        for x, y, att in tqdm(data_loader, desc='eval'):
            x, y, att = x.to(device), y.to(device), att.to(device)
            y_predict = deep_punctuation(x, att)
            y = y.view(-1)
            y_predict = y_predict.view(-1, y_predict.shape[2])
            loss = criterion(y_predict, y)
            val_loss += loss.item()
            num_iteration += 1
            att = att.view(-1)
            correct += torch.sum(att * (torch.argmax(y_predict, dim=1) == y).long()).item()
            total += torch.sum(att).item()
    return correct/total, val_loss/num_iteration


def test(data_loader):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy
    """
    num_iteration = 0
    deep_punctuation.eval()
    # +1 for overall result
    tp = np.zeros(1+len(punctuation_dict), dtype=np.int)
    fp = np.zeros(1+len(punctuation_dict), dtype=np.int)
    fn = np.zeros(1+len(punctuation_dict), dtype=np.int)
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for x, y, att in tqdm(data_loader, desc='test'):
            x, y, att = x.to(device), y.to(device), att.to(device)
            y_predict = deep_punctuation(x, att)
            y = y.view(-1)
            y_predict = y_predict.view(-1, y_predict.shape[2])
            loss = criterion(y_predict, y)
            test_loss += loss.item()
            num_iteration += 1
            att = att.view(-1)
            correct += torch.sum(att * (torch.argmax(y_predict, dim=1) == y).long()).item()
            total += torch.sum(att).item()
            for i in range(y.shape[0]):
                cor = y[i]
                prd = torch.argmax(y_predict[i])
                if cor == prd:
                    tp[cor] += 1
                else:
                    fn[cor] += 1
                    fp[prd] += 1
    # ignore first index which is for no punctuation
    tp[-1] = np.sum(tp[1:])
    fp[-1] = np.sum(fp[1:])
    fn[-1] = np.sum(fn[1:])
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, correct/total


def train():
    with open(log_path, 'a') as f:
        f.write(str(args)+'\n')
    best_val_acc = 0
    for epoch in range(args.epoch):
        train_loss = 0.0
        train_iteration = 0
        correct = 0
        total = 0
        deep_punctuation.train()
        for x, y, att in tqdm(train_loader, desc='train'):
            x, y, att = x.to(device), y.to(device), att.to(device)
            y_predict = deep_punctuation(x, att)

            # flat labels and predictions for calculating loss
            y = y.view(-1)
            y_predict = y_predict.view(-1, y_predict.shape[2])

            optimizer.zero_grad()
            loss = criterion(y_predict, y)
            train_loss += loss.item()
            train_iteration += 1

            loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(deep_punctuation.parameters(), 5)
            optimizer.step()
            att = att.view(-1)
            correct += torch.sum(att * (torch.argmax(y_predict, dim=1) == y).long()).item()
            total += torch.sum(att).item()
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
        # scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(deep_punctuation.state_dict(), model_save_path)

    print('Best validation Acc:', best_val_acc)
    deep_punctuation.load_state_dict(torch.load(model_save_path))
    for loader in test_loaders:
        precision, recall, f1, accuracy = test(loader)
        log = 'Precision: ' + str(precision) + '\n' + 'Recall: ' + str(recall) + '\n' + 'F1 score: ' + str(f1) + \
              '\n' + 'Accuracy:' + str(accuracy) + '\n'
        print(log)
        with open(log_path, 'a') as f:
            f.write(log)
        log_text = ''
        for i in range(1, 5):
            log_text += str(precision[i] * 100) + ' ' + str(recall[i] * 100) + ' ' + str(f1[i] * 100) + ' '
        with open(log_path, 'a') as f:
            f.write(log_text[:-1] + '\n\n')


train()
