import os
import torch
import torch.nn as nn
import numpy as np
import torch.multiprocessing

from argparser import parse_arguments
from dataset import Dataset
from model import DeepPunctuation
from config import *

torch.multiprocessing.set_sharing_strategy('file_system')  # https://github.com/pytorch/pytorch/issues/11201

args = parse_arguments()

# for reproducibility
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(args.seed)

# tokenizer
tokenizer = MODELS[args.pretrained_model][1].from_pretrained(args.pretrained_model)
token_style = MODELS[args.pretrained_model][3]

test_files = os.listdir(os.path.join(args.data_path, 'test'))
test_set = []
for file in test_files:
    test_set.append(Dataset(os.path.join(args.data_path, 'test', file), tokenizer, args.sequence_length, token_style,
                            0, False))

# Data Loaders
data_loader_params = {
    'batch_size': args.batch_size,
    'shuffle': False,
    'num_workers': 0
}

test_loaders = [torch.utils.data.DataLoader(x, **data_loader_params) for x in test_set]

# logs
model_save_path = os.path.join(args.save_path, 'weights.pt')
log_path = os.path.join(args.save_path, args.name + '_logs_test.txt')

# Model
device = torch.device('cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu')
deep_punctuation = DeepPunctuation(args.pretrained_model, args.freeze_bert)
deep_punctuation.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(deep_punctuation.parameters(), lr=args.lr, weight_decay=args.decay)


def test(data_loader):
    """
    :return: precision[numpy array], recall[numpy array], f1 score [numpy array], accuracy
    """
    num_iteration = 0
    deep_punctuation.eval()
    # +1 for overall result
    tp = np.zeros(1 + len(punctuation_dict), dtype=np.int)
    fp = np.zeros(1 + len(punctuation_dict), dtype=np.int)
    fn = np.zeros(1 + len(punctuation_dict), dtype=np.int)
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for x, y, att in data_loader:
            x, y, att = x.to(device), y.to(device), att.to(device)
            y_predict = deep_punctuation(x, att)
            y = y.view(-1)
            y_predict = y_predict.view(-1, y_predict.shape[2])
            loss = criterion(y_predict, y)
            test_loss += loss.item()
            num_iteration += 1
            correct += torch.sum(torch.argmax(y_predict, dim=1) == y).item()
            total += y.shape[0]
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
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, correct / total


def run():
    deep_punctuation.load_state_dict(torch.load(model_save_path))
    for i in range(len(test_loaders)):
        precision, recall, f1, accuracy = test(test_loaders[i])
        log = test_files[i] + '\n' + 'Precision: ' + str(precision) + '\n' + 'Recall: ' + str(recall) + '\n' + \
            'F1 score: ' + str(f1) + '\n' + 'Accuracy:' + str(accuracy) + '\n'
        print(log)
        with open(log_path, 'a') as f:
            f.write(log)


run()
