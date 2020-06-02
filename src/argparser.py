import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Punctuation restoration')
    parser.add_argument('--name', default='punctuation-restore', type=str, help='name of run')
    parser.add_argument('--cuda', default=True, type=lambda x: (str(x).lower() == 'true'), help='use cuda if available')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--pretrained-model', default='xlm-roberta-base', type=str,
                        help='pretrained model to use, available: xlm-roberta-base (default) and xlm-roberta-large')
    parser.add_argument('--freeze-bert', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Freeze BERT layer or not')
    parser.add_argument('--data-path', default='data/', type=str,
                        help='path to train/val/test datasets')
    parser.add_argument('--language', default='english', type=str,
                        help='language, available options are english, bangla, english-bangla')
    parser.add_argument('--sequence-length', default=200, type=int,
                        help='sequence length to use when preparing dataset (default 200)')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate (default: 16)')
    parser.add_argument('--decay', default=0, type=float, help='weight decay (default: 1e-5)')
    parser.add_argument('--gradient-clip', default=-1, type=float, help='gradient clipping (default: -1 or none)')
    parser.add_argument('--batch-size', default=16, type=int, help='batch size (default: 16)')
    parser.add_argument('--epoch', default=10, type=int, help='total epochs (default: 10)')
    parser.add_argument('--save-path', default='out/', type=str, help='model/log save directory')

    args = parser.parse_args()
    return args
