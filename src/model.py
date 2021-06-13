import torch.nn as nn
import torch
from config import *
from torchcrf import CRF


class DeepPunctuation(nn.Module):
    def __init__(self, pretrained_model, freeze_bert=False, lstm_dim=-1):
        super(DeepPunctuation, self).__init__()
        self.output_dim = len(punctuation_dict)
        self.bert_layer = MODELS[pretrained_model][0].from_pretrained(pretrained_model)
        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        bert_dim = MODELS[pretrained_model][2]
        if lstm_dim == -1:
            hidden_size = bert_dim
        else:
            hidden_size = lstm_dim
        self.lstm = nn.LSTM(input_size=bert_dim, hidden_size=hidden_size, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_features=hidden_size*2, out_features=len(punctuation_dict))

    def forward(self, x, attn_masks):
        if len(x.shape) == 1:
            x = x.view(1, x.shape[0])  # add dummy batch for single sample
        # (B, N, E) -> (B, N, E)
        x = self.bert_layer(x, attention_mask=attn_masks)[0]
        # (B, N, E) -> (N, B, E)
        x = torch.transpose(x, 0, 1)
        x, (_, _) = self.lstm(x)
        # (N, B, E) -> (B, N, E)
        x = torch.transpose(x, 0, 1)
        x = self.linear(x)
        return x
