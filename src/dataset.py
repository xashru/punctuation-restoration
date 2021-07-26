import torch
from config import *
from augmentation import *
import numpy as np
from argparser import parse_arguments
import os

# default implementation unchanged
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
    :return: list of [tokens_index, punctuation_index, attention_masks, punctuation_mask, sequence_count], each having sequence_len
    punctuation_mask is used to ignore special indices like padding and intermediate sub-word token during evaluation
    sequence_count is an integer 1 which marks that the sequence is the last being processed as part of a contiguous corpus, 0 otherwise 
    """
    
    data_items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f.read().split('\n') if line.strip()]


    if (file_path.find('LJ_Speech') == -1):
        return sliding_window(lines, tokenizer, sequence_len, token_style, stride_size)

    # To be used when we want to apply sliding window to batches in file
    # In this project, only applicable to LJ_Speech dataset where we sample blocks of 1000 words throughout the full corpus (reference to 'data processing/LJ Speech dataset/LJSpeech_parser.py' when getting train/test split)
    # Any other dataset will undergo sliding window operation for the whole file, as conditional statement above
    # we can set WORD_BATCH accordingly in config.py should you sample fixed blocks of words as inputs
    # parse_data_by_block not necessary if no datasets require this processing, can just rely on sliding_window
    process_at_index = 0
    while (len(lines[process_at_index:]) > 0):
        data_items += sliding_window(lines[process_at_index:process_at_index + WORD_BATCH], tokenizer, sequence_len, token_style, stride_size)
        process_at_index += WORD_BATCH
    
    return data_items

def sliding_window(line_block, tokenizer, sequence_len, token_style, stride_size):
    '''
    :line_block: list of word inputs
    :return:  list of [tokens_index, punctuation_index, attention_masks, punctuation_mask, sequence_count], each having sequence_len
    sequence_count is an integer 1 which marks that the sequence is the last being processed as part of a contiguous corpus, 0 otherwise
    '''
    stride = stride_size
    sequence_size = sequence_len - 2                                                                                # account for start and end token
    stride_size_tokens = round(stride * sequence_size)                                                              # number of word tokens to take in half a sequence
    if stride > 0.5:
        raise Exception("Maximum stride length is 0.5!")
    elif stride_size_tokens <= 0:
        raise Exception("Stride size too small, please increase stride length!")

    data_items = []                                                                                                 # initialize list to return
    lines = line_block
    
    words = list(map(split_word, lines))
    words = list(map(lambda x: [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x[0].lower())), x[1]], words))   # tokenize word inputs
    words = list(map(lambda x: [x[0], [0] * (len(x[0]) - 1) + [punctuation_dict[x[1]]]], words))                    # get y list
    words = list(map(lambda x: [fill_unknown(x[0], token_style), x[1]], words))                                     # fill empty tokens
    words = list(map(lambda x: [x[0], x[1], [0] * (len(x[0]) - 1) + [1]], words))                                   # fill y_mask

    # obtain full list of tokens, their y and y_mask values for all inputs
    full_tokens = []
    full_y = []
    full_y_mask = []
    for tokens, y, y_mask in words:
        full_tokens += tokens
        full_y += y
        full_y_mask += y_mask
    
    process_at_index = 0
    
    while (len(full_tokens[process_at_index:]) >= sequence_size):                                                   # loop until no more tokens to process
        
        # initialize sequence with start token
        x = [TOKEN_IDX[token_style]['START_SEQ']]                                                                   
        y = [0]                                                                                                     
        y_mask = [1]           
        
        # add sequence until required length                                                                                      
        x += full_tokens[process_at_index:process_at_index + sequence_size]                                        
        y += full_y[process_at_index:process_at_index + sequence_size]
        y_mask += full_y_mask[process_at_index:process_at_index + sequence_size]
        x.append(TOKEN_IDX[token_style]['END_SEQ'])
        y.append(0)
        y_mask.append(1)

        attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]                             # initialize attn_mask for full sequence
        data_items.append([x, y, attn_mask, y_mask, 0])                                                             # add sequence

        process_at_index += stride_size_tokens                                                                      # advance by stride size(0.5)
    

    # account for remaining tokens after while loop
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

# below are 2 functions to parse the data such that the sequences do not truncate sentences: for reference only, discontinued after implementing sliding window
# F1 scores only slightly improved
###############################################################################################################################################################
###############################################################################################################################################################
# # modified from parse_data to fit full sentences into sequences
# # uses truncation on obtained sequences-> there might be a lot of unnecessary loops during token processing
###############################################################################################################################################################
# def parse_data_full_sentences_by_truncate(file_path, tokenizer, sequence_len, token_style):
#     '''
#     :return: list of [tokens_index, punctuation_index, attention_masks, punctuation_mask], each having sequence_len but does not cut parts of sentences
#     '''
#     data_items = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = [line for line in f.read().split('\n') if line.strip()]
#         idx = 0                             # index of word
        
#         # loop until end of the entire text
#         while idx < len(lines):
#             last_punc = idx                 # save current index
#             has_sentence = False
#             x_len = 0                       # initialize sequence window length
            
#             x = [TOKEN_IDX[token_style]['START_SEQ']]
#             y = [0]
#             y_mask = [1]                    # which positions we need to consider while evaluating i.e., ignore pad or sub tokens

#             # loop each word to check required sequence length
#             # -1 because we will have a special end of sequence token at the end
#             while len(x) < sequence_len - 1 and idx < len(lines):
#                 word, punc = lines[idx].split('\t')
#                 tokens = tokenizer.tokenize(word)
                
#                 # if taking these tokens exceeds sequence length we finish current sequence with padding
#                 # then start next sequence from this token
#                 if len(tokens) + len(x) >= sequence_len:
#                     break
#                 else:

#                     # heuristic: train model on full sentences, make sure sequences do not break up sentences
#                     # track last word ending with punctuation   
#                     if (punc == "PERIOD"):
#                         last_punc = idx         # track word index of last punctuation
#                         has_sentence = True
#                     elif (punc == "QUESTION"):
#                         last_punc = idx         # track word index of last punctuation
#                         has_sentence = True

#                     #convert word to token, y_mask is 0 except for last token
#                     for i in range(len(tokens) - 1):
#                         x.append(tokenizer.convert_tokens_to_ids(tokens[i]))
#                         y.append(0)
#                         y_mask.append(0)
#                     if len(tokens) > 0:
#                         x.append(tokenizer.convert_tokens_to_ids(tokens[-1]))
#                     else:
#                         x.append(TOKEN_IDX[token_style]['UNK'])
                        
#                     y.append(punctuation_dict[punc])
#                     y_mask.append(1)
#                     idx += 1

#                     # save sequence length to truncate sentence after loop
#                     if (idx - 1 == last_punc and has_sentence):
#                         x_len = len(x)

#             if has_sentence == True:
#                 idx = last_punc + 1
#                 x = x[:x_len]
#                 y = y[:x_len]
#                 y_mask = y_mask[:x_len]

#             x.append(TOKEN_IDX[token_style]['END_SEQ'])
#             y.append(0)
#             y_mask.append(1)

#             if len(x) < sequence_len:
#                 x = x + [TOKEN_IDX[token_style]['PAD'] for _ in range(sequence_len - len(x))]
#                 y = y + [0 for _ in range(sequence_len - len(y))]
#                 y_mask = y_mask + [0 for _ in range(sequence_len - len(y_mask))]
#             attn_mask = [1 if token != TOKEN_IDX[token_style]['PAD'] else 0 for token in x]
#             data_items.append([x, y, attn_mask, y_mask])
            
#     return data_items

###############################################################################################################################################################
# # splits data into full sentences and fit them into sequences
# # same goal as above function but to improve efficiency however might be more prone to implementation bugs
###############################################################################################################################################################
# def parse_data_full_sentences(file_path, tokenizer, sequence_len, token_style):
#     '''
#     :return: list of [tokens_index, punctuation_index, attention_masks, punctuation_mask], each having sequence_len but does not cut parts of sentences
#     '''
#     data_items = []
#     with open(file_path, 'r', encoding='utf-8') as f:
#         lines = [line for line in f.read().split('\n') if line.strip()]
    
#     words = list(map(split_word, lines))
#     words = list(map(lambda x: [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(x[0])), x[1]], words))
#     words = list(map(lambda x: [fill_unknown(x[0], token_style), x[1]], words))

#     ends_of_sentences = list(map(find_end_of_sentence, enumerate(words)))   # map indexes of ends of sentences
#     ends_of_sentences = list(filter(lambda x: x != -1, ends_of_sentences))  # filter out irrelevant indexes

#     # get start of sentence index and its number of tokens
#     start_and_token_count = {}
#     start_index = 0
#     for i in ends_of_sentences:
#         idx = i
#         sentence = words[start_index:idx + 1]
#         start_and_token_count[start_index] = (idx + 1, count_tokens(sentence))
#         start_index = idx + 1
#     if (count_tokens(words[start_index:]) != 0):
#         start_and_token_count[start_index] = (-1, count_tokens(words[start_index:]))

#     x, y, y_mask = initialize_seq(token_style)
#     for key, value in start_and_token_count.items():
        
#         end = value[0]                                              # update last index of sentence
#         if end == -1:
#             end = len(words)
        
#         if (value[1] + len(x) >= sequence_len):                     # current seq + sentence >= than sequence length
#             if (len(x) == 1):                                       # if seq have no tokens then process anyways
#                 for i in range(key, end):                           # loop and send seq data until sequence length
#                     tokens = words[i][0]
#                     if (len(x) + len(tokens) >= sequence_len):       # if exceed, send and initialize new
#                         x, y, attn_mask, y_mask = end_sequence(x, y, y_mask, token_style, sequence_len)
#                         data_items.append([x, y, attn_mask, y_mask])
#                         x, y, y_mask = initialize_seq(token_style) 
#                     #continue processing token    
#                     x += tokens
#                     y += [0 for i in range(len(tokens) - 1)]
#                     y.append(punctuation_dict[words[i][1]])
#                     y_mask += [0 for i in range(len(tokens) - 1)]
#                     y_mask.append(1)

#             else:   # exceed but there are tokens then send data first then process
#                 x, y, attn_mask, y_mask = end_sequence(x, y, y_mask, token_style, sequence_len)
#                 data_items.append([x, y, attn_mask, y_mask])
#                 x, y, y_mask = initialize_seq(token_style)
#                 for i in range(key, end):
#                     tokens = words[i][0]
#                     if (len(x) + len(tokens) >= sequence_len):       # if exceed, send and initialize new
#                         x, y, attn_mask, y_mask = end_sequence(x, y, y_mask, token_style, sequence_len)
#                         data_items.append([x, y, attn_mask, y_mask])
#                         x, y, y_mask = initialize_seq(token_style) 
#                     #continue processing token  
#                     x += tokens
#                     y += [0 for i in range(len(tokens) - 1)]
#                     y.append(punctuation_dict[words[i][1]])
#                     y_mask += [0 for i in range(len(tokens) - 1)]
#                     y_mask.append(1)
#         else:   # process sentence
#             for i in range(key, end):
#                 tokens = words[i][0]
#                 x += tokens
#                 y += [0 for i in range(len(tokens) - 1)]
#                 y.append(punctuation_dict[words[i][1]])
#                 y_mask += [0 for i in range(len(tokens) - 1)]
#                 y_mask.append(1)
#     x, y, attn_mask, y_mask = end_sequence(x, y, y_mask, token_style, sequence_len)
#     data_items.append([x, y, attn_mask, y_mask])
#     return data_items
###############################################################################################################################################################
###############################################################################################################################################################