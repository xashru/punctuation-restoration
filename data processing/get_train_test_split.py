import os
from sklearn.model_selection import train_test_split

f = open('data/en/train2012', 'r', encoding='utf8')
train_file = open('data/en/train_ted_talk_20%', 'w', encoding='utf8')
# val_file = open('data/utt/dev_utt', 'w')
# test_file = open('data/utt/test_utt', 'w')
lines = f.read().splitlines()
# train, test = train_test_split(lines, test_size=0.1, shuffle=False)
train, val = train_test_split(lines, test_size=0.2, shuffle=False)

train_file.write("\n".join(val))
# val_file.write("\n".join(val))
# test_file.write("\n".join(test))

f.close()
train_file.close()
# val_file.close()
# test_file.close()