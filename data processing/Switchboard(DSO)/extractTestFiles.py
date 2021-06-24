import os
import numpy as np

save_path = 'Switchboard_data_split_for_sentence_unit_detection/test_HL.ctm'	
data_file = open(save_path, "r", encoding="utf-8")
lines = data_file.readlines()
iterable = map(lambda x: x.split(" ")[0], lines)
test_files = []
for i in iterable:
	if i not in test_files:
		test_files.append(i)
print(len(test_files))

test_utt = 'new_utt_inputs/utt_test'	
train_utt = 'new_utt_inputs/train_utt'	
dev = 'new_utt_inputs/dev_utt'	
# if not os.path.exists(test_utt):
# 	os.makedirs(test_utt)
# if not os.path.exists(non_test_utt):
# 	os.makedirs(non_test_utt)
# f1 = open(test_utt, 'a')
f2 = open(train_utt, 'a')
f3 = open(dev, 'a')

sample_train = 0.3

for root, dirs, files in os.walk("../UTT data/utt_to_xashru_inputs"):

	for filename in files:
		data_file = open(root + '/' + filename, "r")
		lines = data_file.read()
		filename = filename.split('.')[0]
		if filename in test_files:
			# f1.write(lines)
			# f1.write("\n")
			continue
		else:
			r = np.random.rand()
			if r < sample_train:
				f2.write(lines)
				f2.write("\n")
			else:
				f3.write(lines)
				f3.write("\n")
f3.close()
f2.close()