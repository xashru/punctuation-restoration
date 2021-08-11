import os
import numpy as np

# save inputs to data file
test_utt = '../../data/utt_version2/test'	
train_utt = '../../data/utt_version2/train'	
dev = '../../data/utt_version2/dev'	

f1 = open(test_utt, 'a')
f2 = open(train_utt, 'a')
f3 = open(dev, 'a')

# sample probability
sample_train = 0.3
sample_dev = 0.5

# loop through all files which are already in input form
for root, dirs, files in os.walk("utt_to_xashru_inputs_version2"):
	for filename in files:
		data_file = open(root + '/' + filename, "r")
		lines = data_file.read()
		r = np.random.rand()
		if r > sample_train:
			f2.write(lines)
			f2.write("\n")
		else:
			r2 = np.random.rand()
			if r2 > sample_dev:
				f1.write(lines)
				f1.write('\n')
			else:
				f3.write(lines)
				f3.write("\n")
f3.close()
f2.close()
f1.close()