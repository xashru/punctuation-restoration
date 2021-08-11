import matplotlib.pyplot as plt

################ This section is just for adding your own file paths and opening of files ########################
# training files
en_file = open('../../data/en/train2012', "r", encoding="utf-8")
# utt_file = open('../../data/utt/train_utt', "r", encoding="utf-8")
# mini_file = open('../../data/en/test2011', "r", encoding="utf-8")
# mini_file = open('../../data/en/test2011asr', "r", encoding="utf-8")
# mini_file = open('../../data/LJ_Speech/train_LJ_Speech_20%', "r", encoding="utf-8")
# mini_file = open('../../data/utt_new_test/train_utt', "r", encoding="utf-8")
# sr_split_train = open('../../data/utt_new_test\sr_split\combined_train', "r", encoding="utf-8")
sr_restored_split_by_windowing_version2_train = open('../../data\sr_restored_split_by_windowing_version2/train', "r", encoding="utf-8")

# dev files
# dev_file = open('../../data/en/dev2012', "r", encoding="utf-8")
# mini_file = open('../../data/LJ_Speech/dev_LJ_Speech_10%', "r", encoding="utf-8")
# mini_file = open('../../data/utt_new_test/dev_utt', "r", encoding="utf-8")
# sr_split_dev = open('../../data/utt_new_test\sr_split\combined_dev', "r", encoding="utf-8")
sr_restored_split_by_windowing_version2_dev = open('../../data\sr_restored_split_by_windowing_version2/dev', "r", encoding="utf-8")

# test files
# file1 = open('../../data/test/test_LJ_Speech_70%', "r", encoding="utf-8")
# file2 = open('../../data/test/test_utt', "r", encoding="utf-8")
# file3 = open('../../data/test/test_2011', "r", encoding="utf-8")
# file4 = open('../../data/test/test_2011asr', "r", encoding="utf-8")
# file5 = open('../../data/test/train2012', "r", encoding="utf-8")
# file6 = open('../../data/utt_new_test/utt_test', "r", encoding="utf-8")
sr_restored_split_by_windowing_version2_test = open('../../data\sr_restored_split_by_windowing_version2/test', "r", encoding="utf-8")
################################################################################################

# add all the files you want to plot below
# e.g. if your train set has 2 separate files, open the file above and just add them in the list below accordingly
file_list = [sr_restored_split_by_windowing_version2_test]

def count_labels(data_file):
	corpus = data_file.readlines()
	corpus = list(filter(lambda x: x != '\n', corpus))
	for i in range(len(corpus)):
		if ('\t' not in corpus[i]):
			print(i)
	labels = list(map(lambda x: x.split('\t')[1].strip(), corpus))
	no_punc = len(list(filter(lambda x: x == 'O', labels)))
	comma = len(list(filter(lambda x: x == 'COMMA', labels)))
	question = len(list(filter(lambda x: x == 'QUESTION', labels)))
	period = len(list(filter(lambda x: x == 'PERIOD', labels)))
	return no_punc, comma, question, period
	
no_punc, comma, question, period = 0, 0, 0, 0

for i in file_list:
	values = count_labels(i)
	no_punc += values[0]
	comma += values[1]
	question += values[2]
	period += values[3]

x = ["O", "COMMA", "PERIOD", "QUESTION"]
y = [no_punc, comma, period, question]
bars = plt.bar(x, y)
plt.yscale("log")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + 0.15, yval + .6, yval)

plt.show()