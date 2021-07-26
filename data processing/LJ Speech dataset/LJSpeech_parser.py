import csv
import os
from sklearn.model_selection import train_test_split


def convert_word(text):
	'''
	converts individual words to inputs for model
	'''
	text = text.strip()

	# get last character
	last_char = text[-1]

	if last_char == ',':
		return text[:-1] + "\t" + "COMMA"
	elif last_char == '.':
		return text[:-1] + "\t" + "PERIOD"
	elif last_char == '?':
		return text[:-1] + "\t" + "QUESTION"
	else:
		return text + "\t" + "O"

# 1. convert to full corpus

# save_path = 'full_set'	
# corpus = open(save_path, "a", encoding="utf-8", errors='ignore')
# with open("LJSpeech-1.1\metadata.csv", "r", encoding="utf-8") as f:
# 	reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE)
# 	for i, line in enumerate(reader):
# 		corpus.write(line[1])
# 		corpus.write(' ')
# f.close()


# 2. tokenize file from step 1

# data_file = open('full_set', "r", encoding="utf-8")
# lines = data_file.readlines()
# word_list = lines[0].split(' ')
# word_list = list(map(convert_word, word_list))
# token_list = "\n".join(word_list)
# f = open('tokenized_full_set', "w", encoding="utf-8")
# f.write(token_list)


# 3. get train / test split
# samples every 1000 word block in the full corpus for the split

# data_file = open('../data/LJ_Speech/test_LJ_Speech_80%', "r", encoding="utf-8")
# train_file = open('../data/LJ_Speech/dev_LJ_Speech_10%', 'w', encoding='utf8')
# test_file = open('../data/LJ_Speech/test_LJ_Speech_70%', 'w', encoding='utf8')
# data_lines = []
# process = True
# while process:
# 	block = ""
# 	for i in range(1000):
# 		line = data_file.readline()
# 		if not line:
# 			if block != '':
# 				data_lines.append(block)
# 			process = False
# 			break
# 		block += line
# 	if process and block != '':
# 		data_lines.append(block)
		
# test, train = train_test_split(data_lines, test_size=0.09, shuffle=True)

# train_file.write("".join(train))
# test_file.write("".join(test))

# data_file.close()
# train_file.close()
# test_file.close()