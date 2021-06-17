import os
from sklearn.model_selection import train_test_split


data_file = open('../data/en/test2011', "r", encoding="utf-8")
train_file = open('../data/en/train_ted_talk_20%', 'w', encoding='utf8')
# test_file = open('../data/LJ_Speech/test_LJ_Speech_80%', 'w', encoding='utf8')
data_lines = []
process = True
while process:
	block = ""
	for i in range(1000):
		line = data_file.readline()
		if not line:
			if block != '':
				data_lines.append(block)
			process = False
			break
		block += line
	if process and block != '':
		data_lines.append(block)
# print(data_lines[-1])
test, train = train_test_split(data_lines, test_size=0.2, shuffle=True)

train_file.write("".join(train))
# test_file.write("".join(test))

data_file.close()
train_file.close()
# test_file.close()