import re
import matplotlib.pyplot as plt
import numpy as np

f1 = 'bert-base-uncased.txt'
f2 = 'bert-base-multilingual-uncased.txt'
f3 = 'xlm-roberta-base.txt'
path = 'fullTedTalk_newUTT_triangleHalfTo1_seed1'
data_path = '../../out/' + path + '/' + f2

data_file = open(data_path, "r", encoding="utf-8")

loss = []
acc = []
while True:
	line = data_file.readline()
	
	if line =='':
		break

	if re.search('Val loss', line):
		line = line.split(' ')
		val_loss = float(line[4][:-1])
		val_acc = float(line[-1])
		loss.append(val_loss)
		acc.append(val_acc)
		print(val_loss,val_acc)

plt.xlabel('epoch')
plt.ylabel('val_acc')
x = np.linspace(1,len(acc), num=len(acc))
plt.plot(x, acc, label = "val acc")
plt.show()