import re
import matplotlib.pyplot as plt
import numpy as np

f1 = 'bert-base-uncased.txt'
f2 = 'bert-base-multilingual-uncased.txt'
f3 = 'xlm-roberta-base.txt'

path = 'fullTedTalk_newUTT_triangleHalfTo1_weightedLoss_fractionLog_seed1'	# to be changed accordingly
data_path = '../../out/' + path + '/' + f1									# relative path from file

data_file = open(data_path, "r", encoding="utf-8")

loss = []
acc = []
f1 = []
while True:
	original_line = data_file.readline()
	
	if original_line =='':
		break

	if re.search('Val loss', original_line):
		line = original_line.split(' ')
		val_loss = float(line[4][:-1])
		val_acc = float(line[-1])
		loss.append(val_loss)
		acc.append(val_acc)
		print(val_loss,val_acc)

	if re.search('val_F1', original_line):
		line = original_line.split(' ')
		val_f1 = False
		for i in range(5):
			try:
				val_f1 = float(line[-i][:-2])
				if val_f1 != False:
					print('f1', val_f1)
					break
			except:
				continue
		f1.append(val_f1)
		

# plots validation loss, val accuracy and f1 scores at every epoch
plt.xlabel('epoch')
plt.ylabel('val_loss')
x = np.linspace(1,len(acc), num=len(acc))
plt.plot(x, loss, label = "val loss")
plt.show()
plt.plot(x, acc, label = "val acc")
plt.show()
plt.plot(x, f1, label = "f1")
plt.show()
