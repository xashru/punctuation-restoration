# This file is used on full continuous sr restored test set to get model inputs

def convert_word(text):
	t = text.strip()
	if t == '':
		return ''

	# get first punctuation
	for i in range(len(t)):
		if t[i] == ',':
			return t[:i] + "\t" + "COMMA"
		elif t[i]== '.':
			return t[:i] + "\t" + "PERIOD"
		elif t[i] == '?':
			return t[:i] + "\t" + "QUESTION"

	return t + "\t" + "O"

restored_sr_test = 'Switchboard_data_split_for_sentence_unit_detection/stm_restored_by_inference_version2'	
save_test = open('../../data/sr_restored_split_by_inference_version2/test', 'a')
data_file = open(restored_sr_test, "r", encoding="utf-8")
lines = data_file.readlines()
for l in lines:
	segment = l.split(' ')
	word_list = list(map(convert_word, segment))
	save_test.write('\n'.join(word_list))
	save_test.write('\n')