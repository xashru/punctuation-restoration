# This file is used on sr restored train set (which includes dev segments) tagged with their segment headers
# It splits the set into the dev and train sets using the header information

dev_path = 'dev/text'														# stores dev segment information
restored_sr = 'train/text_restored_by_inference_version2_with_timing'		# restored full train + dev segments with headers

save_dev = open('../../data/sr_restored_split_by_inference_version2/dev', 'a')
save_train = open('../../data/sr_restored_split_by_inference_version2/train','a')

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

def get_dev_headers(file_path):
	'''
	takes in data file processed by sr which labels lines by their id (swXXXXX)
	returns the number of unique files in that data file
	'''
	data_file = open(file_path, "r", encoding="utf-8")
	lines = data_file.readlines()
	lines = list(map(lambda x: x.split(" ")[0], lines))
	lines = list(dict.fromkeys(lines))
	lines.sort()
	return set(lines)

dev_headers = get_dev_headers(dev_path)

data_file = open(restored_sr, "r", encoding="utf-8")
lines = data_file.readlines()
for l in lines:
	line = l.split(' ', 1)
	segment_header = line[0]
	segment_text = line[1]
	segment = segment_text.split(' ')
	word_list = list(map(convert_word, segment))
	if segment_header in dev_headers:
		save_dev.write('\n'.join(word_list))
		save_dev.write('\n')
	else:
		save_train.write('\n'.join(word_list))
		save_train.write('\n')

