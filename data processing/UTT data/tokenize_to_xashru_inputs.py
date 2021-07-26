import os

def convert_word(text):
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

# reads in processed data (after process_utt_data.py) and processes them into model inputs

####################### use this to get all separated files ########################
save_path = 'utt_to_xashru_inputs_version2/'						

for root, dirs, files in os.walk("processed_data_version2"):
	for filename in files:
		save_to = save_path + root.split('\\')[1] + "/"
		#create folder if does not exist
		if not os.path.exists(save_to):
			os.makedirs(save_to)

		#open file
		f = open(save_to + filename, 'w')

		data_file = open(root + '/' + filename, "r")
		lines = data_file.readlines()
		word_list = lines[0].split(' ')
		word_list = list(map(convert_word, word_list))
		token_list = "\n".join(word_list)
		
		f.write(token_list)
		f.close()

######################## use this to combine into a single file ########################
# save_path = 'utt_to_xashru_inputs/single_utt_training_set'	

# # if not os.path.exists(save_path):
# # 	os.makedirs(save_path)
# f = open(save_path, 'a')

# for root, dirs, files in os.walk("processed_data"):
# 	for filename in files:	

# 		data_file = open(root + '/' + filename, "r")
# 		lines = data_file.readlines()
# 		word_list = lines[0].split(' ')
# 		word_list = list(map(convert_word, word_list))
# 		token_list = "\n".join(word_list)
# 		token_list += '\n'
# 		f.write(token_list)

# f.close()
