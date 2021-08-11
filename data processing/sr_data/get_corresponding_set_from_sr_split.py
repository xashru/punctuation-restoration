import os
import re

# This file is used to get the train_dev_split in sr_data (645K train, 75K dev). 
# It is also the sr_split_on_dataset1 in the data file

def get_full_set(file_path):
	'''
	takes in data file processed by sr which labels lines by their id (swXXXXX)
	returns the number of unique files in that data file
	'''
	data_file = open(file_path, "r", encoding="utf-8")
	lines = data_file.readlines()
	lines = list(map(lambda x: x.split(" ")[0][:7], lines))
	lines = list(dict.fromkeys(lines))
	lines.sort()
	return lines

def get_all_lines_with(file_path, target_title):
	'''
	to be used on sr dataset, train and dev set, to extract relevant lines 
	:param file_path: file path to check every line
	:target_title: target string to filter every line which contains it
	:return: all lines that have the target string, stripped and lower cased
	'''
	data_file = open(file_path, "r", encoding="utf-8")
	lines = data_file.readlines()
	relevant_lines = []
	relevant_lines = list(filter(lambda x: True if re.search(target_title, x) else False, lines))
	relevant_lines = list(map(lambda x: re.sub('-', '', x), relevant_lines))	# remove some notations for better processing
	relevant_lines = list(map(lambda x:re.sub('\[', '', x), relevant_lines))
	relevant_lines = list(map(lambda x:re.sub('\]', '', x), relevant_lines))
	relevant_lines = list(map(lambda x: x.split(" ", 1)[1].lower().strip(), relevant_lines))
	
	return relevant_lines

def count_chars(substring, text):
	'''
	counts the number of characters (in order) within substring that is contained in text.
	'''
	char_count = 0
	full_text = text
	current_len = len(full_text)
	for c in substring:
		try:
			index = full_text.index(c)
		except:
			index = -1
		if index != -1:
			full_text = full_text[index + 1:]
		if (len(full_text) < current_len):
			current_len = len(full_text)
			char_count += 1
	return char_count

test_path = '..\sr_data_test_files\Switchboard_data_split_for_sentence_unit_detection\stm'	
train_path = 'train/text'
dev_path = 'dev/text'
new_train_save_path = 'train_dev_split/train'
new_dev_save_path = 'train_dev_split/dev'

train_set = get_full_set(train_path)
dev_set = get_full_set(dev_path)
test_set = get_full_set(test_path)

#calculate number of utt files
full_utt_filenames = []
utt_path = '../UTT data/processed_data'

# loop through every UTT file and handles the file is file is in dev set
for root, dirs, files in os.walk(utt_path):
	for filename in files:	
		full_utt_filenames.append(filename[:-4])
		if filename[:-4] in dev_set and filename[:-4] == 'sw02006':		# second condition for debugging and seeing output on single file
			dev_segments = get_all_lines_with(dev_path, filename[:-4])	# get all dev segment lines
			# print(*dev_segments, sep='\n')							# print to see example of dev segments
			
			# create a dictionary to store each segment: [target length, words in segment, word matches, start index], target length is slightly increased (10%) to account for difference in processing
			dev_segments_dict = {}
			for index, segment in enumerate(dev_segments):
				words_in_segment = segment.split(' ')
				target_segment_size = len(words_in_segment)
				add_buffer = len(words_in_segment)//10			
				if  add_buffer >= 1:
					target_segment_size += add_buffer
				else:
					target_segment_size += 1
				dev_segments_dict[index] = [target_segment_size, words_in_segment, 0, 0]

			# print(dev_segments_dict)									# print to see example of stored dictionary
			
			processed_file = open(os.path.join(root, filename), "r", encoding="utf-8")
			processed_text = processed_file.read().lower().split(' ')

			# for every start index, check if potential segments increases matching character count and update start index for segment

			# first check on full segment
			for i in range(len(processed_text)):
				for key,values in dev_segments_dict.items():
					potential_segment = ' '.join(processed_text[i : i+values[0]])
					count_matching = count_chars(''.join(values[1]), potential_segment)

					if count_matching > values[2]:
						dev_segments_dict[key][2] = count_matching
						dev_segments_dict[key][3] = i

			# second check but count characters at a word level for corner cases to improve accuracy
			for i in range(len(processed_text)):
				for key,values in dev_segments_dict.items():
					potential_segment = ' '.join(processed_text[i : i+values[0]])
					count_matching = 0
					for word in values[1]:
						if word in potential_segment:
							potential_segment = potential_segment.replace(word, ' ', 1)
							count_matching += len(word)
					if count_matching >= values[2]:
						dev_segments_dict[key][2] = count_matching
						dev_segments_dict[key][3] = i

			dev_segments_dict = dict(sorted(dev_segments_dict.items(), key=lambda item: item[1][2], reverse=True))

			############### print segment to see outcome ###############
			# for i in dev_segments_dict.keys():
			# 	print(" ".join(dev_segments_dict[i][1]), '\n', " ".join(processed_text[dev_segments_dict[i][-1]: dev_segments_dict[i][-1] + dev_segments_dict[i][0]]))
			# 	print('\n')
			############################################################

			# now that we have the dev segments, we can save the dev segments and train segments accordingly
			###################################################################################################
			# new_train_file = open(os.path.join(new_train_save_path, filename[:-4]), "w", encoding="utf-8")
			# new_dev_file = open(os.path.join(new_dev_save_path, filename[:-4]), "w", encoding="utf-8")
			# current_index_of_processed_text = 0

			# full_corpus = ' '.join(processed_text)						# get UTT file as full corpus
			# for index, values in dev_segments_dict.items():				# loop through every dev segment
			# 	length = values[0]
			# 	start_index_of_segment = values[3]
			# 	get_segment = ' '.join(processed_text[start_index_of_segment : start_index_of_segment + length])
			# 	full_corpus = full_corpus.replace(get_segment, '', 1)	# extract out dev segments to save
			# 	new_dev_file.write(get_segment)							
			# 	new_dev_file.write('\n')
			# new_train_file.write(full_corpus)							# save remaining file as train 
			###################################################################################################
			
