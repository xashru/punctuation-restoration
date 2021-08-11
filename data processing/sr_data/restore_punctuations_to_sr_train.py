import os
import re
import spacy

# This file is used to restore punctuations to sr train and dev set using the bag of words approach
# it is the same approach as restore_punctuations_to_sr_test.py however since the dev segments are within the train segments,
# there is slightly more processing to be done with header information to match the corresponding segments,
# it might be better to look at restore_punctuations_to_sr_test.py to understand the code first

def remove_special_characters(text):
    t = re.sub('{\w ', '', text)	# this is the version2 step
    t = re.sub('{', '', t)
    t = re.sub('}', '', t)
    t = re.sub('\*\[\[.*\]\]', '', t)
    t = re.sub('\[', '', t)
    t = re.sub('\]', '', t)
    t = re.sub('\(', '', t)
    t = re.sub('\)', '', t)
    t = re.sub('/', '', t)
    t = re.sub('\+', '', t)
    t = re.sub('#', '', t)
    return t

def reduce_to_single_space(text):
    return ' '.join(text.split())

def remove_space_before_punctuation(text):
    return re.sub(r'\s([?.!,])', r'\1', text)

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
	relevant_lines = list(map(lambda x: x.split(" ", 1)[1].lower().strip(), relevant_lines))
	
	return relevant_lines

def get_all_headers_with(file_path, target_title):
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
	relevant_lines = list(map(lambda x: x.split(" ", 1)[0].strip(), relevant_lines))
	return relevant_lines

def remove_punc_from_word(word):
	if len(word) < 1:
		return word
	elif word[-1] == ',' or word[-1] == '?' or word[-1] == '.':
		return word[:-1]
	return word

def get_punc(word):
	if len(word) < 1:
		return ''
	elif word[-1] == ',' or word[-1] == '?' or word[-1] == '.':
		return word[-1]
	return ''


def get_nearest_words(index, original_word_list):
	'''
	returns the index of target word, followed by nearest 13 words in a tuple
	'''
	if index > 5 and index < len(original_word_list) - 6:
		return 6, tuple(original_word_list[index - 6: index + 7])
	elif index <= 5:
		return index, tuple(original_word_list[:13])
	elif index >= len(original_word_list) - 6:
		return 13 - (len(original_word_list) - index), tuple(original_word_list[-13:]	)
	raise Exception('improper handling of window')

def handle_small_utterance(word_list):
	words = list(map(lambda x: 'utterance1' if x == 'uh-huh' or x == 'um-hum' or x == 'uh-hum' else x, word_list))
	words = list(map(lambda x:'utterance2' if x=='uh' or x == 'hm' or x=='hum' else x, words))
	return words

def calculate_score(position_in_window, current_window, words_in_original):
	'''
	:current_window: 13 word block from sr data
	:words_in_original: 13 word block in original UTT file
	:position_in_window: position of word in words_in_original and the target word in current_window
	'''
	# account for utterance
	current_window = handle_small_utterance(current_window)
	words_in_original = handle_small_utterance(words_in_original)

	score = 0
	for i in range(len(current_window)):																# for every word in sr data
		if i == position_in_window and current_window[i] == words_in_original[i]:						# target word is the same in both windows
			word_score = 1
		elif i < position_in_window and current_window[i] in words_in_original[:position_in_window]:	# word is left of target word
			if current_window[i] in words_in_original[i-1:i+2]:											# word is approximately close
				word_score = 0.9
			else:
				word_score = 0.8
		elif i > position_in_window and current_window[i] in words_in_original[position_in_window + 1:]:# word is right of target word
			if current_window[i] in words_in_original[i-1:i+2]:											# word is approximately close
				word_score = 0.9
			else:
				word_score = 0.8
		else:																							# word is not found
			word_score = 0
		score += word_score																				# accumulate word score
	return score
	
def sliding_window_estimate(words_list_sr_data, original_word_list):
	'''
	:words_list_sr_data: segment of sr_data without punctuations
	:original_word_list: Corresponding segments of original data with punctuations
	:return: words_list_sr_data restored with punctuations
	'''

	#############################################################
	# key: (position of target word, tuple of 13 nearest words)
	# value: score, window position of sr data, punctuation
	shape_score = {}	
	#############################################################

	# loop through original word list and save every punctuation position in shape_score	
	for i in range(len(original_word_list)):
		current_word = original_word_list[i]
		current_punc = get_punc(current_word)
		if current_punc == '':
			continue
		position_index, nearest_words = get_nearest_words(i, list(map(remove_punc_from_word, original_word_list))) 
		shape_score[(position_index, nearest_words)] = [0, 0, current_punc]
	
	# for every 13 word block in words_list_sr_data, calculate score for every saved punctuation to find best position in sr_data 
	for i in range(len(words_list_sr_data) - 13):
		current_window = words_list_sr_data[i: i+13]
		for key in shape_score.keys():
			position_in_window = key[0]
			words_in_original = key[1]
			score = calculate_score(position_in_window, current_window, words_in_original)
			if score > shape_score[key][0]:
				shape_score[key][0] = score
				shape_score[key][1] = i
	
	# keep track of restored words
	word_positions_score = {} 

	# restore punctuations to sr data, only restoring the highest score at each word
	restored_sr_data = words_list_sr_data
	for key, value in shape_score.items():
		sr_window_index = value[1]
		word_in_window_index = key[0]

		punctuation_to_position_score = value[0]
		word_position_in_sr = sr_window_index + word_in_window_index
		if word_position_in_sr in word_positions_score:
			if punctuation_to_position_score > word_positions_score[word_position_in_sr]:
				word_positions_score[word_position_in_sr] = punctuation_to_position_score
				restored_sr_data[word_position_in_sr] = restored_sr_data[word_position_in_sr][:-1] + value[2]
			continue
		else:
			word_positions_score[word_position_in_sr] = punctuation_to_position_score
			restored_sr_data[word_position_in_sr] = restored_sr_data[word_position_in_sr] + value[2]
	return restored_sr_data

def get_restored_segments(headers, sr_segments, sr_segments_with_punc):
	'''
	:headers: sr segments headers separated by lines
	:sr_segments: corresponding sr segments separated by lines
	:sr_segments_with_punc: corresponding whole conversation of continuous sr segments
	:return: list of sr segments with punctuations
	'''
	length_of_segments = map(lambda x: len(x.split(' ')), sr_segments)
	lines = []
	current_pointer = 0
	for index, length in enumerate(length_of_segments):
		line = headers[index]
		line += ' '
		line += ' '.join(sr_segments_with_punc[current_pointer: current_pointer + length])
		current_pointer += length
		lines.append(line)
	return lines

def get_speaker_lines(filename):
	'''
	returns speaker A and B conversations as a continuous string lowercased respectively from original data
	'''
	original_file = open(root + '/' + filename, "r")
	lines = original_file.readlines()
	a_lines = []
	b_lines = []
	for l in lines:
		if re.search('A\.\d*\sutt\d*:', l):
			txt = l.split(":")[1]
			txt = txt.strip()
			txt = remove_special_characters(txt)
			txt = reduce_to_single_space(txt)
			txt = remove_space_before_punctuation(txt)
			if txt != '':
				a_lines.append(txt)
		elif re.search('B\.\d*\sutt\d*:', l):
			txt = l.split(":")[1]
			txt = txt.strip()
			txt = remove_special_characters(txt)
			txt = reduce_to_single_space(txt)
			txt = remove_space_before_punctuation(txt)
			if txt != '':
				b_lines.append(txt)
	return " ".join(a_lines).lower(), " ".join(b_lines).lower()

train_path = 'train/text'
dev_path = 'dev/text'
sr_restored_train_path = 'train/text_restored_by_windowing_version2'

train_set = get_full_set(train_path)
dev_set = get_full_set(dev_path)

utt_path = '../UTT data/original_utt_data'

for root, dirs, files in os.walk(utt_path):															# loop through every file in original utt data
	for filename in files:	
		formatted_filename = re.sub('_\d{4}_', '0', filename)
		if formatted_filename[:-4] in train_set:													# Handle if file is in train set
			segment_1_headers = get_all_headers_with(train_path, formatted_filename[:-4] + '-A')	# get speaker A headers from sr data
			segment_2_headers = get_all_headers_with(train_path, formatted_filename[:-4] + '-B')	# get speaker B headers from sr data
			train_segment_1 = get_all_lines_with(train_path, formatted_filename[:-4] + '-A')		# get speaker A lines from sr data
			train_segment_2 = get_all_lines_with(train_path, formatted_filename[:-4] + '-B')		# get speaker B lines from sr data
	
			# get list of words in segments
			words_in_segment_1 = []
			for index, segment in enumerate(train_segment_1):
				words_in_segment_1 += segment.split(' ')
			words_in_segment_2 = []
			for index, segment in enumerate(train_segment_2):
				words_in_segment_2 += segment.split(' ')

			##### get corresponding speaker A and B from original to match with above as speaker labels may be swapped #####
			#################################################################################################################
			a_lines, b_lines = get_speaker_lines(filename)
			a_list = a_lines.split(' ')
			b_list = b_lines.split(' ')
			a_words_only = list(map(lambda x: remove_punc_from_word(x), a_list))
			b_words_only = list(map(lambda x: remove_punc_from_word(x), b_list))

			segment_1_a_count = 0
			segment_1_b_count = 0
			for i in words_in_segment_1:
				if i in a_words_only:
					a_words_only.remove(i)
					segment_1_a_count += 1
				if i in b_words_only:
					b_words_only.remove(i)
					segment_1_b_count += 1
			#################################################################################################################
			
			# add buffer to sr data for better processing start and ends of file
			pad = ['*', '*', '*', '*', '*', '*']
			words_in_segment_1 = pad + words_in_segment_1 + pad
			words_in_segment_2 = pad + words_in_segment_2 + pad

			if segment_1_a_count > segment_1_b_count:
				# segment 1 corresponds to a_segments in original
				segment_1_words = sliding_window_estimate(words_in_segment_1, a_list)
				segment_2_words = sliding_window_estimate(words_in_segment_2, b_list)
			else:
				# segment 1 corresponds to b_segments in original
				segment_1_words = sliding_window_estimate(words_in_segment_1, b_list)
				segment_2_words = sliding_window_estimate(words_in_segment_2, a_list)
			
			# remove pads
			segment_1_words = segment_1_words[6:]
			segment_1_words = segment_1_words[:-6]
			segment_2_words = segment_2_words[6:]
			segment_2_words = segment_2_words[:-6]

			# get list of sr segments from the whole conversation
			restored_segments_1 = get_restored_segments(segment_1_headers, train_segment_1, segment_1_words)
			restored_segments_2 = get_restored_segments(segment_2_headers, train_segment_2, segment_2_words)

			# save data
			sr_restored = open(sr_restored_train_path, "a", encoding="utf-8")
			for i in restored_segments_1:
				sr_restored.write(i)
				sr_restored.write('\n')
			for j in restored_segments_2:
				sr_restored.write(j)
				sr_restored.write('\n')

#sw02073 is found to have incorrectly labeled A and B