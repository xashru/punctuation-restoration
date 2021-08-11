# This file is used to facilitate restoring sr by inference

# step 1: get full text from sr without headers to be restored by inference
# train_path = 'train/text'
# save_file = 'train/text_for_inference'
# data_file = open(train_path, 'r')
# lines = data_file.readlines()
# lines = list(map(lambda x: x.split(' ', 1)[-1].strip(), lines))

# new_file = open(save_file, 'w')
# new_file.write(' '.join(lines))

# step 2: restore header information after inference (full sr train set including dev segments) for matching with segments
original_file = open('train/text', 'r')
inference_file = open('train/text_restored_by_inference_version2', 'r')
save_file = open('train/text_restored_by_inference_version2_with_timing', 'w')

inference_lines = inference_file.readline()
inference_lines = inference_lines.split(' ')

current_pointer = 0
line = original_file.readline()
while line != '':
	header_text = line.split(' ', 1)
	header = header_text[0]
	text = header_text[1]
	len_of_text = len(text.split(' '))
	save_file.write(header + ' ')
	save_file.write(' '.join(inference_lines[current_pointer: current_pointer + len_of_text]))
	save_file.write('\n')
	current_pointer += len_of_text
	line = original_file.readline()


