import os
import re

# This is used to get full text from stm file (sr test) without headers to be restored by inference

test_file = 'Switchboard_data_split_for_sentence_unit_detection\stm'
save_file = 'Switchboard_data_split_for_sentence_unit_detection\stm_for_inference'

data_file = open(test_file, 'r')
lines = data_file.readlines()
lines = list(map(lambda x: x.split(' ', 5)[-1].strip(), lines))

new_file = open(save_file, 'w')
new_file.write(' '.join(lines))