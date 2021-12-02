import sys
sys.path.insert(0, '../')
from my_utils import align_utils as autils, utils
import argparse
# set random seed
config_file = "/mounts/Users/student/ayyoob/Dokumente/code/pbc-ui-demo/config_pbc.ini"
utils.setup(config_file)

params = argparse.Namespace()
params.gold_file = "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/eng_fra_pbc/eng-fra-common_verses.gold"
params.editions_file =  "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/eng_fra_pbc/lang_list.txt"


pros, surs = autils.load_gold(params.gold_file)
all_verses = list(pros.keys())
editions, langs = autils.load_simalign_editions(params.editions_file)
current_editions = [editions[lang] for lang in langs]

first_half = all_verses[:int(len(all_verses)/2)]
second_half = all_verses[int(len(all_verses)/2):]
first_half_train = first_half[:int((len(first_half)/10)*9)]
first_half_valid = first_half[int((len(first_half)/10)*9):]
second_half_train = second_half[:int((len(second_half)/10)*9)]
second_half_valid = second_half[int((len(second_half)/10)*9):]

for verse in all_verses:
    verse_aligns = autils.get_verse_alignments(verse)
    print(verse, len(verse_aligns))
    break