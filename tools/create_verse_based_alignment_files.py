import os
from app import utils
from app.general_align_reader import GeneralAlignReader
from datetime import datetime

align_reader = GeneralAlignReader()


def check_files_existance(files):
    for f in files:
        if not os.path.exists(f):
            print(f"warning could not find file: {f}")
            return False
    return True

def process_language_pair_verses(a_file, i_file, verse_file_contents): 
    ###
    # receives an alignment and an index file and update the verse_file_contents structure
    # a_file: alignment_file
    # i_file: alignments index file
    ###

    print(f"going to process {a_file}, {i_file}")
    
    if check_files_existance([a_file, i_file]):
        with open(a_file, 'r') as afl, open(i_file, 'r') as ifl:
            a_lines = afl.readlines()
            i_lines = ifl.readlines()

            for tup in zip(a_lines, i_lines):
                aligns = tup[0].split('\t')[1]
                verse, s_edit, t_edit = tuple(tup[1].strip().split())
                utils.setup_dict_entry(verse_file_contents, verse, "")
                verse_file_contents[verse] += f"{s_edit}\t{t_edit}\t{aligns.strip()}\n"
    
def persist_verse_contents(verse_file_contents):
    print(f'{datetime.now().strftime("%H:%M:%S")} - going to persist {len(verse_file_contents)} verses')
    for verse in verse_file_contents:
        with open(f"{utils.AlignInduction.verse_alignments_path}/{verse}.txt", 'a') as f:
            f.write(verse_file_contents[verse])

def process_batch_of_language_pairs(lang_pairs):
    ###
    # for a batch of language pair update ver_file_contents dictionary and finally flushes the 
    # dictionary to files
    ###

    print(f'{datetime.now().strftime("%H:%M:%S")} - processing a batch of language pairs with size of {len(lang_pairs)}')
    verse_file_contents = {}
    for pair in lang_pairs:
        s_l, t_l = align_reader.get_ordered_langs(pair[0], pair[1])
        a_file = align_reader.get_align_file_path(s_l, t_l)
        i_file = align_reader.get_index_file_path(s_l, t_l)
        process_language_pair_verses(a_file, i_file, verse_file_contents)

    persist_verse_contents(verse_file_contents)
    print(f'{datetime.now().strftime("%H:%M:%S")} - finished processing a batch of language pairs with size of {len(lang_pairs)}')


if __name__ == "__main__":
    utils.setup(os.environ['CONFIG_PATH'])
    _, langs = utils.read_lang_file_mapping()

    all_pairs = [[]]
    counter = 0
    for i,l1 in enumerate(langs):
        for l2 in langs[i:]:
            all_pairs[-1].append([l1, l2])
            counter += 1
            if counter % 1000 == 0 :
                all_pairs.append([])

    print(f"processing {len(all_pairs)} pair batches of languages")
    
    for i, pair_batch in enumerate(all_pairs):
        process_batch_of_language_pairs(pair_batch)