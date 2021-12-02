import argparse
import os, codecs, pickle

ints = [f'{i}' for i in range(10)]
def is_data_file(f):
    if f[0] in ints and f[1] in ints:
        return True

    return False

def get_files(files_dir):
    all_files = os.listdir(files_dir)
    for f in all_files[:]:
        if not is_data_file(f):
            all_files.remove(f)
    
    return all_files

def prase_line(line):
    splitted = line.split("\t")
    if len(splitted) < 5 or splitted[4] == 'UNTRANSLATED':
        return None, None, None, None

    chapter_verse = splitted[0].split(':')
    chapter = chapter_verse[0][-3:]
    verse = chapter_verse[1]
    token_nom = splitted[1]
    token = splitted[4]

    return chapter, verse, token_nom, token

def add_up_tokens(tokens, all_verses, mappings):
    sentence = ' '.join([item[4].strip() for item in tokens])
    ind = 1
    if len(tokens) < 2:
        ind = 0
        print('Warning - zero length verse {tokens}')
        return
    all_verses.append(f'{tokens[ind][0]}{tokens[ind][1]}{tokens[ind][2]}\t{sentence}')

    token_noms = [item[3] for item in tokens]
    for i, nom in enumerate(token_noms):
        mappings[f'{tokens[ind][0]}{tokens[ind][1]}{tokens[ind][2]}{nom}'] = i

def parse_file(file_path, all_verses, mappings):
    verse_tokens = []
    prev_verse = None
    book_nom = file_name[:2]
    with codecs.open(file_path, 'r', 'utf8') as inf:
        for line in inf:
            chapter, verse, token_nom, token = prase_line(line.strip())

            if verse != prev_verse and len(verse_tokens) > 0:
                add_up_tokens(verse_tokens, all_verses, mappings)
                verse_tokens = []
            
            prev_verse = verse
            verse_tokens.append((book_nom, chapter, verse, token_nom, token))

def read_mappings_file(fpath):
    if os.path.exists(fpath):
        with open(fpath, 'rb') as inf:
            return pickle.load(inf)
    
    return {}

def write_mappings(fpath, mappings):
    with open(fpath, 'wb') as of:
        pickle.dump(mappings, of)

def write_list_to_file(fpath, senteces):
    with codecs.open(fpath, 'w', 'utf8') as of:
        for sent in senteces:
            of.write(f'{sent}\n')

def parse_aligned_to(aligned_to, verse_id, source_tok, mappings):
    res = ''
    all_toks = aligned_to.split()

    for tok in all_toks:
        if tok != '-' and '/' not in tok:
            sep = '-'
            if tok[0] == '(':
                tok = tok[1:-1]
                sep = 'p'
            elif '%' in tok:
                ind = tok.index('%') + 1
                tok = tok[ind:]
            if f'{verse_id}{tok}' not in mappings:
                print(f'Warning - not found mapping: {verse_id}{tok}')
                return None
            j = mappings[f'{verse_id}{tok}']
            res += f'{source_tok}-{j} '
    return res

def add_up_finnish_tokens(tokens, all_verses, mappings, alignments):
    sentence = ' '.join([item[4].strip().replace('␣','') for item in tokens])
    ind = 1
    
    aligns = ''
    aligned_tos = [item[3] for item in tokens]
    for i, aligned_to in enumerate(aligned_tos):
        align = parse_aligned_to(aligned_to, f'{tokens[ind][0]}{tokens[ind][1]}{tokens[ind][2]}', i, mappings)
        if align == None:
            aligns = ''
            break
        aligns += align
    if aligns != '':
        alignments.append(f"{tokens[ind][0]}{tokens[ind][1]}{tokens[ind][2]}\t{aligns[:-1]}")
        all_verses.append(f'{tokens[ind][0]}{tokens[ind][1]}{tokens[ind][2]}\t{sentence}')

def parse_finnish_file(fpath, all_verses, mappings, heb_alignments, grc_alignments):
    verse_tokens = []
    prev_verse = None
    book_nom = file_name[:2]

    alignments = grc_alignments
    if int(book_nom) < 40:
        alignments = heb_alignments

    with codecs.open(fpath, 'r', 'utf8') as inf:
        for line in inf:
            chapter, verse, aligned_to, token = prase_line(line.strip())
            if chapter == None:
                continue

            if verse != prev_verse and len(verse_tokens) > 0:
                add_up_finnish_tokens(verse_tokens, all_verses, mappings, alignments)
                verse_tokens = []
            elif prev_verse != None: #skip first line of each sentence for Finnish
                verse_tokens.append((book_nom, chapter, verse, aligned_to, token))
            prev_verse = verse

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Create ParCourE format corpora from CES format.", 
    #epilog="example: python -i 66rv.txt -a fin_grc_gold.txt -t fin-x-bible-helfi.txt")

    #parser.add_argument("-i", default="", help="input file name")
    
    #files_dir = "/mounts/Users/student/ayyoob/Dokumente/code/HELFI/Hebrew1008"
    files_dir = "/mounts/Users/student/ayyoob/Dokumente/code/HELFI/Greek1904"
    #files_dir = "/mounts/Users/student/ayyoob/Dokumente/code/HELFI/Finnish1938/fi-FABC-2020m3-HELFI-alignment"
    is_finish = False
    lang_code = 'grc'
    out_dir = "/mounts/Users/student/ayyoob/Dokumente/code/pbc-ui-demo/helfi"
    
    files = get_files(files_dir)
    #files = ['16-ne.txt']
    all_verses = []
    heb_alignments = []
    grc_alignments = []
    mappings = read_mappings_file(os.path.join(out_dir, "mappings.bin"))
    for file_name in files:
        print("processing file ", file_name)
        if is_finish:
            parse_finnish_file(os.path.join(files_dir, file_name), all_verses, mappings, heb_alignments, grc_alignments)
        else:
            parse_file(os.path.join(files_dir, file_name), all_verses, mappings)

    write_list_to_file(os.path.join(out_dir, f"{lang_code}-x-bible-helfi.txt"), all_verses)

    if is_finish:
        write_list_to_file(os.path.join(out_dir, f"helfi-heb-fin-gold-alignments.txt"), heb_alignments)
        write_list_to_file(os.path.join(out_dir, f"helfi-grc-fin-gold-alignments.txt"), grc_alignments)
    else:
        write_mappings(os.path.join(out_dir, "mappings.bin"), mappings)
    # TODO check the strange char at end of tokens
    # TODO check the empty sentence again
    # TODO run for HEBREW and Greek again