import configparser
import os
import xml.dom.minidom
import codecs
import argparse
import multiprocessing
from os import listdir
from os.path import isfile, join
from collections import defaultdict

 
def synchronized_method(method):
	
	outer_lock = multiprocessing.Lock()
	lock_name = "__"+method.__name__+"_lock"+"__"
	
	def sync_method(self, *args, **kws):
		with outer_lock:
			if not hasattr(self, lock_name): setattr(self, lock_name, multiprocessing.Lock())
			lock = getattr(self, lock_name)
			with lock:
				return method(self, *args, **kws)  

	return sync_method


class Cache():

    def __init__(self, retriever, cache_size=1000):
        self.cache_keys = []
        self.cache = {}
        self.retriever = retriever
        self.cache_size = cache_size
        
    @synchronized_method  
    def get(self, key):
        if key in self.cache_keys:
            self.cache_keys.remove(key)
            self.cache_keys.append(key)
        else:
            self.cache[key] = self.retriever(key)
            self.cache_keys.append(key)

            if len(self.cache_keys) > self.cache_size:
                to_remove_key = self.cache_keys.pop(0)
                del self.cache[to_remove_key]
        return self.cache[key] 

def setup_dict_entry(_dict, entry, val):
	if entry not in _dict:
		_dict[entry] = val

def read_files(editions):
	res = {}
	for f in editions:
		res[f] = {}
		if os.path.exists(corpora_dir + '/' + f + ".txt"):
			with codecs.open(corpora_dir + '/' + f + ".txt", "r", "utf-8") as fi:
				for l in fi:
					if l[0] == "#":
						continue
					l = l.strip().split("\t")
					if len(l) != 2:
						continue
					res[f][l[0]] = l[1]
		else:
			print(f"file {corpora_dir + f}.txt not found")
	return res 

def get_alignment_files(dir):
    res = [f"{dir}/{f}" for f in listdir(dir) if isfile(join(dir, f))]
    return res

def read_config(f):
    global CES_alignment_files
    global CES_corpus_dir
    global parser
    global ParCourE_data_dir


    parser = configparser.ConfigParser()
    parser.read(f)

    CES_corpus_dir = parser['section']['CES_corpus_dir']
    CES_alignment_files = get_alignment_files(parser['section']['CES_alignment_files'])
    ParCourE_data_dir = parser['section']['ParCourE_data_dir']
    ParCourE_data_dir += "/parCourE"


def save_config(f):
    with open(f, 'w') as cfile:
        parser.write(cfile)

def create_dirs():
    global corpora_dir
    global config_dir
    
    config_dir = ParCourE_data_dir + "/config/"
    data_dir = ParCourE_data_dir + "/data/"
    corpora_dir = ParCourE_data_dir + "/data/corpora/"
    alignments_dir = ParCourE_data_dir + "/data/alignments/"
    aligns_index_dir = ParCourE_data_dir + "/data/align_index/"
    lexicon_dir = ParCourE_data_dir + "/data/lexicon/"
    elastic_dir = ParCourE_data_dir + "/data/elastic/"
    stats_dir = ParCourE_data_dir + "/data/stats/"

    if not os.path.exists(ParCourE_data_dir):
        os.makedirs(ParCourE_data_dir)
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if not os.path.exists(corpora_dir):
        os.mkdir(corpora_dir)
    if not os.path.exists(alignments_dir):
        os.mkdir(alignments_dir)
    if not os.path.exists(lexicon_dir):
        os.mkdir(lexicon_dir)
    if not os.path.exists(elastic_dir):
        os.mkdir(elastic_dir)
    if not os.path.exists(aligns_index_dir):
        os.mkdir(aligns_index_dir)
    if not os.path.exists(stats_dir):
        os.mkdir(stats_dir)
    if not os.path.exists(stats_dir + "/lang_pair_stats"):
        os.mkdir(stats_dir + "/lang_pair_stats")
    if not os.path.exists(stats_dir + "/edition_pair_stats"):
        os.mkdir(stats_dir + "/edition_pair_stats")
    if not os.path.exists(stats_dir + "/lang_stats"):
        os.mkdir(stats_dir + "/lang_stats")
    if not os.path.exists(stats_dir + "/edition_stats"):
        os.mkdir(stats_dir + "/edition_stats")
    

    parser['section']['config_dir'] = config_dir
    parser['section']['corpora_dir'] = corpora_dir
    parser['section']['alignments_dir'] = alignments_dir
    parser['section']['aligns_index_dir'] = aligns_index_dir
    parser['section']['lexicon_dir'] = lexicon_dir
    parser['section']['elastic_dir'] = elastic_dir
    parser['section']['stats_dir'] = stats_dir

def get_index(input_str, mode_nom):
    string = input_str[:10]

    x = ord(string[0]) << 7
    for chr in string[1:]:
        x = ((1000003 * x) ^ ord(chr)) & (1<<32)
    
    return x % mode_nom

def get_sentence_id(s_file, t_file, rel_string):
    global last_sentense_id
    global file_id_mapping

    dict_count = 10

    s_side = rel_string.split(";")[0].strip()
    t_side = rel_string.split(";")[1].strip()

    if s_file not in file_id_mapping:
        file_id_mapping[s_file] = [defaultdict(str) for i in range(dict_count)]
    if t_file not in file_id_mapping:
        file_id_mapping[t_file] = [defaultdict(str) for i in range(dict_count)]

    s_idx = get_index(s_side, dict_count)
    t_idx = get_index(t_side, dict_count)

    id = file_id_mapping[s_file][s_idx][s_side]

    if id != '':
        file_id_mapping[t_file][t_idx][t_side] = id
    else:
        id = file_id_mapping[t_file][t_idx][t_side]

    if id != '':
        file_id_mapping[s_file][s_idx][s_side] = id
    else:
        id = str(last_sentense_id)
        last_sentense_id += 1
        file_id_mapping[t_file][t_idx][t_side] = id
        file_id_mapping[s_file][s_idx][s_side] = id
    
    return id

PC_files = {}
def process_alignment_file(file):
    global PC_files
    print(f'parsing alignment file: {file}')
    doc = xml.dom.minidom.parse(file)
    align_grps = doc.getElementsByTagName("linkGrp")

    for align_grp in align_grps:
        s_lang = align_grp.getAttribute("fromDoc").split("/")[0]
        s_file = align_grp.getAttribute("fromDoc").split("/")[1]
        if s_file[-3:] == ".gz":
            s_file = s_file[:-3]

        t_lang = align_grp.getAttribute("toDoc").split("/")[0]
        t_file = align_grp.getAttribute("toDoc").split("/")[1]
        if t_file[-3:] == ".gz":
            t_file = t_file[:-3]

        aligns = align_grp.getElementsByTagName("link")

        extract_sentence_alignments(s_lang, f"/{s_lang}/{s_file}", t_lang, f"/{t_lang}/{t_file}", aligns)

    write_PC_format(s_lang, PC_files[s_lang])
    write_PC_format(t_lang, PC_files[t_lang])

def fix_file_name(file):
    file = file.split('/')
    for i in file:
        if i == '':
            continue
        file = i
        break

    if len(file) > 3:
        if file[-4] == '.':
            return file[:-4]
    
    return file


def get_CES_text(node):
    res = ''

    if node.nodeType == node.TEXT_NODE:
        res = node.data
    else:
        
        for ch_node in node.childNodes:
            text = get_CES_text(ch_node)
            if text != '':
                res += text + ' '

    return res.strip()

def compose_text(CES_ids, CES_file):
    res = ''
    for CES_id in CES_ids:
        try:
            res += CES_file[CES_id]
            res += ' '
        except Exception as e:
            print(e)

    res = res.strip()
    return  res

def create_res_sentences(PC_s_file, PC_t_file, sentence_id, CES_s_file, CES_t_file, rel_string):
    s_CES_ids = rel_string.split(";")[0].strip().split()
    t_CES_ids = rel_string.split(";")[1].strip().split()

    # if not sentence_id in PC_s_file:
    text = compose_text(s_CES_ids, CES_s_file)
    
    PC_s_file[sentence_id] = text

    # if not sentence_id in PC_t_file:
    text = compose_text(t_CES_ids, CES_t_file)
    PC_t_file[sentence_id] = text
            
def read_CES_senteces_file(file):
    print(f"reading CES file: {file}")
    res = {}
    nodes = xml.dom.minidom.parse(file).getElementsByTagName("s")
    for node in nodes:
        id = node.getAttribute("id")
        text = get_CES_text(node)
        if text[-2:] == ' .':
            text = text[:-2]+'.'
        res [id] = text

    return res
        
def write_PC_format(file, content):
    with codecs.open(corpora_dir + '/' + file + ".txt", "w", "utf-8") as fo:
        for id in content:
            fo.write(f"{id}\t{content[id]}\n")

def valid_alignment(align_string):
    splitted = align_string.split(";")
    if len(splitted) != 2 or splitted[0] == '' or splitted[1] == '':
        return False
    
    return True

def save_PC_config_files():
    global config_dir
    global lang_files
    global langs_order
    global file_edition_mapping
    global bert_100
    global prefixes

    with open(config_dir + "/lang_files.txt", 'w') as of:
        for item in lang_files:
            of.write(f"{item} {lang_files[item]}\n")
    
    with open(config_dir + "/languages_order_file.txt", 'w') as of:
        for item in langs_order:
            of.write(f"{item}\n")
    
    with open(config_dir + "/edition_file_mapping.txt", 'w') as of1, open(config_dir + "/file_edition_mapping.txt", 'w') as of2:
        for item in file_edition_mapping:
            of1.write(f"{file_edition_mapping[item]}\t{item}\n")
            of2.write(f"{item}\t{file_edition_mapping[item]}\n")
    
    with open(config_dir + "/bert_100.txt", 'w') as of:
        for item in bert_100:
            of.write(f"{item}\n")

    with open(config_dir + "/prefixes.txt", 'w') as of:
        for item in prefixes:
            of.write(f"{item} {prefixes[item]}\n")
    
    with open(config_dir + "/numversesplit.txt", 'w') as of:
        for item in prefixes:
            of.write(f"{prefixes[item]} {item}\n")

def add_to_PC_config_files(s_lang, s_edition, t_lang, t_edition):
    global lang_files
    global langs_order
    global file_edition_mapping
    global bert_100
    global prefixes
    global numversesplit

    lang_files[s_edition] = s_lang
    lang_files[t_edition] = t_lang

    langs_order.append(f"{s_lang},{t_lang}")

    file_edition_mapping[s_edition] = s_edition
    file_edition_mapping[t_edition] = t_edition


def extract_sentence_alignments(s_lang, s_file, t_lang, t_file, aligns):
    global ces_cache
    global PC_files

    s_edition = fix_file_name(s_file)
    t_edition = fix_file_name(t_file)

    # for CES format corpora we use the same name for edition and file !
    if s_edition not in PC_files:
        PC_files[s_edition] = read_files([s_edition])[s_edition]
    
    if t_edition not in PC_files:
        PC_files[t_edition] = read_files([t_edition])[t_edition]

    PC_s_file = PC_files[s_edition]
    PC_t_file = PC_files[t_edition]

    print(f"reading {CES_corpus_dir}/{s_file} and {CES_corpus_dir}/{t_file}")
    CES_s_file = ces_cache.get(CES_corpus_dir + "/" + s_file)
    CES_t_file = ces_cache.get(CES_corpus_dir + "/" + t_file)
    
    print(f"res size: {len(CES_s_file.keys())} { len(CES_t_file.keys())}")

    for align in aligns:
        align_string = align.getAttribute("xtargets")
        if valid_alignment(align_string):
            sentence_id = get_sentence_id(s_file, t_file, align_string)
            create_res_sentences(PC_s_file, PC_t_file, sentence_id, CES_s_file, CES_t_file, align_string)
    
    add_to_PC_config_files(s_lang, s_edition, t_lang, t_edition)


file_id_mapping = {}
last_sentense_id = 1
ces_cache = Cache(read_CES_senteces_file)     
lang_files = {}
langs_order = []
file_edition_mapping = {}
bert_100 = []
prefixes = {}
numversesplit = {}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create ParCourE format corpora from CES format.", 
	epilog="example: python -m convert_corpus_from_CES_format -c config.ini")

    parser.add_argument("-c", default="")


	
    args = parser.parse_args()
    if args.c == "":
        print("please specify config file")
        exit()

    read_config(args.c)
    create_dirs()
    save_config(args.c)


    for file in CES_alignment_files:
        process_alignment_file(file)
        
    save_PC_config_files()