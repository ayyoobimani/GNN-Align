# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch, sys
sys.path.insert(0, '../')
from my_utils import gpu_utils
import importlib
import gc
from my_utils import align_utils as autils, utils
from my_utils.alignment_features import *
from tqdm import tqdm
run = 0
from tqdm import tqdm
from gnn_utils import eval_utils


# %%
# !pip install torch-geometric
# !pip install tensorboardX

# !wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
# !unzip ngrok-stable-linux-amd64.zip

#  print(torch.version.cuda)
#  print(torch.__version__)    

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
free_gpu1 = '6'
free_gpu2 = '5'


# %%

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import time
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch.optim as optim

from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

import torch_geometric.transforms as T

from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt



# %%
import argparse
from multiprocessing import Pool

# set random seed
config_file = "/mounts/Users/student/ayyoob/Dokumente/code/pbc-ui-demo/config_pbc.ini"
utils.setup(config_file)

params = argparse.Namespace()
#params.editions_file =  "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/eng_fra_pbc/lang_list.txt"

#params.gold_file = "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/eng_fra_pbc/eng-fra.gold"
#params.gold_file = "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi/splits/helfi-heb-fin-gold-alignments_test.txt"

params.gold_file = "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi/splits/helfi-grc-fin-gold-alignments_train.txt"
pros, surs = autils.load_gold(params.gold_file)
all_verses = list(pros.keys())
params.gold_file = "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi/splits/helfi-heb-fin-gold-alignments_train.txt"
pros, surs = autils.load_gold(params.gold_file)
all_verses.extend(list(pros.keys()))
all_verses = list(set(all_verses))
print(len(all_verses))

params.editions_file =  "/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/helfi/splits/helfi_lang_list.txt"
editions, langs = autils.load_simalign_editions(params.editions_file)
current_editions = [editions[lang] for lang in langs]

def get_pruned_verse_alignments(args):
    verse, current_editions = args
    print(verse)
    verse_aligns_inter = autils.get_verse_alignments(verse)
    verse_aligns_gdfa = autils.get_verse_alignments(verse, gdfa=True)

    autils.prune_non_necessary_alignments(verse_aligns_inter, current_editions)
    autils.prune_non_necessary_alignments(verse_aligns_gdfa, current_editions)

    gc.collect()
    return verse_aligns_inter, verse_aligns_gdfa
    

#verse_alignments_inter = {}
#verse_alignments_gdfa = {}
#args = []
#for i,verse in enumerate(all_verses):
#    args.append((verse, current_editions[:]))

#print('going to get alignments')
#with Pool(80) as p:
#    all_res = p.map(get_pruned_verse_alignments, args)

#for i,verse in enumerate(all_verses):
#    verse_aligns_inter, verse_aligns_gdfa = all_res[i]
    
#    verse_alignments_inter[verse] = verse_aligns_inter
#    verse_alignments_gdfa[verse] = verse_aligns_gdfa

#for verse in all_verses[:]:
#    if len(verse_alignments_inter[verse].keys()) < 10:
#        all_verses.remove(verse)

#torch.save(all_verses, 'all_verses.pickle')

utils.LOG.info("done")


# %%
import pickle
import gnn_utils.graph_utils as gutil
importlib.reload(gutil)
sys.setrecursionlimit(100000)

train_verses = all_verses[:]
test_verses = all_verses[:]
editf1 = "eng-x-bible-mixed"
editf2 = 'fra-x-bible-louissegond'

small_editions = current_editions[:]
#if editf1 not in small_editions:
#    small_editions.append[editf1]
#if editf2 not in small_editions:
#    small_editions.append(editf2)

if 'jpn-x-bible-newworld' in small_editions:
    small_editions.remove('jpn-x-bible-newworld')
if 'grc-x-bible-unaccented' in small_editions:
    small_editions.remove('grc-x-bible-unaccented')


#train_dataset = torch.load("/mounts/work/ayyoob/models/gnn/dataset_eng_fra_full_community.pickle")
train_dataset, train_nodes_map = gutil.create_dataset(train_verses, verse_alignments_inter, small_editions)
torch.save(train_dataset, "/mounts/work/ayyoob/models/gnn/dataset_helfi_train_community.pickle")
features = train_dataset.features
train_nodes_map = train_dataset.nodes_map
#edge_index_intra_sent = train_dataset.edge_index_intra_sent
#test_edge_index_intra_sent = edge_index_intra_sent

# test_dataset, test_nodes_map = create_dataset(test_verses, verse_alignments_inter, small_editions)
test_dataset, test_nodes_map = train_dataset, train_nodes_map
test_verses = train_verses
print(train_dataset.x.shape, train_dataset.edge_index.shape, len(train_dataset.features))

# augment_features(test_dataset)

# x_edge, features_edge = create_edge_attribs(train_nodes_map, train_verses, small_editions, verse_alignments_inter, train_dataset.x.shape[0])
# with open("./dataset.pickle", 'wb') as of:
#     pickle.dump(train_dataset, of)


# %%
#run on delta, extract w2v features
sys.path.insert(0, '../')
import pickle
from gensim.models import Word2Vec
from app import document_retrieval
from my_utils import utils
config_file = "/mounts/Users/student/ayyoob/Dokumente/code/pbc-ui-demo/config_pbc.ini"
utils.setup(config_file)
import torch
import my_utils.alignment_features as feat_utils
importlib.reload(document_retrieval)

doc_retriever = document_retrieval.DocumentRetriever()

#model_w2v = Word2Vec.load("/mounts/work/ayyoob/models/w2v/word2vec_helfi_langs_15e.model")
train_dataset = torch.load("/mounts/work/ayyoob/models/gnn/dataset_helfi_grc_test_community.pickle")
nodes_map = train_dataset.nodes_map

x = [[] for i in range(train_dataset.x.shape[0])]
for edition_f in nodes_map:
    utils.LOG.info(f"processing edition {edition_f}")
    for verse in nodes_map[edition_f]:         #toknom nodecount
        line = doc_retriever.retrieve_document(f'{verse}@{edition_f}')
        line = line.strip().split()

        for tok in nodes_map[edition_f][verse]:
            w_emb = model_w2v.wv.key_to_index[f'{edition_f[:3]}:{line[tok]}']
            x[nodes_map[edition_f][verse][tok]].extend([w_emb])

x = torch.tensor(x, dtype=torch.float)
train_dataset.x = torch.cat((train_dataset.x[:, :-100], x), dim=1)
train_dataset.features.pop()
train_dataset.features.append(feat_utils.MappingFeature(100, 'word'))

print(x.shape, train_dataset.x.shape, len(train_dataset.features))

torch.save(train_dataset, "/mounts/work/ayyoob/models/gnn/dataset_helfi_grc_test_community_word.pickle")
print('done adding w2v features')

# %%
train_dataset = torch.load("/mounts/work/ayyoob/models/gnn/dataset_helfi_grc_test_community.pickle")

# %%
features = train_dataset.features[:]
from pprint import pprint

print(train_dataset.x.shape)
for i in features:
    print(vars(i))

print(train_dataset.nodes_map['eng-x-bible-mixed'].keys())
# %%
nodes_map = train_dataset.nodes_map
bad_edition_files = []
for edit in nodes_map:
    bad_count = 0
    for verse in nodes_map[edit]:
        if len(nodes_map[edit][verse].keys()) < 6:
            bad_count += 1
        if bad_count > 1:
            bad_edition_files.append(edit)
            break
print(bad_edition_files)


## %%
#all_japanese_nodes = set()
#nodes_map = train_dataset.nodes_map

#for verse in nodes_map['jpn-x-bible-newworld']:
#    for item in nodes_map['jpn-x-bible-newworld'][verse].items():
#        all_japanese_nodes.add(item[1])

#print(" all japansese nodes: ", len(all_japanese_nodes))
#edge_index = train_dataset.edge_index.to('cpu')
#remaining_edges_index = []
#for i in tqdm(range(0, edge_index.shape[1], 2)):
#    if edge_index[0, i].item() not in all_japanese_nodes and edge_index[0, i+1].item() not in all_japanese_nodes:
#        remaining_edges_index.extend([i, i+1])

#print('original total edges count', edge_index.shape)
#print('remaining edge count', len(remaining_edges_index))
#train_dataset.edge_index = edge_index[:, remaining_edges_index]
#train_dataset.edge_index.shape

