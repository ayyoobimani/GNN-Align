from numpy import dtype
from numpy.core.fromnumeric import shape
import torch, sys, os, signal
import networkx as nx, numpy as np
from torch.nn.functional import embedding
sys.path.insert(0, '../')
from my_utils import utils, align_utils as autils
from my_utils.alignment_features import *
import my_utils.alignment_features as afeatures
import torch_geometric.utils as pyg_utils
from torch_geometric.data import Data
import math, tqdm
from random import  randint, choice
from karateclub import Node2Vec, AE, Role2Vec
from scipy.sparse import csr, coo_matrix
from networkx.algorithms.community import greedy_modularity_communities, label_propagation_communities, asyn_fluidc
from multiprocessing import Pool, Manager



#########################################################
##################### dataset creation #####################
#########################################################
def node_nom(verse, editf, tok_nom, node_count, nodes_map, x=None, edit_fs=None, features = None):
    utils.setup_dict_entry(nodes_map, editf, {})
    utils.setup_dict_entry(nodes_map[editf], verse, {})
    if not tok_nom in nodes_map[editf][verse]:
        nodes_map[editf][verse][tok_nom] = node_count
        x.append([edit_fs.index(editf), tok_nom]) # TODO we should have better representation 
        if len(features) == 0:
            features.extend([OneHotFeature(20, len(edit_fs), 'edit_f'), OneHotFeature(32, 150, 'position')])
        if tok_nom > 150:
            print('sequence len: ', tok_nom)
        # , all_verses.index(verse)/len(all_verses)
        # x.append([1])
        node_count += 1

    return nodes_map[editf][verse][tok_nom], node_count

def get_negative_edges(verses, edition_files, nodes_map,  alignments):
    neg_edges = [[],[]]
    node_count = 0 # not need here
    for verse in verses:
        for i,editf1 in enumerate(edition_files):
            for j,editf2 in enumerate(edition_files[i+1:]):
                
                lent2 = len(nodes_map[editf2][verse])
                lent1 = len(nodes_map[editf1][verse])
                if lent1 < 2 or lent2 < 2:
                    continue

                aligns = autils.get_aligns(editf1, editf2, alignments[verse])   
                if aligns != None:
                    for align in aligns:
                        
                        # for ii in range(20):
                        idx2 = randint(align[1]+1, align[1] + lent2 -1 ) % lent2
                        n1, node_count = node_nom(verse, editf1, align[0], node_count, nodes_map)
                        n2 = nodes_map[editf2][verse][list(nodes_map[editf2][verse].keys())[idx2]]


                        idx1 = randint(align[0]+1, align[0] + lent1 - 1) % lent1
                        n1_ = nodes_map[editf1][verse][list(nodes_map[editf1][verse].keys())[idx1]]
                        n2_, node_count = node_nom(verse, editf2, align[1], node_count, nodes_map)

                        neg_edges[0].extend([n1, n2, n1_, n2_])
                        neg_edges[1].extend([n2, n1, n2_, n1_])
    return torch.tensor(neg_edges, dtype=torch.long)

def create_dataset(verses, alignments, edition_files):
    node_count = 0
    edges = [[],[]]
    x = []
    nodes_map = {}
    features = []
    args = []
    padding = 0
    utils.LOG.info(f"adding verses")
    for verse in verses:
        x_tmp = []
        edges_tmp = [[],[]]
        for i,editf1 in enumerate(edition_files):
            for j,editf2 in enumerate(edition_files[i+1:]):
                aligns = autils.get_aligns(editf1, editf2, alignments[verse])
                if aligns != None:
                    for align in aligns:
                        n1, node_count = node_nom(verse, editf1, align[0], node_count, nodes_map, x_tmp, edition_files, features)
                        n2, node_count = node_nom(verse, editf2, align[1], node_count, nodes_map, x_tmp, edition_files, features)
                        edges_tmp[0].extend([n1, n2])
                        edges_tmp[1].extend([n2, n1])
        args.append((padding, x_tmp, edges_tmp, verse))
        #augment_features_addup(args[-1])
        padding += len(x_tmp)
        #feat = augment_features_addup(len(x), x_tmp, edges_tmp)
        #x.extend(x_tmp)
        edges[0].extend(edges_tmp[0])
        edges[1].extend(edges_tmp[1])

        #if verse == verses[0]:
        #    features.extend(feat)
    utils.LOG.info('augmenting node features')
    with Pool(70) as p:
        res = p.map(augment_features_addup, args)

    features.extend(res[0][0])
    for item in res :
        x.extend(item[1])

    #neg_edge_index = get_negative_edges(verses, edition_files, nodes_map, alignments)
    edge_index = torch.tensor(edges, dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)
    res = Data(x=x, edge_index=edge_index)
    #res.neg_edge_index = neg_edge_index
    res.nodes_map = nodes_map
    res.features = features
    return res, nodes_map


#########################################################
##################### node features #####################
#########################################################

def convert_to_netx(edges):
    utils.LOG.info('converting to nx')
    new_edges = []
    for i in range(0, len(edges[0]), 2):
        new_edges.append((edges[0][i], edges[0][i+1]))
    
    g = nx.Graph()
    g.add_edges_from(new_edges)
    return g
    
def concat(vals, x):
    sum = 0
    sd = 0
    
    for val in vals.items():
        x[val[0]].append(val[1])
        sum += val[1]
        sd += val[1] * val[1]

    try:
        mean = sum/len(x)
        sd = math.sqrt(sd/len(x) - mean*mean)
        for i in range(len(x)):
            x[i][-1] = (x[i][-1] - mean)/sd
    except Exception as e:
        print(e)
        print(len(x), mean, sd)
        raise e


def add_centerality(netx, x, features, centrality_func):
    #utils.LOG.info(f"adding {centrality_func} feature")
    cent = centrality_func(netx)
    concat(cent, x)
    features.append(FloatFeature(4, f"{centrality_type}"))

def add_node_community(communities, x, features, name):
    #utils.LOG.info(f"adding community {name}")
    for i,com in enumerate(communities):
        for node in com:
            x[node].append(i)
    
    #assert len(communities) <= 200
    if len(communities) > 250:
        print('communities len: ', len(communities))

    features.append(OneHotFeature(32, 250, f'{name}'))

def augment_features(dataset, x):
    features = []
    netx = pyg_utils.convert.to_networkx(dataset, to_undirected=True)   # TODO add component size featuer

    utils.LOG.info(f"adding centrality features, nodes: {len(list(netx.nodes))}, edges: {len(list(netx.edges))}")
    add_centerality(netx, x, features, nx.degree_centrality)
    #add_centerality(netx, x, features, nx.katz_centrality)
    add_centerality(netx, x, features, nx.closeness_centrality)
    #add_centerality(netx, x, features, nx.current_flow_closeness_centrality) needs connected graph
    add_centerality(netx, x, features, nx.betweenness_centrality)
    #add_centerality(netx, x, features, nx.current_flow_betweenness_centrality) needs connected graph
    #add_centerality(netx, x, features, nx.communicability_betweenness_centrality)
    add_centerality(netx, x, features, nx.load_centrality)
    #add_centerality(netx, x, features, nx.estrada_index)
    add_centerality(netx, x, features, nx.harmonic_centrality)
    #add_centerality(netx, x, features, nx.percolation_centrality)
    #add_centerality(netx, x, features, nx.second_order_centrality)

    print("creating communities")
    c1 = list(greedy_modularity_communities(netx))
    add_node_community(c1, x, features, 'greedy_modularity_community')
    c3 = list(label_propagation_communities(netx))
    add_node_community(c3, x, features, 'label_propagation_community')

    return features

def augment_features_addup(args):
    padding, x_tmp, edges_tmp, verse = args
    
    edge_index = torch.tensor(edges_tmp, dtype=torch.long) - padding
    x = torch.tensor(x_tmp, dtype=torch.float)
    dataset = Data(x=x, edge_index=edge_index)    
    
    try:
        features = augment_features(dataset, x_tmp)
    except Exception as e:
        print(verse)
        raise e

    return features, x_tmp

def embedding_features(args):
    edges, node_count, rows, cols, features = args
    utils.LOG.info(f"extracting edge features for {node_count} nodes")
    dim = 80
    g = nx.Graph()
    g.add_nodes_from(list(range(node_count)))
    g.add_edges_from(edges)

    res_emb, res_feat = [ ], [ ]
    n2v = Node2Vec(dimensions=dim)
    #n2v.fit(g)

    r2v = Role2Vec(dimensions=dim)
    #r2v.fit(g)

    
    for i in range(6):
        utils.LOG.info(f"fitting {i}")
        ae = AE()
        ae.fit(g, coo_matrix( (features[:,i], (rows, cols)), shape=(node_count, node_count))  )
        res_emb.append(ae.get_embedding())
        res_feat.append(afeatures.PassFeature(dim, f'ae edge embedding {i}'))

    utils.LOG.info(f"finished extracting edge features for {node_count} nodes!!")
    return res_emb, res_feat



def get_embedding_node_features(nodes_map, verses, edition_files, alignments, x_edge_np, x_edge_vals):
    res = None 
    max_ = 0
    args = []
    for verse in verses:
        prev_max = max_
        edges_tmp = []
        edges_pyg = [[],[]]
        utils.LOG.info(f"adding {verse}")
        for i,editf1 in enumerate(edition_files):
            for j,editf2 in enumerate(edition_files[i+1:]):
                aligns = autils.get_aligns(editf1, editf2, alignments[verse])
                if aligns != None:
                    for align in aligns:
                        n1, node_count = node_nom(verse, editf1, align[0], 0, nodes_map, None, None)
                        n2, node_count = node_nom(verse, editf2, align[1], 0, nodes_map, None, None)
                        edges_tmp.append((n1-prev_max, n2-prev_max))
                        edges_pyg[0].append(n1)
                        edges_pyg[1].append(n2)
                        max_ = max(max_, n1+1, n2+1)

        
        val_indices = x_edge_np[edges_pyg[0], edges_pyg[1]]
        val_indices = np.squeeze(np.asarray(val_indices))
        vals = x_edge_vals[val_indices, :]
        rows = np.asarray(edges_pyg[0]) - prev_max
        cols = np.asarray(edges_pyg[1]) - prev_max
        args.append((edges_tmp, max_ - prev_max, rows, cols, vals))

    with Pool(30) as p:  
        res_all = p.map(embedding_features, args)

    for xs,features in res_all:
        res_xs = tuple((torch.from_numpy(x) for x in xs))
        res_x = torch.cat(res_xs, dim=1)
        if res == None:
            res = res_x
        else:
            res = torch.cat((res, res_x), dim=0)

    return res, features

#########################################################
##################### edge features #####################
#########################################################
def concat_edge(cent, x):
    cent = list(cent)
    sum = 0
    std = 0
    for item in cent:
        val = item[2]
        sum += val
        std += val*val
    
    mean = sum/(len(cent))
    std = math.sqrt(std/len(cent) - mean*mean)
    
    for item in cent:
        val = (item[2] - mean)/std # TODO should be sqrt(std)
        x[item[0]][item[1]].append(val)
        x[item[1]][item[0]].append(val)
        
def add_link_prediction_feature(netx, x, prediction_type, ebunch, features, name_post_fix=""):
    utils.LOG.info(f"adding {prediction_type} feature with {name_post_fix}")

    cent = eval(f"nx.{prediction_type}(netx, ebunch)")
   
    concat_edge(cent, x)
    features.append(FloatFeature(2, prediction_type+name_post_fix))

def add_community_features(communities, netx, res, com_name, ebunch, features):
    for i,com in enumerate(communities):
        for node in com:
            netx.nodes[node]['community'] = i

    add_link_prediction_feature(netx, res, 'cn_soundarajan_hopcroft', ebunch, features, com_name)
    add_link_prediction_feature(netx, res, 'ra_index_soundarajan_hopcroft', ebunch, features, com_name)
    add_link_prediction_feature(netx, res, 'within_inter_cluster', ebunch, features, com_name)

def add_graph_size_feature(res, nodes, features, val):
    for i in nodes:
        for j in nodes:
            res[i][j].append(val)

    features.append(FloatFeature(4, global_normalize=True))

def get_edge_features(params):
    edges_tmp, range_ = params
    utils.LOG.info("converting to networkx")
    features = []
    netx = convert_to_netx(edges_tmp)
    
    # add_graph_size_feature(res, range(len(x)), edge_features, len(res), 'verse_node_count')
    # components =  list(nx.algorithms.components.connected_components(netx))
    # add_graph_size_feature(res, range(len(x)), edge_features, len(components)/len(x), )
    res = [[[] for j in range(range_)] for i in range(range_)]
    utils.LOG.info("creating ebunch")
    ebunch = []
    for i, item1 in enumerate(tqdm(list(netx.nodes))):
        for item2 in list(netx.nodes)[i+1:]:
            ebunch.append((item1,item2))

    add_link_prediction_feature(netx, res, 'resource_allocation_index', ebunch, features)
    add_link_prediction_feature(netx, res, 'jaccard_coefficient', ebunch, features)
    add_link_prediction_feature(netx, res, 'adamic_adar_index', ebunch, features)
    add_link_prediction_feature(netx, res, 'preferential_attachment', ebunch, features)

    print("creating communities")
    c1 = list(greedy_modularity_communities(netx))
    c2 = list(asyn_lpa_communities(netx))
    c3 = list(label_propagation_communities(netx))

    add_community_features(c1, netx, res, 'modular', ebunch, features)
    add_community_features(c2, netx, res, 'lpa', ebunch, features)
    add_community_features(c3, netx, res, 'label', ebunch, features)

    return features, res

def build_up_final_res(res_all, args, args2, total_node_count):
    utils.LOG.info('setting up final result')
    res = [[None for j in range(total_node_count)] for i in range(total_node_count)]
    
    utils.LOG.info('adding values')
    for i, edges2 in enumerate(tqdm(args2)):
        edges, _ = args[i]
        diff = edges2[0][0] - edges[0][0]
        for j in range(len(res_all[i][1])):
            for k in range(len(res_all[i][1])):
                res[j+diff][k+diff] =  res_all[i][1][j][k]
    
    return res

def create_edge_attribs(nodes_map, verses, edition_files, alignments, total_node_count):
    args = []
    args2 = []
    min_ = 0
    max_ = 0
    for verse in verses:
        edges_tmp = [[],[]]
        edges_tmp2 = [[],[]]
        utils.LOG.info(f"extracting edge features for {verse}")
        for i,editf1 in enumerate(edition_files):
            for j,editf2 in enumerate(edition_files[i+1:]):
                aligns = autils.get_aligns(editf1, editf2, alignments[verse])
                if aligns != None:
                    for align in aligns:
                        n1, node_count = node_nom(verse, editf1, align[0], 0, nodes_map, None, None)
                        n2, node_count = node_nom(verse, editf2, align[1], 0, nodes_map, None, None)
                        edges_tmp[0].extend([n1-min_, n2-min_])
                        edges_tmp[1].extend([n2-min_, n1-min_])

                        edges_tmp2[0].extend([n1, n2])
                        edges_tmp2[1].extend([n2, n1])

                        max_ = max(n1, n2, max_)
        range_ = max_ - min_ + 1
        min_ = max_+1
        args.append((edges_tmp, range_))
        args2.append(edges_tmp2)

    with Pool(80) as p:  
        res_all = p.map(get_edge_features, args)
    

    res = build_up_final_res(res_all, args, args2, total_node_count)

    return res, res_all[0][0]


####################################################################
##################### intersentence connection #####################
####################################################################

def get_inter_sentence_connections(nodes_map):
    res_all = [[],[]]
    res_neighbor = [[],[]]
    for editf in nodes_map:
        for verse in nodes_map[editf]:
            tokens = sorted(nodes_map[editf][verse].keys())
            for i in range(len(tokens)-1):
                for j in range(i+1, len(tokens)):
                    n1,n2 = nodes_map[editf][verse][tokens[i]],nodes_map[editf][verse][tokens[j]]
                    res_all[0].extend([n1,n2])
                    res_all[1].extend([n2,n1])
                    if abs(tokens[j] - tokens[i]) < 2:
                        res_neighbor[0].extend([n1,n2])
                        res_neighbor[1].extend([n2,n1])
    return res_all, res_neighbor

def get_negative_edges_seq(nodes_map):
    res_edges = [[],[]]
    for editf in nodes_map:
        for verse in nodes_map[editf]:
            tokens = sorted(nodes_map[editf][verse].keys())
            for i in range(len(tokens)-1):
                candid = choice(range(len(tokens)))
                if candid not in [i, i-1, i+1]:
                    n1,n2 = nodes_map[editf][verse][tokens[i]],nodes_map[editf][verse][tokens[candid]]
                    res_edges[0].extend([n1,n2])
                    res_edges[1].extend([n2,n1])
            
    return torch.tensor(res_edges, dtype=torch.long)
