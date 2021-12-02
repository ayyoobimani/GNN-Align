import torch, numpy as np, sys, os
import torch.nn.functional as F
sys.path.insert(0, '../')
from my_utils import align_utils as autils, utils
import gc
from tqdm import tqdm

intra_sent_edges = {}
def get_all_edges(verse, editf, nodes_map, verse_info):
    if verse in intra_sent_edges and editf in intra_sent_edges[verse]:
        return intra_sent_edges[verse][editf]

    res = [[],[]]
    mlist = list(nodes_map[editf][verse].values())
    for i, n1 in enumerate(mlist):
        n1 -= verse_info[verse]['padding']
        for n2 in mlist[i:]:
            n2 -= verse_info[verse]['padding']
            res[0].extend([n1, n2])
            res[1].extend([n2, n1])
    
    utils.setup_dict_entry(intra_sent_edges, verse, {})
    intra_sent_edges[verse][editf] = res

    return res

def create_all_edges(editf1, editf2, verse, nodes_map, verses_info):
    res = [[],[]]
    
    for n1 in sorted(list(nodes_map[editf1][verse].keys())):
        for n2 in sorted(list(nodes_map[editf2][verse].keys())):
            res[0].append(nodes_map[editf1][verse][n1] - verses_info[verse]['padding']) 
            res[1].append(nodes_map[editf2][verse][n2] - verses_info[verse]['padding']) 
    

    return  torch.tensor(res, dtype=torch.long)

def compute_preds(editf1, editf2, verse, edge_index, nodes_map, dev, model, x, verses_info): # TODO consider directed edges too

    all_edges = create_all_edges(editf1, editf2, verse, nodes_map, verses_info).to(dev)
    #intra = [[],[]]
    #intra[0].extend(get_all_edges(verse, editf1, nodes_map, verses_info)[0])
    #intra[1].extend(get_all_edges(verse, editf1, nodes_map, verses_info)[1])
    #intra[0].extend(get_all_edges(verse, editf2, nodes_map, verses_info)[0])
    #intra[1].extend(get_all_edges(verse, editf2, nodes_map, verses_info)[1])

    model.eval()
    z = model.encode(verses_info[verse]['x'].to(dev), verses_info[verse]['edge_index'].to(dev))
    #z2 = model.encoder2(z, torch.tensor(intra, dtype=torch.long).to(dev))
    #z = torch.cat((z, z2), dim=1)
    with torch.no_grad():
        preds = model.decode(z, all_edges)
        #preds = model.decoder.get_alignments(z, all_edges)
        if len(preds.shape) > 1 and preds.shape[1] > 1:
            preds = F.softmax(preds)
            preds = preds[:,1]

    all_edges = None

    return preds

def update_data(editions, verses, edge_index, nodes_map, dev, model, x, all_edges_index, targets, alignments):

    for i,editf1 in enumerate(editions):
        for editf2 in editions[i+1:]:
            preds = compute_preds(editf1, editf2, verses, edge_index, nodes_map, dev, model, x) #this function has changed
            start_pos = 0
            for verse in verses:
                lent = len(nodes_map[editf1][verse]) * len(nodes_map[editf2][verse])
                verse_probs = np.array(preds.cpu()[start_pos:lent + start_pos])
                start_pos += lent
                verse_probs = verse_probs.reshape( (-1, len(nodes_map[editf2][verse])) )

                matrix_argmax = autils.iter_max(verse_probs, max_count=1)
                for i,n1 in enumerate(sorted(list(nodes_map[editf1][verse].keys()))):
                    for j,n2 in enumerate(sorted(list(nodes_map[editf2][verse].keys()))):
                        if matrix_argmax[i,j] != 0:
                            node1 = nodes_map[editf1][verse][n1]
                            node2 = nodes_map[editf2][verse][n2]
                            idx = all_edges_index[node1,node2]
                            if targets[idx] == 0:
                                if editf1 in alignments[verse] and editf2 in alignments[verse][editf1]:
                                    alignments[verse][editf1][editf2] = alignments[verse][editf1][editf2] + f" {n1}-{n2}"
                                else:
                                    utils.setup_dict_entry(alignments[verse], editf2, {})
                                    utils.setup_dict_entry(alignments[verse][editf2], "")
                                    alignments[verse][editf2][editf1] = alignments[verse][editf2][editf1].strip() + f" {n2}-{n1}"
                                targets[idx] = 1
                                edge_index = torch.cat((edge_index, torch.tensor([[node1, node2],[node2, node1]], dtype=torch.long).to(dev)), dim=1)
    return edge_index

def convert(nodes_map, editf1, editf2, verse, align):
    res = set()
    for i,n1 in enumerate(sorted(list(nodes_map[editf1][verse].keys()))):
        for j,n2 in enumerate(sorted(list(nodes_map[editf2][verse].keys()))):
            if (n1, n2) in align:
                res.add((i,j))
    
    return res

def calc_res(verse, verse_probs, nodes_map, editf1, editf2, align_gdfa=None, align_inter=None):
    import importlib
    importlib.reload(autils)
    res = {}

    align_inter = convert(nodes_map, editf1, editf2, verse, align_inter)
    align_gdfa = convert(nodes_map, editf1, editf2, verse, align_gdfa)

    s_list_poses, t_list_poses = sorted(list(nodes_map[editf1][verse].keys())), sorted(list(nodes_map[editf2][verse].keys()))
    
    matrix_argmax = autils.iter_max(verse_probs, max_count=1)
    #matrix_iter29 = autils.iter_max(verse_probs, max_count=2, alpha_ratio = 0.9)
    #matrix_iter295 = autils.iter_max(verse_probs, max_count=2, alpha_ratio = 0.95)
    #matrix_iter285 = autils.iter_max(verse_probs, max_count=2, alpha_ratio = 0.85)
    pp = []
    for i in range(verse_probs.shape[0]):
        for j in range(verse_probs.shape[1]):
            if verse_probs[i,j]>0.5:
                pp.append((i,j))

    matrix_my_gd = autils.my_gd(verse_probs, s_list_poses, t_list_poses)
    matrix_mygd_gdfa = autils.my_gd(verse_probs, s_list_poses, t_list_poses , prev_gdfa=align_gdfa)
    matrix_gdfa = autils.grow_diag_final_and(verse_probs)


    #thresh = 0.0
    #verse_probs1 = torch.softmax(torch.from_numpy(verse_probs)*500, dim=1)
    #verse_probs2 = torch.softmax(torch.from_numpy(verse_probs)*500, dim=0)
    #new1 = (verse_probs2 > (1/verse_probs.shape[0])+thresh) * (verse_probs1 > (1/verse_probs.shape[1])+thresh)
    #intersect = set()
    #for i in range(new1.shape[0]):
    #    for j in range(new1.shape[1]):
    #        if new1[i,j] == 1:
    #            intersect.add((i,j))
    #new_mygd = autils.my_gd(verse_probs, s_list_poses, t_list_poses, alignment=intersect)
    #new_mygd_gdfa = autils.my_gd(verse_probs, s_list_poses, t_list_poses, alignment=intersect, prev_gdfa=align_gdfa)
    #new_mygd = autils.my_gd(verse_probs, alignment=new_mygd, tresh=0)


    argMax = []
    res_norm = []
    res_iter29 = []
    res_iter295 = []
    res_iter285 = []
    res_my_gd = []
    res_my_gd_gdfa = []
    res_gdfa = []
    res_new1 = []
    res_new_mygd = []
    res_new_mygd_gdfa = []
    for i,n1 in enumerate(sorted(list(nodes_map[editf1][verse].keys()))):
        for j,n2 in enumerate(sorted(list(nodes_map[editf2][verse].keys()))):
            if matrix_argmax[i,j] != 0:
                argMax.append((n1,n2))
            if verse_probs[i,j] > 0.5:
                res_norm.append((n1,n2))
            #if matrix_iter29[i,j] != 0:
            #    res_iter29.append((n1,n2))
            #if matrix_iter295[i,j] != 0:
            #    res_iter295.append((n1,n2))
            #if matrix_iter285[i,j] != 0:
            #    res_iter285.append((n1,n2))
            if (i,j) in matrix_my_gd:
                res_my_gd.append((n1,n2))
            if (i,j) in matrix_mygd_gdfa:
                res_my_gd_gdfa.append((n1,n2))
            if (i,j) in matrix_gdfa:
                res_gdfa.append((n1,n2))
            #if new1[i,j] == 1:
            #    res_new1.append((n1,n2))
            #if (i,j) in new_mygd:
            #    res_new_mygd.append((n1,n2))
            #if (i,j) in new_mygd_gdfa:
            #    res_new_mygd_gdfa.append((n1,n2))
    
    
    #print(verse, len(sum1), len(res_my_gd), len(res_new_mygd))
    res = {'my_gdfa':res_gdfa, 'argmax': argMax, 'resnorm': res_norm, 'itermax2-.9':res_iter29, 'itermax2-.95':res_iter295, 'itermax2-.8':res_iter285, 
            'my_gd':res_my_gd, 'my_gd_gdfa':res_my_gd_gdfa, 'new1':res_new1, 'new_mygd': res_new_mygd, 'new_mygd_gdfa': res_new_mygd_gdfa }

    return res


def alignment_test(epoch, edge_index, editf1, editf2, verses, nodes_map, dev,
                     model, x, pros, surs, alignments_inter, alignments_gdfa, writer, verses_info, calc_numbers=True):
    
    res = {}
    measures = {}
    measures['intersection'] = {"p_hit_count": 0, "s_hit_count": 0, "total_hit_count": 0, "gold_s_hit_count": 0, "prec": 0, "rec": 0, "f1": 0, "aer": 0}
    measures['gdfa'] = {"p_hit_count": 0, "s_hit_count": 0, "total_hit_count": 0, "gold_s_hit_count": 0, "prec": 0, "rec": 0, "f1": 0, "aer": 0}
    for verse in tqdm(verses):
        preds = compute_preds(editf1, editf2, verse, edge_index, nodes_map, dev, model, x, verses_info)
        inter_aligns = autils.get_aligns(editf1, editf2, alignments_inter[verse])
        gdfa_aligns = autils.get_aligns(editf1, editf2, alignments_gdfa[verse])
        verse_probs = np.array(preds.cpu())
        verse_probs = verse_probs.reshape( (-1, len(nodes_map[editf2][verse])) )
        pred_aligns = calc_res(verse, verse_probs, nodes_map, editf1, editf2, align_gdfa=gdfa_aligns, align_inter=inter_aligns)
        res[verse] = { 'GNN': pred_aligns['my_gd'], 'GNN_GDFA': pred_aligns['my_gd_gdfa'], 'GNN_my_gd': pred_aligns['my_gd']}

        if calc_numbers:
            for method in pred_aligns:
                utils.setup_dict_entry(measures, method, {"p_hit_count": 0, "s_hit_count": 0, "total_hit_count": 0, "gold_s_hit_count": 0, "prec": 0, "rec": 0, "f1": 0, "aer": 0})
                autils.calc_and_update_alignment_score(pred_aligns[method], pros[verse], surs[verse], measures[method])
            autils.calc_and_update_alignment_score(inter_aligns, pros[verse], surs[verse], measures['intersection'])
            autils.calc_and_update_alignment_score(gdfa_aligns, pros[verse], surs[verse], measures['gdfa'])
    
    if calc_numbers:
        writer.add_scalar('alignment_Prec', measures['my_gd']['prec'], epoch)
        writer.add_scalar('alignment_Rec', measures['my_gd']['rec'], epoch)
        writer.add_scalar('alignment_F1', measures['my_gd']['f1'], epoch)
        print("\n")
        for method in measures:
            print(method, f"prec: {measures[method]['prec']}, rec: {measures[method]['rec']}, F1: {measures[method]['f1']}, AER: {measures[method]['aer']}")
    
    return res

def checkpoint(F1, epoch, net, run):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': F1,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('/mounts/work/ayyoob/models/gnn/checkpoint'):
        os.mkdir('/mounts/work/ayyoob/models/gnn/checkpoint')
    torch.save(state, '/mounts/work/ayyoob/models/gnn//checkpoint/ckpt.t7.' + str(run))    
    