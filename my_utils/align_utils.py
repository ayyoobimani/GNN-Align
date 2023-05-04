import os, torch, math
from torch.nn.functional import threshold
from my_utils import utils
import pickle
import numpy as np
from collections import defaultdict


def load_gold(g_path):
    gold_f = open(g_path, "r")
    pros = {}
    surs = {}

    for line in gold_f:
        line = line.strip().split("\t")
        line[1] = line[1].split()

        pros[line[0]] = set([x.replace("p", "-") for x in line[1]])
        surs[line[0]] = set([x for x in line[1] if "p" not in x])

        # pros[line[0]] = set([x.replace("p", "-") for x in line[1]])
        # surs[line[0]] = set([x for x in line[1] if "p" not in x])

    return pros, surs

def load_simalign_editions(editions_file):
    editions = {}
    langs = []
    #langs = ['eng', 'fra']
    #editions['eng'] = 'eng-x-bible-mixed'
    #editions['fra'] = 'fra-x-bible-louissegond'
    with open(editions_file) as f_lang_list:
        lines = f_lang_list.read().splitlines()
        for line in lines[:]: # start reading from the third line
            comps = line.split('\t')
            editions[comps[0]] = comps[1] #.replace('-x-bible','')
            langs.append(comps[0])
    return editions, langs

def get_verse_alignments(verse_id, verse_alignments=None, is_gdfa=False):
    #utils.LOG.info(f"reading verse alignment file {verse_id}")
    if verse_alignments != None:
        return verse_alignments
    
    gdfa = 'inter'
    if is_gdfa:
        gdfa = 'gdfa'

    f_path = utils.AlignInduction.verse_alignments_path + f"/{gdfa}/{verse_id}.txt"
    f_path_bin = f_path + ".bin"

    if os.path.exists(f_path_bin):
        #utils.LOG.info(f"loading pickle file {f_path_bin}")
        with open(f_path_bin, 'rb') as inf:
            try:
                return pickle.load(inf)
            except:
                pass
    
    if not os.path.exists(f_path):
        utils.LOG.info(f_path)
        utils.LOG.info(f"==================================alignment file for verse {verse_id} dos not exist==================================")
        return None
        
    res = {}

    with open(f_path, 'r') as f:
        lines = list(f.readlines())
        for line in lines:
            s_file, t_file, aligns = tuple(line.split('\t'))
            utils.setup_dict_entry(res, s_file, {})
            res[s_file][t_file] = aligns
    
    with open(f_path_bin, 'wb') as of:
        pickle.dump(res, of)

    return res

def get_aligns(rf, cf, alignments):
    raw_align = ''
    #print(rf, cf, alignments)
    if rf in alignments and cf in alignments[rf]:
        raw_align = alignments[rf][cf]
        alignment_line = [x.split('-') for x in raw_align.split()]
        res = []
        for x in alignment_line:
            res.append( ( int(x[0]), int(x[1]) ) )
    elif cf in alignments and rf in alignments[cf]: # re: aak, ce: aai, 
        raw_align = alignments[cf][rf]
        alignment_line = [x.split('-') for x in raw_align.split()]
        res = []
        for x in alignment_line:
            res.append( ( int(x[1]), int(x[0]) ) )
    elif rf in alignments and rf == cf: # if source and target are the same
        keys = list(alignments[rf].keys())
        max_count = 0
        for key in keys:
            align = alignments[rf][key]
            for x in align.split():
                count = int(x.split('-')[0])
                if count > max_count:
                    max_count = count
        raw_align = "0-0"
        for i in range(1,max_count):
            raw_align += f" {i}-{i}"

        alignment_line = [x.split('-') for x in raw_align.split()]
        res = []
        for x in alignment_line:
            res.append( ( int(x[0]), int(x[1]) ) )
    else:
        # utils.LOG.info(f"re: {rf}\nce: {cf}\nalignments.keys: {list(alignments.keys())}")
        return None
    
    return res

def iter_max(sim_matrix: np.ndarray, max_count: int=2, alpha_ratio = 0.7) -> np.ndarray:
    m, n = sim_matrix.shape
    forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
    backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
    inter = forward * backward.transpose()

    if min(m, n) <= 2:
        return inter

    new_inter = np.zeros((m, n))
    count = 1
    while count < max_count:
        mask_x = 1.0 - np.tile(inter.sum(1)[:, np.newaxis], (1, n)).clip(0.0, 1.0)
        mask_y = 1.0 - np.tile(inter.sum(0)[np.newaxis, :], (m, 1)).clip(0.0, 1.0)
        mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
        mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
        if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
            mask *= 0.0
            mask_zeros *= 0.0

        new_sim = sim_matrix * mask
        fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
        bac = np.eye(m)[new_sim.argmax(axis=0)].transpose() * mask_zeros
        new_inter = fwd * bac

        if np.array_equal(inter + new_inter, inter):
            break
        inter = inter + new_inter
        count += 1
    return inter

def calc_and_update_alignment_score(aligns, pros, surs, results):
    #if len(aligns) == 0: return None

    aligns = set(aligns)
    tmp_aligns = set([str(align[0]) + "-" + str(align[1]) for align in aligns])
    aligns = tmp_aligns

    p_hit = len(aligns & pros)
    s_hit = len(aligns & surs)
    total_hit = len(aligns)

    results["p_hit_count"] += p_hit
    results["s_hit_count"] += s_hit
    results["total_hit_count"] += total_hit
    results["gold_s_hit_count"] += len(surs)

    prec = round(p_hit/ max(total_hit, 0.01), 3)
    rec = round(s_hit/len(surs), 3)
    f1 = round(2*prec*rec/max(prec+rec, 0.01), 3)

    results["prec"] = round(results["p_hit_count"] / max(results["total_hit_count"], 0.01), 3)
    results["rec"] = round(results["s_hit_count"] / results["gold_s_hit_count"], 3)
    results["f1"] = round(2. * results["prec"] * results["rec"] / max((results["prec"] + results["rec"]), 0.01), 3)
    results["aer"] = round(1 - (results["s_hit_count"] + results["p_hit_count"]) / (results["total_hit_count"] + results["gold_s_hit_count"]), 3)

    return prec, rec, f1

#from collections import defaultdict

def my_gd(sim_matrix, s_list_poses, t_list_poses, tresh = 0.0, alignment=None, union=None, prev_gdfa=set(), prev_inter=set()):
    
    sim_matrix = np.copy(sim_matrix)
    aligned = defaultdict(set)
    srclen, trglen = sim_matrix.shape
    fow = sim_matrix.argmax(axis=1)
    bac = sim_matrix.argmax(axis=0)
    e2f = list(zip(range(srclen), fow))
    f2e = list(zip(bac, range(trglen)))

    #print('e2f', e2f)
    #print('f2e', f2e)

    if alignment == None:
        alignment = set(e2f).intersection(set(f2e))  # Find the intersection.

    #print('shapes', sim_matrix.shape)
    fow_tresh = (2/(sim_matrix.shape[1]))
    bac_tresh = (2/(sim_matrix.shape[0]))
    #fow_tresh = (math.exp(2)/math.exp(sim_matrix.shape[1]+3))
    #bac_tresh = (math.exp(2)/math.exp(sim_matrix.shape[0]+3))
    #print('fow tresh', fow_tresh, 'backtresh', bac_tresh)
    
    sim_fow = torch.softmax(torch.from_numpy(sim_matrix), dim=1) 
    sim_bac = torch.softmax(torch.from_numpy(sim_matrix), dim=0) 
    #sim_fow = sim_matrix/sim_matrix.sum(axis=1)[:, np.newaxis]
    #sim_bac = sim_matrix/sim_matrix.sum(axis=0)[np.newaxis, :]


    #print('simfow', sim_fow)
    #print('simback', sim_bac)
    for item in list(alignment):
        if sim_fow[item[0], item[1]] < fow_tresh or sim_bac[item[0], item[1]] < bac_tresh:
            alignment.remove(item)
    
    
    #fow_tresh = (1/sim_matrix.shape[1])
    #bac_tresh = (1/sim_matrix.shape[0])

    #print('len alignments1', len(alignment))

    neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    alignment = alignment.union(prev_inter)
    
    if union == None:
        union = set(e2f).union(set(f2e))
    union = union.union(prev_gdfa)

    for i, j in alignment:
        aligned["e"].add(i)
        aligned["f"].add(j)
    
    for i in range(8):
        for e in range(srclen):
                    # for foreign word f = 0 ... fn
            for f in range(trglen):
                # if ( e aligned with f)
                if (e, f) in alignment:
                    # for each neighboring point (e-new, f-new)
                    for neighbor in neighbors:
                        neighbor = tuple(i + j for i, j in zip((e, f), neighbor))
                        e_new, f_new = neighbor
                        # if ( ( e-new not aligned and f-new not aligned)
                        # and (e-new, f-new in union(e2f, f2e) )
                        if (
                            neighbor in union and 
                            #e_new < sim_matrix.shape[0]  and f_new < sim_matrix.shape[1] and 
                            (( sim_fow[neighbor] > fow_tresh and neighbor in e2f ) or ( sim_bac[neighbor] > bac_tresh and neighbor in f2e)
                            or ((neighbor in (prev_gdfa) and (sim_fow[neighbor] > 0 or sim_bac[neighbor] > 0 )) )
                            )
                            
                            and (abs(s_list_poses[e_new] - s_list_poses[e]) <= 1 and abs(t_list_poses[f_new] - t_list_poses[f]) <= 1)
                            #and  (neighbor in prev_gdfa and (sim_fow[neighbor] > tresh or sim_bac[neighbor]>tresh or True))
                            
                            
                        ):
                            alignment.add(neighbor)
                            aligned["e"].add(e_new)
                            aligned["f"].add(f_new)
    

    #print('len alignments2', len(alignment))
    #print('len union', len(union))

    #fow_tresh = (4/(sim_matrix.shape[1]+2))
    #bac_tresh = (4/(sim_matrix.shape[0]+2))

    for i in range(2):
        for e in range(srclen):
            for f in range(trglen):
                neighbor = (e,f)
                if (
                    e not in aligned['e']
                    and f not in aligned['f']
                    and (e,f) in union
                    and ((neighbor in e2f and sim_fow[neighbor] > fow_tresh) or (neighbor in f2e and sim_bac[neighbor] > bac_tresh) or neighbor in prev_gdfa)
                    ):
                    alignment.add(neighbor)
                    aligned['e'].add(e)
                    aligned['f'].add(f)
    
    #print('len alignments3', len(alignment))
    return alignment
    


def grow_diag_final_and(sim_matrix, tresh = 0.0):
    """
    This module symmetrisatizes the source-to-target and target-to-source
    word alignment output and produces, aka. GDFA algorithm (Koehn, 2005).

    Step 1: Find the intersection of the bidirectional alignment.

    Step 2: Search for additional neighbor alignment points to be added, given
            these criteria: (i) neighbor alignments points are not in the
            intersection and (ii) neighbor alignments are in the union.

    Step 3: Add all other alignment points thats not in the intersection, not in
            the neighboring alignments that met the criteria but in the original
            foward/backward alignment outputs.

        >>> forw = ('0-0 2-1 9-2 21-3 10-4 7-5 11-6 9-7 12-8 1-9 3-10 '
        ...         '4-11 17-12 17-13 25-14 13-15 24-16 11-17 28-18')
        >>> back = ('0-0 1-9 2-9 3-10 4-11 5-12 6-6 7-5 8-6 9-7 10-4 '
        ...         '11-6 12-8 13-12 15-12 17-13 18-13 19-12 20-13 '
        ...         '21-3 22-12 23-14 24-17 25-15 26-17 27-18 28-18')
        >>> srctext = ("この よう な ハロー 白色 わい 星 の Ｌ 関数 "
        ...            "は Ｌ と 共 に 不連続 に 増加 する こと が "
        ...            "期待 さ れる こと を 示し た 。")
        >>> trgtext = ("Therefore , we expect that the luminosity function "
        ...            "of such halo white dwarfs increases discontinuously "
        ...            "with the luminosity .")
        >>> srclen = len(srctext.split())
        >>> trglen = len(trgtext.split())
        >>>
        >>> gdfa = grow_diag_final_and(srclen, trglen, forw, back)
        >>> gdfa == sorted(set([(28, 18), (6, 6), (24, 17), (2, 1), (15, 12), (13, 12),
        ...         (2, 9), (3, 10), (26, 17), (25, 15), (8, 6), (9, 7), (20,
        ...         13), (18, 13), (0, 0), (10, 4), (13, 15), (23, 14), (7, 5),
        ...         (25, 14), (1, 9), (17, 13), (4, 11), (11, 17), (9, 2), (22,
        ...         12), (27, 18), (24, 16), (21, 3), (19, 12), (17, 12), (5,
        ...         12), (11, 6), (12, 8)]))
        True

    References:
    Koehn, P., A. Axelrod, A. Birch, C. Callison, M. Osborne, and D. Talbot.
    2005. Edinburgh System Description for the 2005 IWSLT Speech
    Translation Evaluation. In MT Eval Workshop.

    :type srclen: int
    :param srclen: the number of tokens in the source language
    :type trglen: int
    :param trglen: the number of tokens in the target language
    :type e2f: str
    :param e2f: the forward word alignment outputs from source-to-target
                language (in pharaoh output format)
    :type f2e: str
    :param f2e: the backward word alignment outputs from target-to-source
                language (in pharaoh output format)
    :rtype: set(tuple(int))
    :return: the symmetrized alignment points from the GDFA algorithm
    """

 
    srclen, trglen = sim_matrix.shape
    fow = sim_matrix.argmax(axis=1)
    bac = sim_matrix.argmax(axis=0)
    e2f = list(zip(range(srclen), fow))
    f2e = list(zip(bac, range(trglen)))
    
    neighbors = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    alignment = set(e2f).intersection(set(f2e))  # Find the intersection.
    union = set(e2f).union(set(f2e))

    # *aligned* is used to check if neighbors are aligned in grow_diag()
    aligned = defaultdict(set)
    for i, j in alignment:
        aligned["e"].add(i)
        aligned["f"].add(j)

    def grow_diag():
        """
        Search for the neighbor points and them to the intersected alignment
        points if criteria are met.
        """
        prev_len = len(alignment) - 1
        # iterate until no new points added
        while prev_len < len(alignment):
            no_new_points = True
            # for english word e = 0 ... en
            for e in range(srclen):
                # for foreign word f = 0 ... fn
                for f in range(trglen):
                    # if ( e aligned with f)
                    if (e, f) in alignment:
                        # for each neighboring point (e-new, f-new)
                        for neighbor in neighbors:
                            neighbor = tuple(i + j for i, j in zip((e, f), neighbor))
                            e_new, f_new = neighbor
                            # if ( ( e-new not aligned and f-new not aligned)
                            # and (e-new, f-new in union(e2f, f2e) )
                            if (
                                e_new not in aligned and f_new not in aligned
                            ) and neighbor in union and sim_matrix[neighbor] > tresh:
                                alignment.add(neighbor)
                                aligned["e"].add(e_new)
                                aligned["f"].add(f_new)
                                prev_len += 1
                                no_new_points = False
            # iterate until no new points added
            if no_new_points:
                break

    def final_and(a):
        """
        Adds remaining points that are not in the intersection, not in the
        neighboring alignments but in the original *e2f* and *f2e* alignments
        """
        # for english word e = 0 ... en
        for e_new in range(srclen):
            # for foreign word f = 0 ... fn
            for f_new in range(trglen):
                # if ( ( e-new not aligned and f-new not aligned)
                # and (e-new, f-new in union(e2f, f2e) )
                if (
                    e_new not in aligned
                    and f_new not in aligned
                    and (e_new, f_new) in union
                    and sim_matrix[e_new, f_new] > tresh
                ):
                    alignment.add((e_new, f_new))
                    aligned["e"].add(e_new)
                    aligned["f"].add(f_new)

    grow_diag()
    final_and(e2f)
    final_and(f2e)
    return sorted(alignment)


def prune_non_necessary_alignments(verse_aligns, necessary_editions):
    non_necessary = list(verse_aligns.keys())
    for i in necessary_editions:
        if i in non_necessary:
            non_necessary.remove(i)
    
    for i in non_necessary:
        if i in verse_aligns:
            verse_aligns[i] = {}
        for editf in verse_aligns:
            if i in verse_aligns[editf]:
                verse_aligns[editf][i] = None
