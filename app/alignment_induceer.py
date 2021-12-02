from surprise import Dataset, Reader, similarities
import pandas as pd
from surprise import SVD, NMF, KNNBasic
from app.utils import setup_dict_entry
from app import utils
from random import  randrange, shuffle
from app.document_retrieval import DocumentRetriever
from app.general_align_reader import GeneralAlignReader
import time, os, pickle, argparse, random, math, json
from nltk.translate.gdfa import grow_diag_final_and
import numpy as np
from multiprocessing import Pool
from app import utils_induction as iutils

doc_retriever = DocumentRetriever()
align_reader = GeneralAlignReader()


def get_row_col_editions(source_edition, target_edition, verse_id, verse_alignments, edition_count, single_edition, align_reader, cols_twice_rows, all_editions=None):
    source_lang = align_reader.get_lang_from_edition(source_edition)
    target_lang = align_reader.get_lang_from_edition(target_edition)

    bad_langs = ["jpn", "bod", "lzh", "ksw", "kos", "khm", "quy"]
    bad_langs = []
    if all_editions == None:
        all_editions = [align_reader.file_edition_mapping[f] for f in verse_alignments.keys()]
        random.shuffle(all_editions)
    else:
        all_editions = [align_reader.file_edition_mapping[f] for f in all_editions]

    valid_editions = []
    used_langs = set()
    for edition in all_editions:
        lang = align_reader.get_lang_from_edition(edition)
        
        if edition != source_edition and edition != target_edition and \
            lang not in bad_langs and not (single_edition and (lang in used_langs) ):
            used_langs.add(lang)
            valid_editions.append(edition)

    row_editions = []
    col_editions = []
    for edition in valid_editions[:min(edition_count, len(valid_editions))]:
        row_editions.append(edition)
        col_editions.append(edition)

    if cols_twice_rows:
        for edition in valid_editions[:min(2 * edition_count, len(valid_editions))]:
            if edition not in col_editions:
                col_editions.append(edition)
    
    row_editions.append(source_edition)
    col_editions.append(target_edition)

    # utils.LOG.info(f"edition count: {len(row_editions)}\n single edition {single_edition}")

    return row_editions, col_editions

def remove_lang_from_rowcol(lang, row_langs, column_langs):
    if lang in row_langs:
        row_langs.remove(lang)
    if lang in column_langs:
        column_langs.remove(lang)

def splitted_alignments(re, ce, raw_align):
    alignment_line = [x.split('-') for x in raw_align.split()]
    revert = False

    res = []
    _, _ , s_e, t_e = align_reader.get_ordered_editions(re, ce)
    revert = True if s_e == ce else False
    for x in alignment_line:
        res.append((int(x[1]), int(x[0])) if revert else (int(x[0]), int(x[1])) )
    return res


def add_aligns(aligns, aligns_dict, token_counts, re, ce, existing_items):
    for align in aligns:

        aligns_dict['userID'].append(re + str(align[0]))
        aligns_dict['itemID'].append(ce + str(align[1]))
        aligns_dict['rating'].append(3)

        if align[0] > token_counts[re]:
            token_counts[re] = align[0]
        if align[1] > token_counts[ce]:
            token_counts[ce] = align[1]
        
        existing_items[re][ce].append(f"{align[0]},{align[1]}")

def add_negative_samples(aligns_dict, existing_items, token_counts, verse_id):
    for re in existing_items:
        if token_counts[re] < 2:
        #     utils.LOG.info(f"for verse {verse_id} and row edition {re} only {token_counts[re]} token alignment exists")
            continue
        for ce in existing_items[re]:
            if token_counts[ce] < 2:
                # utils.LOG.info(f"for verse {verse_id} and column edition {ce} only {token_counts[ce]} token alignment exists")
                continue
            for item in existing_items[re][ce]:
                i,j = tuple(item.split(","))
                i,j = (int(i), int(j))
                jp = random.randint(math.ceil(j+1), math.ceil(j+token_counts[ce] ))
                ip = random.randint(math.ceil(i+1), math.ceil(i+token_counts[re] ))

                jp %= (token_counts[ce] + 1)
                aligns_dict['userID'].append(re + str(i))
                aligns_dict['itemID'].append(ce + str(jp))
                aligns_dict['rating'].append(1)
                
                ip %= (token_counts[re] + 1) 
                aligns_dict['userID'].append(re + str(ip))
                aligns_dict['itemID'].append(ce + str(j))
                aligns_dict['rating'].append(1)

def get_alignments_df(row_editions, col_editions, verse_alignments,
        verse_alignments_gdfa, source_edition, target_edition, verse_id, align_reader): #TODO can be improved a lot
    utils.LOG.info("creating dataframe")
    token_counts = {}
    existing_items = {}
    existing_items_gdfa = {}
    existing_items_inde = {}

    aligns_dict = {'itemID': [], 'userID': [], 'rating': []}
    aligns_dict_gdfa = {'itemID': [], 'userID': [], 'rating': []}
    aligns_dict_inde = {'itemID': [], 'userID': [], 'rating': []}
    for no, re in enumerate(row_editions):
        token_counts[re] = 0
        existing_items[re] = {}
        existing_items_gdfa[re] = {}
        existing_items_inde[re] = {}
        for ce in col_editions:

            setup_dict_entry(token_counts, ce, 0)
            existing_items[re][ce] = []
            existing_items_gdfa[re][ce] = []
            existing_items_inde[re][ce] = []

            aligns = iutils.get_aligns(align_reader.edition_file_mapping[re], align_reader.edition_file_mapping[ce], verse_alignments)
            aligns_gdfa = aligns
            aligns_inde = aligns
            if re == source_edition and ce == target_edition:
                aligns_gdfa = iutils.get_aligns(align_reader.edition_file_mapping[re], align_reader.edition_file_mapping[ce], verse_alignments_gdfa)
                
                aligns_inde = None

            if not aligns is None:
                add_aligns(aligns, aligns_dict, token_counts, re, ce, existing_items)
            if not aligns_gdfa is None:
                add_aligns(aligns_gdfa, aligns_dict_gdfa, token_counts, re, ce, existing_items_gdfa)
            if not aligns_inde is None:
                add_aligns(aligns_inde, aligns_dict_inde, token_counts, re, ce, existing_items_inde)

    add_negative_samples(aligns_dict, existing_items, token_counts, verse_id)
    add_negative_samples(aligns_dict_gdfa, existing_items_gdfa, token_counts, verse_id)
    add_negative_samples(aligns_dict_inde, existing_items_inde, token_counts, verse_id)
     
    # utils.LOG.info(f"token_counts: {token_counts}, source_edition: {source_edition}, target_edition: {target_edition}")
    return pd.DataFrame(aligns_dict), pd.DataFrame(aligns_dict_gdfa), pd.DataFrame(aligns_dict_inde), token_counts[source_edition], token_counts[target_edition]
 
def remove_aligned_words(source_predictions, target_predictions, i , j):
    # utils.LOG.info(f"remove aligned word, {i}, {j}")
    for item in source_predictions:
        to_remove = 1000000
        for ind, score_tuple in enumerate(source_predictions[item]):
            if score_tuple[0] == j:
                to_remove = ind
        if to_remove != 1000000:
            source_predictions[item].pop(to_remove)
    
    for item in target_predictions:
        to_remove = 1000000
        for ind, score_tuple in enumerate(target_predictions[item]):
            if score_tuple[0] == i:
                to_remove = ind
        if to_remove != 1000000:
            target_predictions[item].pop(to_remove)
    
def normalize_scores(source_predictions, target_predictions):
    res = {}
    for item in target_predictions:
        res[item] = []

    for item in source_predictions:
        scores = source_predictions[item]
        s = sum([pair[1] for pair in scores])

        for pair in scores:
            res[pair[0]].append((item, pair[1]/s))
    
    return res

def get_gdfa_predictions(source_predictions, target_predictions):
    threshold = [2.5, 2.3, 2, 1.7]
    first_k = [-1, 2, 3, 4]
    res = {}
    for ind in range(4):
        forw = ''
        back = ''
        forw_t = ''
        back_t = ''


        for i in range(len(source_predictions)):
            j = source_predictions[i][0][0]
            if i in [ x[0] for x in target_predictions[j][:first_k[ind]]]:
                back += f"{i}-{j} "
            if source_predictions[i][0][1] > threshold[ind] or target_predictions[j][0][0] == i:
                back_t += f"{i}-{j} "
        
        for i in range(len(target_predictions)):
            j = target_predictions[i][0][0]
            if i in [ x[0] for x in source_predictions[j][:first_k[ind]]]:
                forw += f"{j}-{i} "
            if target_predictions[i][0][1] > threshold[ind] or source_predictions[j][0][0] == i:
                forw_t += f"{j}-{i} "
        res[f"gdfa_fk[{first_k[ind]}]"] = list(grow_diag_final_and(len(source_predictions), len(target_predictions), forw.strip(), back.strip()))
        res[f"gdfa_th[{threshold[ind]}]"] = list(grow_diag_final_and(len(source_predictions), len(target_predictions), forw_t.strip(), back_t.strip()))
    
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
    
def get_itermax_predictions(raw_s_predictions, max_count=2, alpha_ratio=0.9):
    rows = len(raw_s_predictions)
    cols = len(raw_s_predictions[0])
    matrix = np.ndarray(shape=(rows, cols), dtype=float)

    for i in raw_s_predictions:
        for j, s in raw_s_predictions[i]:
            matrix[i,j] = s
    
    itermax_res = iter_max(matrix, max_count, alpha_ratio)
    res = []
    for i in range(rows):
        for j in range(cols):
            if itermax_res[i,j] != 0:
                res.append((i,j))
    
    return res


def get_itermax_predictions_a(source_preds, target_preds, s_tok_count, t_tok_count, iter_count, alignment_type):
    aligns = []
    for ind in range(iter_count): # itermax param
        for i in range(s_tok_count + 1):
            for j in range(t_tok_count + 1):
                if len(source_preds[i]) > 0 and source_preds[i][0][0] == j and len(target_preds[j]) > 0 and target_preds[j][0][0] == i:
                    aligns.append((i,j))

        for align in aligns:
            i,j = align
            if alignment_type == "one-to-one":
                remove_aligned_words(source_preds, target_preds, i , j)
            elif alignment_type == "one-to-many":
                source_preds[i].pop(0)
                target_preds[j].pop(0)
            else:
                raise ValueError(f"alignment_type '{alignment_type}' is not valid")
                    #print(f"sourcepre[i]: {source_preds[i")
    return aligns

def get_gdfa_mix_predictions(source_predictions, target_predictions, source_edition, target_edition, verse_alignments, verse_alignments_gdfa, verse_id, prev_res):
    s_lang, t_lang, s_edition, t_edition = align_reader.get_ordered_editions(source_edition, target_edition)

    file_path = align_reader.get_align_file_path(s_lang, t_lang)
    fwd_file_path = file_path[:file_path.rfind('.')] + ".fwd"
    rev_file_path = file_path[:file_path.rfind('.')] + ".rev"
    index_path = align_reader.get_index_file_path(s_lang, t_lang)

    utils.LOG.info(f"index_path: {index_path}")
    utils.LOG.info(f"s_edition: {s_edition}")
    utils.LOG.info(f"t_edition: {t_edition}")
    index = align_reader.read_index_file(index_path)[align_reader.edition_file_mapping[s_edition]][align_reader.edition_file_mapping[t_edition]][verse_id]

    fwd_aligns_raw = align_reader.read_alignment_file(fwd_file_path)[index]
    rev_aligns_raw = align_reader.read_alignment_file(rev_file_path)[index]

    old_inter_aligns = iutils.get_aligns(align_reader.edition_file_mapping[source_edition], align_reader.edition_file_mapping[target_edition], verse_alignments)
    gdfa_aligns = iutils.get_aligns(align_reader.edition_file_mapping[source_edition], align_reader.edition_file_mapping[target_edition], verse_alignments_gdfa)
    fwd_aligns = splitted_alignments(source_edition, target_edition, fwd_aligns_raw)
    rev_aligns = splitted_alignments(source_edition, target_edition, rev_aligns_raw)
   
    new_inter_aligns = []
    for i in source_predictions:
        j = source_predictions[i][0][0]
        if target_predictions[j][0][0] == i:
            new_inter_aligns.append((i,j))

    #print(f"old_inter: {old_inter_aligns}")
    #print(f"gdfa: {gdfa_aligns}")        
    #print(f"fwd_aligns: {fwd_aligns}")
    #print(f"rev_aligns: {rev_aligns}")        
    #print(f"new_inter_aligns: {new_inter_aligns}")

    res = {}
    res['orig_inter'] = old_inter_aligns[:]
    res['orig_gdfa'] = gdfa_aligns[:]

    res['gdfa_replace_inter'] = gdfa_aligns[:]
    for item in old_inter_aligns:
        if item in gdfa_aligns:
            res['gdfa_replace_inter'].remove(item)
    for item in new_inter_aligns:
        res['gdfa_replace_inter'].append(item)
    
    for item in prev_res:
        res[f"gdfa_add_{item}"] = gdfa_aligns[:]
        for align in prev_res[item]:
            if align not in res[f"gdfa_add_{item}"]:
                res[f"gdfa_add_{item}"].append(align)

        res[f"gdfa_intersect_{item}"] = gdfa_aligns[:]
        for align in gdfa_aligns:
            if align not in prev_res[item]:
                res[f"gdfa_intersect_{item}"].remove(align)
            
        res[f"gdfa_intersect2_{item}"] = gdfa_aligns[:]
        for align in gdfa_aligns:
            if align not in prev_res[item] and align not in old_inter_aligns:
                res[f"gdfa_intersect2_{item}"].remove(align)

    for item in new_inter_aligns:
        for it in fwd_aligns[:]:
            if item[0] == it[0] or item[1] == it[1]:
                fwd_aligns.remove(it)
        for it in rev_aligns[:]:
            if item[0] == it[0] or item[1] == it[1]:
                rev_aligns.remove(it)
    fwd_aligns.extend(new_inter_aligns)
    rev_aligns.extend(new_inter_aligns)
    fwd_aligns_str = " ".join([f"{x[0]}-{x[1]}" for x in fwd_aligns])
    rev_aligns_str = " ".join([f"{x[0]}-{x[1]}" for x in rev_aligns])

    #print(f"fwd_aligns_str: {fwd_aligns_str}")
    #print(f"rev_aligns_str: {rev_aligns_str}")
    res['gdfa_new_mix'] = list(grow_diag_final_and(len(source_predictions), len(target_predictions), fwd_aligns_str.strip(), rev_aligns_str.strip()))
    return res


def predict_alignments(algo, source_edition, target_edition, verse_id, verse_alignments, verse_alignments_gdfa):
    utils.LOG.info("pridicting user - item based alignments")
    
    raw_s_predictions = {}
    raw_t_predictions = {}
    res = {}

    #s_tokens = doc_retriever.retrieve_document(verse_id + "@" + align_reader.edition_file_mapping[source_edition]).split() # TODO remove me
    #t_tokens = doc_retriever.retrieve_document(verse_id + "@" + align_reader.edition_file_mapping[target_edition]).split()

    for i in range(algo.s_tok_count + 1):
        for j in range(algo.t_tok_count + 1):
            pred = algo.predict(source_edition + str(i), target_edition + str(j))

            setup_dict_entry(raw_s_predictions, i, [])
            setup_dict_entry(raw_t_predictions, j, [])

            raw_s_predictions[i].append((j, pred.est))
            raw_t_predictions[j].append((i, pred.est))
            #utils.LOG.info(f"{s_tokens[i]}, {t_tokens[j]}, {pred.est}")

    #target_predictions = normalize_scores(raw_s_predictions, raw_t_predictions)
    #source_predictions = normalize_scores(raw_t_predictions, raw_s_predictions)

    target_predictions = raw_t_predictions
    source_predictions = raw_s_predictions

    for i in range(algo.s_tok_count + 1):
        source_predictions[i].sort(key=lambda tup: tup[1], reverse=True)
    for i in range(algo.t_tok_count + 1):
        target_predictions[i].sort(key=lambda tup: tup[1], reverse=True)

    counts = [1,2,3,4]
    ratios = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    for c in counts:
        for ratio in ratios:
            res[f"itermax_c[{c}]_[ratio]{ratio}"] = get_itermax_predictions(raw_s_predictions, max_count=c, alpha_ratio=ratio)

    # TODO Remove Me later
    # old_inter_aligns = get_aligns(source_edition, target_edition, verse_alignments)
    # res['orig_inter'] = old_inter_aligns[:]

    res_gdfa = get_gdfa_predictions(source_predictions, target_predictions)
    res_gdfamix = get_gdfa_mix_predictions(source_predictions, target_predictions, source_edition, target_edition, verse_alignments, verse_alignments_gdfa, verse_id, res)
    res.update(res_gdfa)
    res.update(res_gdfamix)


    #itermax_predictions_a = {}
    #for alignment_type in ["one-to-one", "one-to-many"]:
    #    for iter_count in range(1,4):
    #        itermax_predictions_a[f"{alignment_type}_{iter_count}"] = get_itermax_predictions_a(source_predictions, target_predictions, iter_count, alignment_type)

    #for i in range(s_tok_count):
    #    aligns.append((i,source_predictions[i][0]))

    #for j in range(t_tok_count):
    #    if (target_predictions[j][0], j) not in aligns:
    #        aligns.append((target_predictions[j][0], j))
    return res

def make_all_res_set(res, res_gdfa):
    f_res = {}
    for r in res:
        f_res[f"init-inter_{r}"] = res[r]
    for r in res_gdfa:
        f_res[f"init-gdfa_{r}"] = res_gdfa[r]

def train_model(df, model_path, s_tok_count, t_tok_count, row_editions, col_editions):
    #algo = SVD()
    #algo = KNNBasic() #sim_options={'user_based':False})

    if False: #os.path.exists(model_path):
        utils.LOG.info(f"going to retrive model {model_path}")
        with open(model_path, 'rb') as inf:
            algo = pickle.load(inf)
    else:
        utils.LOG.info(f"going to train model {model_path}")
        algo = NMF()
        reader = Reader(rating_scale=(1, 3))
        data = Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
        trainset = data.build_full_trainset()
        algo.fit(trainset)

        algo.s_tok_count = s_tok_count
        algo.t_tok_count = t_tok_count
        algo.row_editions = row_editions
        algo.col_editions = col_editions
        algo.df = df
        with open(model_path, 'wb') as of:
            pickle.dump(algo, of)
    
    return algo

def update_res(res, res_new, prefix):
    for item in res_new:
        res[f"{prefix}_{item}"] = res_new[item]

def add_prefix_to_res(input, pref):
    res = {}
    for item in input:
        res[f"{pref}_{item}"] = input[item]

def get_induced_alignments(source_edition, target_edition, verse_id, verse_alignments, 
            verse_alignments_gdfa, edition_count, single_edition, align_reader, 
            cols_twice_rows, all_editions):

    # This does nothing if verse_alignments already exists
    verse_alignments = iutils.get_verse_alignments(verse_id, verse_alignments)
    verse_alignments_gdfa = iutils.get_verse_alignments(verse_id, verse_alignments_gdfa, True)

    # if verse_alignments == None or verse_alignments_gdfa == None:
    #     return {}

    # model_path = f"{models_dir}/{source_edition}_{target_edition}_{verse_id}_{edition_count}_{single_edition}_{cols_twice_rows}_algo"
    model_path_gdfa = f"{models_dir}/{base_align}_{source_edition}_{target_edition}_{verse_id}_{edition_count}_{single_edition}_{cols_twice_rows}_gdfa"
    # model_path_inde = f"{models_dir}/{source_edition}_{target_edition}_{verse_id}_{edition_count}_{single_edition}_{cols_twice_rows}_inde"

    df, df_gdfa, df_inde, s_tok_count, t_tok_count, row_editions, col_editions = None, None, None, None, None, None, None
    # if not ( os.path.exists(model_path) and os.path.exists(model_path_gdfa) and os.path.exists(model_path_inde) ):
    # if not ( os.path.exists(model_path_gdfa) ):
        
    ###  source -> row, target-> col###
    row_editions, col_editions = get_row_col_editions(source_edition, target_edition, verse_id, verse_alignments, edition_count, single_edition, align_reader, cols_twice_rows, all_editions)
    #itemid -> col, user -> row
    df, df_gdfa, df_inde, s_tok_count, t_tok_count = get_alignments_df(row_editions, col_editions, verse_alignments, verse_alignments_gdfa, source_edition, target_edition, verse_id, align_reader)


    # algo = train_model(df, model_path, s_tok_count, t_tok_count, row_editions, col_editions)
    algo_gdfa = train_model(df_gdfa, model_path_gdfa, s_tok_count, t_tok_count, row_editions, col_editions)
    # algo_inde = train_model(df_inde, model_path_inde, s_tok_count, t_tok_count, row_editions, col_editions)

    # res = predict_alignments(algo, source_edition, target_edition, verse_id, verse_alignments, verse_alignments_gdfa)
    res_gdfa = predict_alignments(algo_gdfa, source_edition, target_edition, verse_id, verse_alignments, verse_alignments_gdfa)
    # res_inde = predict_alignments(algo_inde, source_edition, target_edition, verse_id, verse_alignments, verse_alignments_gdfa)

    res = {}
    update_res(res, res_gdfa, "init_gdfa")
    # update_res(res, res_inde, "init_inde")
    add_prefix_to_res(res, f"{edition_count}_{single_edition}_{cols_twice_rows}")
    return res, len(algo_gdfa.row_editions), len(algo_gdfa.col_editions)

def load_simalign_editions(editions_file):
    editions = {}
    langs = []
    # langs = ['eng', 'fra']
    # editions['eng'] = 'eng-x-bible-mixed'
    # editions['fra'] = 'fra-x-bible-louissegond'
    with open(editions_file) as f_lang_list:
        lines = f_lang_list.read().splitlines()
        for line in lines[:]:
            comps = line.split('\t')
            editions[comps[0]] = comps[1] #.replace('-x-bible','')
            langs.append(comps[0])
    return editions, langs

def load_simalign_verse_alignments(alignment_path, verse_ids, editions, langs):
    verse_ids_set = set(verse_ids)

    all_verse_alignments_inter = {}
    all_verse_alignments_gdfa = {} # gdfa is actually itermax for simalign
    for no1, lang1 in enumerate(langs[:-1]):
        edition1 = editions[lang1]
        for no2, lang2 in enumerate(langs[no1+1:]):
            edition2 = editions[lang2]
            for m in ["itermax", "inter"]:
                if m == 'itermax':
                    verse_dict = all_verse_alignments_gdfa
                elif m == 'inter':
                    verse_dict = all_verse_alignments_inter

                # TODO implement this in a better way             
                alignment_file = f"{alignment_path}{lang1}-{lang2}.simalign.{m}"
                if not os.path.exists(alignment_file): # naming is different for helfi
                    alignment_file = f"{alignment_path}{lang1}_{lang2}_bpe.{m}"
                
                with open(alignment_file) as f_in:
                    lines = f_in.read().splitlines()
                    for line in lines:
                        comps = line.split('\t')
                        verse_id = comps[0]
                        
                        # get alignments for only verses in the gold data
                        if verse_id not in verse_ids_set:
                            continue

                        aligns_tmp = comps[1].split()
                        # remove probability info from alignments
                        aligns = ' '.join([f"{align.split('-')[0]}-{align.split('-')[1]}" for align in aligns_tmp])
                        
                        if verse_id not in verse_dict:
                            verse_dict[verse_id] = {}
                        
                        if edition1 not in verse_dict[verse_id]:
                            verse_dict[verse_id][edition1] = {}
                        
                        verse_dict[verse_id][edition1][edition2] = aligns

    utils.LOG.info(f"Created verse aligjments from {alignment_path}")
    return all_verse_alignments_inter, all_verse_alignments_gdfa

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

def calc_and_update_score(aligns, pros, surs, results):
    if len(aligns) == 0: return None

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

    results["prec"] = round(results["p_hit_count"] / max(results["total_hit_count"], 0.01), 3)
    results["rec"] = round(results["s_hit_count"] / results["gold_s_hit_count"], 3)
    results["f1"] = round(2. * results["prec"] * results["rec"] / max((results["prec"] + results["rec"]), 0.01), 3)
    results["aer"] = round(1 - (results["s_hit_count"] + results["p_hit_count"]) / (results["total_hit_count"] + results["gold_s_hit_count"]), 3)


def main(params):
    random.seed(params.seed)
    gold_name = params.gold_file.split("/")[-1].split('.')[0]
    args = []
    pros, surs = load_gold(params.gold_file)

    results_all = {}
    all_verses =list(pros.keys())
    # shuffle(all_verses)
    # all_verses = all_verses[:20]

    # Get editions whose simalign resutls are available
    editions, langs = load_simalign_editions(params.editions_file)
    all_editions = [editions[lang] for lang in langs]
    # utils.LOG.info(f"edition count: {len(all_editions)}")

    global base_align
    base_align = params.base_align_method

    if base_align == 'simalign':
        all_verse_alignments_inter, all_verse_alignments_gdfa = load_simalign_verse_alignments(params.alignments_path, all_verses, editions, langs)
    elif base_align == 'eflomal':   
        verse_alignments = None
        verse_alignments_gdfa = None

    for verse_id in all_verses:
        if base_align == 'simalign':
            verse_alignments = all_verse_alignments_inter[verse_id]
            verse_alignments_gdfa = all_verse_alignments_gdfa[verse_id]

        args.append((params.source_edition, params.target_edition, verse_id, verse_alignments, \
                    verse_alignments_gdfa, params.edition_count, params.single_edition, align_reader, 
                    params.cols_twice_rows, all_editions[:]))

    utils.LOG.info(f"Core count: {params.core_count}")
    with Pool(params.core_count) as p:  
        all_alignments = p.starmap(get_induced_alignments, args)
    
    row_counts = []
    column_counts = []
    for id, verse_id in enumerate(all_verses):
        res_aligns, row_count, column_count = all_alignments[id]
        row_counts.append(row_count)
        column_counts.append(column_count)

        for item in res_aligns:
            # print(item)
            setup_dict_entry(results_all, item, {"p_hit_count": 0, "s_hit_count": 0, "total_hit_count": 0, "gold_s_hit_count": 0, "prec": 0, "rec": 0, "f1": 0, "aer": 0})
            calc_and_update_score(res_aligns[item], pros[verse_id], surs[verse_id], results_all[item])
        
    out_file_name = f"{params.base_align_method}_source_{params.source_edition}_target_{params.target_edition}_max_{max(row_counts)-1}_rows_{max(column_counts)-1}_columns_editions_single_edition_{str(params.single_edition)}_{gold_name}"
    with open(os.path.join(params.save_path, out_file_name + '.txt'), 'w') as f_out:
    
        for i in results_all:
            f_out.write(f'result for : {i}\nPrecision: {results_all[i]["prec"]}\nRecall: {results_all[i]["rec"]}\nF1: {results_all[i]["f1"]}\nAER: {results_all[i]["aer"]}\nHits: {results_all[i]["total_hit_count"]}\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model_save_path', default= "/mounts/work/ayyoob/models/multi_parallel_align_induction", type=str)
    parser.add_argument('--save_path', default="/mounts/work/ayyoob/results/multi_parallel_align_induction/", type=str)
    parser.add_argument('--gold_file', default="/mounts/Users/student/ayyoob/Dokumente/code/pbc_utils/data/eng_fra_pbc/eng-fra-major.gold", type=str)    
    parser.add_argument('--base_align_method', default='eflomal', type=str, choices = ['eflomal', 'simalign'])
    parser.add_argument('--alignments_path', default=None, type=str, help="provide the path only for simalign alignments")
    parser.add_argument('--source_edition', default="eng-mixed", type=str) 
    parser.add_argument('--target_edition', default="fra-louissegond", type=str) 
    parser.add_argument('--edition_count', default=50, type=int)
    parser.add_argument('--editions_file', default=None, type=str)
    parser.add_argument('--single_edition', default=False, action='store_true')
    parser.add_argument('--cols_twice_rows', default=False, action='store_true')
    parser.add_argument('--core_count', default=1, type=int)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()
    
    models_dir = args.model_save_path

    main(args)
    
