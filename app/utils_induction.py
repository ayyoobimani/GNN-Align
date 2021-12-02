from app import utils
import pickle, os

def get_verse_alignments(verse_id, verse_alignments=None, gdfa=False):
    utils.LOG.info(f"reading verse alignment file {verse_id}")
    if verse_alignments != None:
        return verse_alignments
    
    f_path = utils.AlignInduction.verse_alignments_path + f"/{verse_id}"
    if gdfa:
        f_path += "_gdfa.txt"
    else:
        f_path += "_inter.txt"
    f_path_bin = f_path + ".bin"

    if not os.path.exists(f_path):
        utils.LOG.info(f_path)
        utils.LOG.info(f"=================================={verse_id} dos not exist==================================")
        return None

    if os.path.exists(f_path_bin):
        with open(f_path_bin, 'rb') as inf:
            try:
                return pickle.load(inf)
            except:
                pass
    
    res = {}

    with open(f_path, 'r') as f:
        for line in f:
            s_file, t_file, aligns = tuple(line.split('\t'))
            utils.setup_dict_entry(res, s_file, {})
            res[s_file][t_file] = aligns
    
    with open(f_path_bin, 'wb') as of:
        pickle.dump(res, of)

    return res



def get_aligns(rf, cf, alignments):
    # utils.LOG.info(f"getting aligns for {re}, {ce}")
    raw_align = ''
    
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
        return None
    
    return res