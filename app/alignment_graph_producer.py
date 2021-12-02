from app.document_retrieval import DocumentRetriever
from app.general_align_reader import GeneralAlignReader
import app.utils_induction as iutils
import networkx as nx
from app import utils
import matplotlib.pyplot as plt
#use pyvis


iutils.get_verse_alignments
doc_retriever = DocumentRetriever()
align_reader = GeneralAlignReader()

colors = {'eng-x-bible-newworld2013':'green', 'pes-x-bible-newworld':'blue', 'arb-x-bible-newworld':'red', 'deu-x-bible-newworld':'yellow', 'fra-x-bible-newworld':'orange', 'afr-x-bible-newworld':'brown'}
language_files = ['eng-x-bible-newworld2013', 'pes-x-bible-newworld', 'arb-x-bible-newworld', 'deu-x-bible-newworld', 'fra-x-bible-newworld', 'afr-x-bible-newworld']
verse = '41006040'

def add_nodes(verse, graph, lf, lang, tokens):
    utils.LOG.info(f"Adding nodes of verse {verse} in lang {lf}")
    nodes = doc_retriever.retrieve_document(f'{verse}@{lf1}').strip().split()
    graph.add_nodes_from([(f"{lang}:{node}", {'color':colors[lf]}) for node in nodes])

    tokens[lf] = nodes

def add_edges(graph, aligns, tokens, lf1, lf2, l1, l2):
    utils.LOG.info(f"Adding edges for {lf1} and {lf2}")
    l1ts = tokens[lf1]
    l2ts = tokens[lf2]
    graph.add_edges_from([(f'{l1}:{l1ts[i[0]]}',f'{l2}:{l2ts[i[1]]}') for i in aligns])


if __name__ == "__main__":
    
    aligns_inter = iutils.get_verse_alignments(verse_id=verse, gdfa=False)
    #aligns_gdfa = iutils.get_verse_alignments(verse_id=verse)
    Gi = nx.Graph()
    

    tokens = {}
    for i, lf1 in enumerate(language_files):
        add_nodes(verse, Gi, lf1, align_reader.file_lang_mapping[lf1], tokens)
    
    for i, lf1 in enumerate(language_files):
        l1 = align_reader.file_lang_mapping[lf1]
        for lf2 in language_files[i+1:]:
            l2 = align_reader.file_lang_mapping[lf2]
            aligns = iutils.get_aligns(lf1, lf2, aligns_inter)
            add_edges(Gi, aligns, tokens, lf1, lf2, l1, l2)
    
    colored_dict = nx.get_node_attributes(Gi, 'color')
    color_seq = [colored_dict.get(node) for node in Gi.nodes()]

    plt.subplot(111)
    nx.draw(Gi, with_labels=True, font_weight='bold', node_color=color_seq)
    #nx.draw_shell(Gi, with_labels=True, font_weight='bold')
    plt.show()



