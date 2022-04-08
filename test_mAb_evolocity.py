import evolocity as evo
import numpy as np
import scanpy as sc

#/Users/gianhiltbrunner/Desktop/prot_bert_untuned

test_seqs = [
    'MKTVRQERLKSIVRILERSKEPV',
    'SGAQLAEELSVSRQVIVQDIA',
    'SGAQLAYNIVASRQVIVQDIA',
    'YLRSLGTPRGYVLAGG',
    'KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRS',
    'KALTARQQEIFDLIRDHISQTGMPPIVRAEIAQRLGFRS',
    'PNAAEEHLKALARKGVIEIVSGASR',
    'GIRLLQEE',
]

adata = evo.pp.featurize_seqs(test_seqs, model_name='protbert', model_path='/Users/gianhiltbrunner/Desktop/prot_networks/LY16')
evo.pp.neighbors(adata)
sc.tl.umap(adata)

evo.tl.velocity_graph(adata, model_name='protbert', model_path='/Users/gianhiltbrunner/Desktop/prot_networks/LY16')