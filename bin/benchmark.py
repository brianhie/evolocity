import anndata
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.metrics import roc_auc_score as auroc

def benchmark_temporal(protein, setting, label):
    namespace = protein
    if setting != 'base':
        namespace += f'_{setting}'

    adata_cache = f'target/ev_cache/{protein}_adata.h5ad'
    adata = anndata.read_h5ad(adata_cache)
    if 'homologous' in namespace:
        adata = adata[adata.obs['homology'] > 80.]
    with open(f'target/ev_cache/{namespace}_pseudotime.txt') as f:
        adata.obs['pseudotime'] = np.loadtxt(f)
    return ss.spearmanr(adata.obs[label], adata.obs['pseudotime'],
                        nan_policy='omit')[0]
    

def benchmark_class(protein, setting, labels):
    namespace = protein
    if setting != 'base':
        namespace += f'_{setting}'

    adata_cache = f'target/ev_cache/{protein}_adata.h5ad'
    adata = anndata.read_h5ad(adata_cache)
    if 'homologous' in namespace:
        adata = adata[adata.obs['homology'] > 80.]
    with open(f'target/ev_cache/{namespace}_pseudotime.txt') as f:
        adata.obs['pseudotime'] = np.loadtxt(f)

    class_name = labels[0]
    class_map = labels[1]

    classes, pseudotimes = [], []
    for class_, ptime in zip(adata.obs[class_name], adata.obs['pseudotime']):
        if class_ not in class_map:
            continue
        classes.append(class_map[class_])
        pseudotimes.append(ptime)

    if len(set(classes)) > 2:
        return ss.spearmanr(classes, pseudotimes, nan_policy='omit')[0]

    return auroc(classes, pseudotimes)
    

if __name__ == '__main__':
    proteins = [
        'np',
        'h1',
        'gag',
        'cov',
        'glo',
        'cyc',
        'eno',
        'pgk',
        'ser',
    ]

    settings = [
        'base', # ESM-1b for features and velocities.
        'homologous',
        'tape',
        'onehot',
        'blosum62',
        'jtt',
        'wag',
        'esm1b-rand',
        'edgerand',
        'unit',
    ]

    temporal_benchmarks = {
        'np': 'year',
        'h1': 'Collection Date',
        'cov': 'timestamp',
    }
    class_benchmarks = {
        'gag': ('subtype', {
            'B': 0, 'C': 0, 'BC': 1,
        }),
        'glo': ('globin_type', {
            'neuroglobin': 0, 'hemoglobin_beta': 1,
        }),
        'cyc': ('tax_group', {
            'eukaryota': 0,
            'viridiplantae': 1,
            'fungi': 2,
            'arthropoda': 3,
            'chordata': 4,
            'mammalia': 5,
        }),
        'eno': ('tax_kingdom', {
            'archaea': 0, 'eukaryota': 1,
        }),
        'pgk': ('tax_kingdom', {
            'archaea': 0, 'eukaryota': 1,
        }),
        'ser': ('tax_kingdom', {
            'archaea': 0, 'bacteria': 0, 'eukaryota': 1,
        }),
    }

    data = []
    for protein in proteins:
        for setting in settings:
            if setting == 'homologous' and protein in { 'np', 'cov' }:
                continue
            print(protein, setting)
            if protein in temporal_benchmarks:
                value = benchmark_temporal(
                    protein,
                    setting,
                    temporal_benchmarks[protein],
                )
                data.append([
                    protein, setting, 'spearmanr', value
                ])
            if protein in class_benchmarks:
                value = benchmark_class(
                    protein,
                    setting,
                    class_benchmarks[protein]
                )
                data.append([
                    protein, setting, 'auroc', value
                ])
            if not setting:
                setting = 'esm1b'

    df = pd.DataFrame(data, columns=[
        'protein', 'setting', 'score_type', 'value',
    ])

    df.to_csv('benchmark_results.txt', sep='\t')
