import anndata
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.metrics import roc_auc_score as auroc

def benchmark_temporal(protein, setting, percentage, label):
    namespace = protein
    if percentage != 100.:
        namespace += f'_{setting}{percentage}'

    adata_cache = f'target/ev_cache/{protein}_adata.h5ad'
    adata = anndata.read_h5ad(adata_cache)

    if percentage != 100.:
        with open(f'target/ev_cache/{namespace}_rand_idx.txt') as f:
            rand_idx = [ int(x) for x in f.read().rstrip().split() ]
        adata = adata[rand_idx]
    
    with open(f'target/ev_cache/{namespace}_pseudotime.txt') as f:
        adata.obs['pseudotime'] = np.loadtxt(f)
    return ss.spearmanr(adata.obs[label], adata.obs['pseudotime'],
                        nan_policy='omit')[0]
    

def benchmark_class(protein, setting, percentage, labels):
    namespace = protein
    if percentage != 100.:
        namespace += f'_{setting}{percentage}'

    adata_cache = f'target/ev_cache/{protein}_adata.h5ad'
    adata = anndata.read_h5ad(adata_cache)

    if percentage != 100.:
        with open(f'target/ev_cache/{namespace}_rand_idx.txt') as f:
            rand_idx = [ int(x) for x in f.read().rstrip().split() ]
        adata = adata[rand_idx]
    
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
        #'h1',
        #'gag',
        #'cov',
        #'glo',
        'cyc',
        #'eno',
        #'pgk',
        #'ser',
    ]

    settings = [
        'downsample',
        #'wdownsample', # Weighted downsample.
    ]

    percentages = [
        10.,
        25.,
        50.,
        75.,
        100.,
    ]

    # Below configuration should be same as benchmark.py.

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
            for percentage in percentages:
                print(protein, setting, percentage)
                if protein in temporal_benchmarks:
                    value = benchmark_temporal(
                        protein,
                        setting,
                        percentage,
                        temporal_benchmarks[protein],
                    )
                    data.append([
                        protein, setting, percentage, 'spearmanr', value
                    ])
                if protein in class_benchmarks:
                    value = benchmark_class(
                        protein,
                        setting,
                        percentage,
                        class_benchmarks[protein]
                    )
                    data.append([
                        protein, setting, percentage, 'auroc', value
                    ])
                if not setting:
                    setting = 'esm1b'

    df = pd.DataFrame(data, columns=[
        'protein', 'setting', 'percentage', 'score_type', 'value',
    ])

    df.to_csv('benchmark_downsample_results.txt', sep='\t')
