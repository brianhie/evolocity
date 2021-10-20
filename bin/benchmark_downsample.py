import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy.stats as ss
import seaborn as sns
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
                        nan_policy='omit')[0], len(adata)
    

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
        return (ss.spearmanr(classes, pseudotimes, nan_policy='omit')[0],
                'spearmanr', len(adata))

    return auroc(classes, pseudotimes), 'auroc', len(adata)
    

if __name__ == '__main__':
    proteins = [
        'np',
        'h1',
        'gag',
        #'cov',
        'glo',
        'cyc',
        'eno',
        'pgk',
        'ser',
    ]

    settings = [
        'downsample',
        'wdownsample', # Weighted downsample.
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
                    value, n_samples = benchmark_temporal(
                        protein,
                        setting,
                        percentage,
                        temporal_benchmarks[protein],
                    )
                    data.append([
                        protein,
                        setting,
                        percentage,
                        n_samples,
                        'spearmanr',
                        value,
                    ])
                if protein in class_benchmarks:
                    value, score_type, n_samples = benchmark_class(
                        protein,
                        setting,
                        percentage,
                        class_benchmarks[protein]
                    )
                    data.append([
                        protein,
                        setting,
                        percentage,
                        n_samples,
                        score_type,
                        value,
                    ])
                if not setting:
                    setting = 'esm1b'

    df = pd.DataFrame(data, columns=[
        'protein',
        'setting',
        'percentage',
        'n_samples',
        'score_type',
        'value',
    ])

    df.to_csv('benchmark_downsample_results.txt', sep='\t')

    for setting in settings:
        # Seed the jitter.
        np.random.seed(1)
        random.seed(1)

        df_setting = df[df['setting'] == setting]
        df_setting['value'] = [
            value * 2 - 1 if score_type == 'auroc' else value
            for value, score_type in
            zip(df_setting['value'], df_setting['score_type'])
        ]
    
        plt.figure(figsize=(5, 6))
        sns.stripplot(
            x='percentage',
            y='value',
            hue='protein',
            data=df_setting,
        )
        sns.boxplot(
            showmeans=True,
            meanline=True,
            meanprops={'color': 'k', 'ls': '-', 'lw': 2},
            medianprops={'visible': False},
            whiskerprops={'visible': False},
            zorder=10,
            x='percentage',
            y='value',
            data=df_setting,
            showfliers=False,
            showbox=False,
            showcaps=False,
        )
        plt.axhline(0., color='#888888', linestyle='--')
        plt.ylim([ -1.09, 1.09 ])
        plt.savefig(f'figures/benchmark_{setting}.svg')
        plt.close()
