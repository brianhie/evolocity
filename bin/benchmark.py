import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
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
        return (ss.spearmanr(classes, pseudotimes, nan_policy='omit')[0],
                'spearmanr')

    return auroc(classes, pseudotimes), 'auroc'
    

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
        'blosum62',
        'jtt',
        'wag',
        'onehot',
        'edgerand',
        'esm1b-rand',
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
            print(protein, setting)
            if setting == 'homologous' and protein in { 'np', 'cov' }:
                continue
            if protein in temporal_benchmarks:
                value = benchmark_temporal(
                    protein,
                    setting,
                    temporal_benchmarks[protein],
                )
                data.append([
                    protein, setting, 'spearmanr', value
                ])
                if protein in { 'np', 'cov' } and setting == 'base':
                    data.append([
                        protein, 'homologous', 'spearmanr', value
                    ])
            if protein in class_benchmarks:
                value, score_type = benchmark_class(
                    protein,
                    setting,
                    class_benchmarks[protein]
                )
                data.append([
                    protein, setting, score_type, value
                ])

    df = pd.DataFrame(data, columns=[
        'protein', 'setting', 'score_type', 'value',
    ])

    df.to_csv('benchmark_results.txt', sep='\t')

    df['value'] = [
        value * 2 - 1 if score_type == 'auroc' else value
        for value, score_type in zip(df['value'], df['score_type'])
    ]

    plt.figure(figsize=(5, 6))
    sns.stripplot(
        x='setting',
        y='value',
        hue='protein',
        data=df,
        order=settings,
    )
    sns.boxplot(
        showmeans=True,
        meanline=True,
        meanprops={'color': 'k', 'ls': '-', 'lw': 2},
        medianprops={'visible': False},
        whiskerprops={'visible': False},
        zorder=10,
        x='setting',
        y='value',
        data=df,
        showfliers=False,
        showbox=False,
        showcaps=False,
        order=settings,
    )
    plt.axhline(0., color='#888888', linestyle='--')
    plt.savefig('figures/benchmark.svg')
    plt.close()
