import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy.stats as ss
import seaborn as sns
from sklearn.metrics import roc_auc_score as auroc

def benchmark_temporal(protein, model, label):
    namespace = protein
    if model != 'base':
        namespace += '_' + model

    adata_cache = f'target/ev_cache/{protein}_adata.h5ad'
    adata = anndata.read_h5ad(adata_cache)

    with open(f'target/ev_cache/{namespace}_pseudotime.txt') as f:
        adata.obs['pseudotime'] = np.loadtxt(f)
    return ss.spearmanr(adata.obs[label], adata.obs['pseudotime'],
                        nan_policy='omit')[0]
    

def benchmark_class(protein, model, labels):
    namespace = protein
    if model != 'base':
        namespace += '_' + model

    adata_cache = f'target/ev_cache/{protein}_adata.h5ad'
    adata = anndata.read_h5ad(adata_cache)

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

    models = [
        'base',
        'esm1v1',
        'esm1v2',
        'esm1v3',
        'esm1v4',
        'esm1v5',
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
        for model in models:
            print(protein, model)
            if protein in temporal_benchmarks:
                value = benchmark_temporal(
                    protein,
                    model,
                    temporal_benchmarks[protein],
                )
                data.append([
                    protein,
                    model,
                    'spearmanr',
                    value,
                ])
                if model != 'base':
                    data.append([
                        protein,
                        'esm1v',
                        'spearmanr',
                        value,
                    ])
            if protein in class_benchmarks:
                value, score_type = benchmark_class(
                    protein,
                    model,
                    class_benchmarks[protein]
                )
                data.append([
                    protein,
                    model,
                    score_type,
                    value,
                ])
                if model != 'base':
                    data.append([
                        protein,
                        'esm1v',
                        score_type,
                        value,
                    ])

    df = pd.DataFrame(data, columns=[
        'protein',
        'model',
        'score_type',
        'value',
    ])

    df.to_csv('benchmark_esm1v_results.txt', sep='\t')

    df['value'] = [
        value * 2 - 1 if score_type == 'auroc' else value
        for value, score_type in zip(df['value'], df['score_type'])
    ]

    # Seed the jitter.
    np.random.seed(1)
    random.seed(1)

    plt.figure(figsize=(5, 6))
    sns.stripplot(
        x='model',
        y='value',
        hue='protein',
        data=df[df['score_type'] == 'spearmanr'],
        order=models + [ 'esm1v' ],
        size=6,
    )
    sns.stripplot(
        x='model',
        y='value',
        hue='protein',
        data=df[df['score_type'] == 'auroc'],
        order=models + [ 'esm1v' ],
        marker='X',
        palette='husl',
        size=6,
    )
    sns.boxplot(
        showmeans=True,
        meanline=True,
        meanprops={'color': 'k', 'ls': '-', 'lw': 2},
        medianprops={'visible': False},
        whiskerprops={'visible': False},
        zorder=10,
        x='model',
        y='value',
        data=df,
        showfliers=False,
        showbox=False,
        showcaps=False,
        order=models + [ 'esm1v' ],
    )
    plt.axhline(0., color='#888888', linestyle='--')
    plt.ylim([ -1.09, 1.09 ])
    plt.savefig(f'figures/benchmark_esm1v.svg')
    plt.close()
