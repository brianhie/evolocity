from utils import *

def load_model(model_name):
    data = []
    with open(f'target/dms/dms_{model_name}.log') as f:
        for line in f:
            line = line.rstrip().split(' | ')[-1]

            if line.startswith('Results for '):
                curr_prot = line.split('/')[-1].split('.')[0]
                continue

            if line.startswith('\tDMS'):
                line = line.rstrip(':').strip()
                dms_name, model_name = line.split('-')
                continue

            if line.startswith('\t\tSpearman r = '):
                corr = abs(float(line.split(',')[0].split()[-1]))
                p = float(line.split()[-1])
                data.append([ curr_prot, dms_name, model_name, corr, p ])

    return data

if __name__ == '__main__':
    data = []
    data += load_model('esm1b')
    data += load_model('tape')

    df = pd.DataFrame(data, columns=[
        'protein',
        'dms',
        'model_name',
        'corr',
        'pval',
    ])

    plt.figure(figsize=(12, 5))
    sns.barplot(
        data=df,
        x='protein',
        y='corr',
        hue='model_name',
        ci=None,
    )
    sns.stripplot(
        data=df,
        x='protein',
        y='corr',
        hue='model_name',
        dodge=True,
    )
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('|Spearman r|')
    plt.tight_layout()
    plt.savefig('figures/plot_dms.svg')
    plt.close()

    print(df.to_csv())
