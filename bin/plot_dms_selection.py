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
                corr = float(line.split(',')[0].split()[-1])
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

    prot_dms = {
        'adrb2': [
            'DMS_0',
            'DMS_0.15',
            'DMS_0.625',
            'DMS_5',
        ],
        'bla': [
            'DMS_amp_0_(b)',
            'DMS_amp_10_(b)',
            'DMS_amp_39_(b)',
            'DMS_amp_156_(b)',
            'DMS_amp_625_(b)',
            'DMS_amp_2500_(b)',
        ],
        'gal4': [
            'DMS_nonsel_24',
            'DMS_selA_24',
            'DMS_selB_40',
            'DMS_selC_64',
        ],
        'gmr_therm': [
            'DMS_30C',
            'DMS_37C',
            'DMS_42C',
        ],
        'gmr_gm': [
            'DMS_37C',
            'DMS_37C_Gm_25mug/mL',
        ],
        'haeiiim': [
            'DMS_G3',
            'DMS_G7',
            'DMS_G17',
        ],
        'hras': [
            'DMS_g12v',
            'DMS_unregulated',
            'DMS_attenuated',
            'DMS_regulated',
        ],
        'mapk1': [
            'DMS_DOX',
            'DMS_SCH',
            'DMS_VRT',
        ],
        'ubi4': [
            'DMS_excess_(a)',
            'DMS_(a)',
            'DMS_limiting_(b)',
        ],
    }

    for idx, name in enumerate(prot_dms):
        protein = name.split('_')[0]
        order = prot_dms[name]

        df_protein = df[df['protein'] == 'dms_' + protein]
        
        plt.figure(figsize=(len(order) * 1.2, 4))
        sns.barplot(
            data=df_protein,
            x='dms',
            y='corr',
            hue='model_name',
            ci=None,
            order=order,
        )
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Spearman r')
        plt.axhline(0., color='#888888', linestyle='--')
        plt.savefig(f'figures/dms_selection_{name}.svg')
        plt.close()
        
