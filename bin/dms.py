from mutation import *
from evolocity_graph import *

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='DMS sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    args = parser.parse_args()
    return args

def dms_results(fname, args, model, vocabulary):
    with open(fname):
        df = pd.read_csv(fname, delimiter=',')

    wt_seq = ''
    for mut in df['variant']:
        aa_orig, pos = mut[0], int(mut[1:-1]) - 1
        if len(wt_seq) > pos:
            assert(wt_seq[pos] == aa_orig)
        else:
            wt_seq += aa_orig

    y_pred = predict_sequence_prob(
        args, wt_seq, vocabulary, model, verbose=False,
    )
    seq_cache = {}
    seq_cache[wt_seq] = np.array([
        y_pred[i + 1, (
            vocabulary[wt_seq[i]]
            if wt_seq[i] in vocabulary else
            model.unk_idx_
        )] for i in range(len(wt_seq))
    ])

    scores_pred = []
    for mut in df['variant']:
        pos, aa_mut = int(mut[1:-1]) - 1, mut[-1]
        seq_mut = wt_seq[:pos] + aa_mut + wt_seq[(pos + 1):]
        if aa_mut == wt_seq[pos]:
            scores_pred.append(0)
        else:
            scores_pred.append(
                likelihood_muts(wt_seq, seq_mut, args, vocabulary, model,
                                seq_cache=seq_cache)
            )
    scores_pred = np.array(scores_pred)

    tprint(f'Results for {fname}:')

    for column in df.columns:
        if column.startswith('DMS'):
            tprint(f'\t{column}-{args.model_name}:')
            scores_dms = np.array(df[column].values)
            tprint('\t\tSpearman r = {}, P = {}'.format(
                *ss.spearmanr(scores_pred, scores_dms, nan_policy='omit')
            ))

            tprint(f'\t{column}-DeepSequence:')
            if 'DeepSequence' not in df or \
               df['DeepSequence'].isnull().values.all(axis=0):
                tprint('\t\tNo DeepSequence results')
            else:
                scores_deepseq = np.array(df['DeepSequence'].values)
                scores_dms = np.array(df[column].values)
                tprint('\t\tSpearman r = {}, P = {}'.format(
                    *ss.spearmanr(scores_deepseq, scores_dms, nan_policy='omit')
                ))

if __name__ == '__main__':
    args = parse_args()

    model = get_model(args, -1, -1)

    if 'esm' in args.model_name:
        vocabulary = { tok: model.alphabet_.tok_to_idx[tok]
                       for tok in model.alphabet_.tok_to_idx
                       if '<' not in tok }
        args.checkpoint = args.model_name
    elif args.model_name == 'tape':
        vocabulary = { tok: model.alphabet_[tok]
                       for tok in model.alphabet_ if '<' not in tok }
        args.checkpoint = args.model_name
    else:
        raise NotImplementedError(f'Invalid model {args.model_name}')

    fnames = [
        'data/dms/dms_adrb2.csv',
        'data/dms/dms_bla.csv',
        'data/dms/dms_brca1.csv',
        'data/dms/dms_calm1.csv',
        'data/dms/dms_cas9.csv',
        'data/dms/dms_ccdb.csv',
        'data/dms/dms_env.csv',
        'data/dms/dms_gal4.csv',
        'data/dms/dms_gmr.csv',
        'data/dms/dms_ha_h1.csv',
        'data/dms/dms_ha_h3.csv',
        'data/dms/dms_haeiiim.csv',
        'data/dms/dms_hras.csv',
        'data/dms/dms_hsp82.csv',
        'data/dms/dms_infa.csv',
        'data/dms/dms_mapk1.csv',
        'data/dms/dms_p53.csv',
        'data/dms/dms_pab1.csv',
        'data/dms/dms_pten.csv',
        'data/dms/dms_sumo1.csv',
        'data/dms/dms_tpk1.csv',
        'data/dms/dms_tpmt.csv',
        'data/dms/dms_ube2i.csv',
        'data/dms/dms_ubi4.csv',
    ]

    for fname in fnames:
        dms_results(fname, args, model, vocabulary)
