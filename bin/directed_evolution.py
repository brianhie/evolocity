from mutation import *
from evolocity_graph import *

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Directed evolution sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., esm1b, tape)')
    parser.add_argument('--namespace', type=str, default='de',
                        help='Model namespace')
    args = parser.parse_args()
    return args

def introduce_muts(wt_seq, muts, trajectory=None):
    mut_seq = [ c for c in wt_seq ]

    for mut in muts:
        if mut == '-':
            continue
        aa_orig, pos, aa_mut = mut[0], int(mut[1:-1]), mut[-1]
        if trajectory is not None:
            if trajectory == 'P450LA1 anti-Markovnikov oxidation of styrene':
                pos -= 1
            elif trajectory == 'Rma cyt c':
                pos += 20
        assert(mut_seq[pos] == aa_orig)
        mut_seq[pos] = aa_mut

    return ''.join(mut_seq)

def load_data():
    fname = 'data/directed_evolution/trajectories.txt'

    data, round_count = [], {}
    with open(fname) as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith('>'):
                traj_name, wt_seq = line[1:].rstrip().split('\t')
                round_count[traj_name] = []
                continue
            round_name, muts = line.rstrip().split('\t')
            muts = muts.replace(' ', '').rstrip(',').split(',')
            mut_seq = introduce_muts(wt_seq, muts, traj_name)
            round_number = len(round_count[traj_name])
            round_count[traj_name].append(round_name)
            data.append([ traj_name, wt_seq,
                          round_name, round_number,
                          muts, mut_seq ])

    df = pd.DataFrame(data, columns=[
        'trajectory', 'wildtype', 'round', 'round_num', 'muts', 'seq'
    ])

    return df

def directed_evolution(args, model, vocabulary):
    df_trajectories = load_data()

    for trajectory in set(df_trajectories['trajectory']):
        tprint(f'Trajectory: {trajectory}')
    
        df_curr = df_trajectories[
            df_trajectories['trajectory'] == trajectory
        ]
        nodes = [
            (str(round_num), seq)
            for round_num, seq in zip(df_curr['round_num'], df_curr['seq'])
        ]
    
        data = []
        for idx, (name, seq) in enumerate(nodes):
            if idx > 0:
                seq_prev = nodes[idx - 1][1]
                score = likelihood_muts(seq_prev, seq, args, vocabulary, model,)
                data.append([ name, seq, score ])
                tprint('Round {}: {}'.format(name, score))
    
        df = pd.DataFrame(data, columns=[ 'name', 'seq', 'score' ])
        tprint('Sum of scores: {}'.format(sum(df['score'])))
        tprint('')

if __name__ == '__main__':
    args = parse_args()

    namespace = args.namespace
    if args.model_name == 'tape':
        namespace += '_tape'

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
        raise ValueError('Invalid model {}'.format(args.model_name))

    directed_evolution(args, model, vocabulary)
