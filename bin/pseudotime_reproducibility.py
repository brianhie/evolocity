from utils import *

if __name__ == '__main__':
    prots = [
        'np',
        'h1',
        'gag',
        'cov',
        'cyc',
        'glo',
        'pgk',
        'eno',
        'ser'
    ]

    for prot in prots:
        fname_esm1b = f'target/ev_cache/{prot}_pseudotime.txt'
        fname_tape = f'target/ev_cache/{prot}_tape_pseudotime.txt'

        x_esm1b = np.loadtxt(fname_esm1b)
        x_tape = np.loadtxt(fname_tape)

        tprint(prot)
        tprint('Spearman r = {}, P = {}'.format(*ss.spearmanr(x_esm1b, x_tape)))
        tprint('')

        #if prot == 'np':
        #    fname_onehot = f'target/ev_cache/{prot}_onehot_pseudotime.txt'
        #    x_onehot = np.loadtxt(fname_onehot)
        #    tprint(prot + ' onehot')
        #    tprint('Spearman r = {}, P = {}'
        #           .format(*ss.spearmanr(x_esm1b, x_onehot)))
        #    tprint('')
        #
        #    fname_blosum = f'target/ev_cache/{prot}_blosum62_pseudotime.txt'
        #    x_blosum = np.loadtxt(fname_blosum)
        #    tprint(prot + ' blosum')
        #    tprint('Spearman r = {}, P = {}'
        #           .format(*ss.spearmanr(x_esm1b, x_blosum)))
        #    tprint('')
        #
        #    fname_blosum = f'target/ev_cache/{prot}_onehot_blosum62_pseudotime.txt'
        #    x_blosum = np.loadtxt(fname_blosum)
        #    tprint(prot + ' onehot and blosum')
        #    tprint('Spearman r = {}, P = {}'
        #           .format(*ss.spearmanr(x_esm1b, x_blosum)))
        #    tprint('')
