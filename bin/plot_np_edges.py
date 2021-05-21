from utils import *

if __name__ == '__main__':
    fname = sys.argv[1]

    df = pd.read_csv(
        fname,
        delimiter='\t',
        names=[
            'node1',
            'node2',
            'year1',
            'year2',
            'mutscore'
        ],
    )

    df['diff'] = df['year2'] - df['year1']

    df = df[df['diff'] >= 0]
    df = df[np.isfinite(df['mutscore'])]

    plt.figure()
    plt.scatter(df['mutscore'], df['diff'], alpha=0.1)
    plt.axvline(0, c='r', linestyle='--')
    plt.axhline(0.5, c='#aaaaaa')
    plt.axhline(5.5, c='#aaaaaa')
    plt.axhline(20.5, c='#aaaaaa')
    plt.xlim([ -8.1, 8.1 ])
    plt.savefig('figures/np_edges.png', dpi=300)
    plt.close()

    for start, end in zip(
            [ 0, 1, 6,  21, ],
            [ 0, 5, 20, float('inf') ],
    ):
        df_interval = df[(df['diff'] >= start) &
                         (df['diff'] <= end)]
        n_pos = sum(df_interval['mutscore'] > 0)
        n_neg = sum(df_interval['mutscore'] < 0)
        binom_p = ss.binom_test([ n_pos, n_neg ])
        print(f'Year interval: {start}-{end}, {n_pos} pos, {n_neg} neg, '
              f'binom P = {binom_p}')
