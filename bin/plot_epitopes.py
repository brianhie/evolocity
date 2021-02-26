from utils import *

if __name__ == '__main__':
    valid_fname = 'data/influenza/np_epitopes_iedb_consensus.txt'
    valid_ids = {}
    with open(valid_fname) as f:
        for line in f:
            fields = line.rstrip().split()
            valid_id = fields[0]
            n_pubs = int(fields[-2])
            if n_pubs < 4:
                continue
            valid_ids[valid_id] = n_pubs

    fname = 'data/influenza/np_epitopes_iedb.csv'
    positions = []
    with open(fname) as f:
        f.readline()
        for line in f:
            fields = line.rstrip().split(',')
            epitope_id = fields[0].split('/')[-1]
            if epitope_id in valid_ids:
                mult = valid_ids[epitope_id]
            else:
                continue
            try:
                start = int(fields[5])
                end = int(fields[6])
                positions += [ i for i in range(start, end + 1) ] * mult
            except:
                continue

    seq_start = 1
    seq_end = 498
    namespace = fname.split('/')[-1].split('.')[0]

    plt.figure(figsize=(10, 4))
    plt.hist(positions, bins=(seq_end - seq_start)*2)
    plt.xlim([ seq_start, seq_end ])
    for i in [ 104, 238, 373, 455, 480 ]:
        plt.axvline(x=i, c='maroon')
        print('Found {} publications at position {}'.format(
            positions.count(i), i
        ))
    plt.savefig(f'figures/epitopes_{namespace}.png', dpi=500)
    plt.close()
