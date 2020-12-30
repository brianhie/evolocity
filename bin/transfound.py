from utils import *

def load_keele2008():
    seqs = {}

    with open('data/hiv/transfound_keele2008/sequence.gb') as f:
        for line in f:
            if '##HIVDataBaseData-START##' in line:
                meta = {}
                line = f.readline()
                while line:
                    if '##HIVDataBaseData-END##' in line:
                        break
                    key, val = line.rstrip().split('::')
                    key, val = key.strip(), val.strip()
                    key = key.replace(' ', '_').lower()
                    if key == 'sample_date':
                        date = dparse(val)
                        meta[key] = str(date)
                        meta['year'] = int(date.year)
                    else:
                        meta[key] = val
                    line = f.readline()
                meta['country'] = 'USA'
                meta['corpus'] = 'keele2008'
                if 'fiebig_stage' in meta:
                    meta['status'] = meta['fiebig_stage']
                elif 'patient_health_status' in meta:
                    meta['status'] = meta['patient_health_status']
                else:
                    assert(False)

            if '/product="envelope glycoprotein"' in line:
                f.readline()
                line = f.readline()
                seq = line.rstrip().split('"')[1]
                line = f.readline()
                while line:
                    line = line.strip()
                    seq += line
                    if line.endswith('"'):
                        break
                    line = f.readline()

                seq = seq.replace('"', '')
                if seq not in seqs:
                    seqs[seq] = []
                seqs[seq].append(meta)

    return seqs

if __name__ == '__main__':
    load_keele2008()
