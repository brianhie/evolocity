from collections import Counter
from Bio import SeqIO
import sys

in_sequence, in_member = False, False
n_seqs = 0

def load_taxonomy():
    tax_fnames = [
        'data/cyc/taxonomy_archaea.tab.gz',
        'data/cyc/taxonomy_bacteria.tab.gz',
        'data/cyc/taxonomy_eukaryota.tab.gz',
        'data/cyc/taxonomy_unclassified.tab.gz',
        'data/cyc/taxonomy_viruses.tab.gz',
    ]

    import gzip

    taxonomy = {}

    for fname in tax_fnames:
        with gzip.open(fname) as f:
            header = f.readline().decode('utf-8').rstrip().split('\t')
            assert(header[0] == 'Taxon' and header[8] == 'Lineage')
            for line in f:
                fields = line.decode('utf-8').rstrip().split('\t')
                tax_id = fields[0]
                lineage = fields[8]
                taxonomy[tax_id] = lineage

    return taxonomy

if __name__ == '__main__':
    taxonomy = load_taxonomy()

    lineages = []
    with open('data/uniref/uniref50.xml') as f:
        for line_num, line in enumerate(f):
            if line.startswith('<representativeMember>'):
                in_member = True
    
            if line.startswith('</representativeMember>'):
                in_member = False
    
            if in_member:
                if '"NCBI taxonomy"' in line:
                    tax_id = line.split('"')[-2]
                    if tax_id in taxonomy:
                        lineages.append(taxonomy[tax_id])
                    else:
                        lineages.append('not_found')

    groups, kingdoms = [], []
    for lineage in lineages:
        if 'Archaea' in lineage:
            tax_group = 'archaea'
            tax_kingdom = 'archaea'
        if 'Bacteria' in lineage:
            tax_group = 'bacteria'
            tax_kingdom = 'bacteria'
        if 'Eukaryota' in lineage:
            tax_group = 'eukaryota'
            tax_kingdom = 'eukaryota'
        if 'Fungi' in lineage:
            tax_group = 'fungi'
            tax_kingdom = 'eukaryota'
        if 'Viridiplantae' in lineage:
            tax_group = 'viridiplantae'
            tax_kingdom = 'eukaryota'
        if 'Arthropoda' in lineage:
            tax_group = 'arthropoda'
            tax_kingdom = 'eukaryota'
        if 'Chordata' in lineage:
            tax_group = 'chordata'
            tax_kingdom = 'eukaryota'
        if 'Mammalia' in lineage:
            tax_group = 'mammalia'
            tax_kingdom = 'eukaryota'
        if 'Primate' in lineage:
            tax_group = 'primate'
            tax_kingdom = 'eukaryota'
        if 'unclassified sequences;' in lineage:
            tax_group = 'unclassified'
            tax_kingdom = 'other'
        if 'metagenomes;' in lineage:
            tax_group = 'metagenome'
            tax_kingdom = 'other'

        groups.append(tax_group)
        kingdoms.append(tax_kingdom)

    print('Taxonomic groups:')
    for tax_group, count in Counter(groups).most_common():
        print(f'{tax_group}: {count}')

    print('\n\nTaxonomic kingdoms:')
    for tax_kingdom, count in Counter(kingdoms).most_common():
        print(f'{tax_kingdom}: {count}')
