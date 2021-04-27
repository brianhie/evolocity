mafft \
    --thread 40 \
    --auto \
    data/eno/eno_curated.fa \
    > target/eno/eno_curated_aligned.fa

python bin/fasta2phylip.py \
    target/eno/eno_curated_aligned.fa \
    target/eno/eno_curated_aligned.phy

phyml \
    -i target/eno/eno_curated_aligned.phy \
    -d aa -m JTT -c 4 -a e -b 0
