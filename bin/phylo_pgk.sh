mafft \
    --thread 40 \
    --auto \
    data/pgk/pgk_curated.fa \
    > target/pgk/pgk_curated_aligned.fa

python bin/fasta2phylip.py \
    target/pgk/pgk_curated_aligned.fa \
    target/pgk/pgk_curated_aligned.phy

phyml \
    -i target/pgk/pgk_curated_aligned.phy \
    -d aa -m JTT -c 4 -a e -b 0
