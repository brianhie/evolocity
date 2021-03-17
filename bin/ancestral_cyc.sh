mafft \
    --thread 40 \
    --auto \
    data/cyc/cyc_curated.fa \
    > target/cyc/cyc_curated_aligned.fa

python bin/fasta2phylip.py \
    target/cyc/cyc_curated_aligned.fa \
    target/cyc/cyc_curated_aligned.phy

phyml \
    -i target/cyc/cyc_curated_aligned.phy \
    -d aa -m JTT -c 4 -a e -b 0
