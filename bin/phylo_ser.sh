mafft \
    --thread 40 \
    --auto \
    data/ser/ser_curated.fa \
    > target/ser/ser_curated_aligned.fa

python bin/fasta2phylip.py \
    target/ser/ser_curated_aligned.fa \
    target/ser/ser_curated_aligned.phy

phyml \
    -i target/ser/ser_curated_aligned.phy \
    -d aa -m JTT -c 4 -a e -b 0
