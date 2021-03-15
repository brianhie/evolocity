mafft \
    --thread 40 \
    --auto \
    data/cyc/uniprot_model_organisms.fasta \
    > target/cyc/uniprot_models_aligned.fa

python bin/fasta2phylip.py \
    target/cyc/uniprot_models_aligned.fa \
    target/cyc/uniprot_models_aligned.phy

phyml \
    -i target/cyc/uniprot_models_aligned.phy \
    -d aa -m JTT -c 4 -a e -b 0
