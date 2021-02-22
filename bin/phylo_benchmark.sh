
declare -a proteins=("np" "h1" "cyc" "glo" "pgk" "eno")

for protein in ${proteins[@]}
do
    FastTree \
        target/evolocity_alignments/$protein/aligned.fasta \
        > target/evolocity_alignments/$protein/fasttree.tree \
        2> fasttree_$protein.log &
done

wait

for protein in ${proteins[@]}
do
    python bin/fasta2phylip.py \
           target/evolocity_alignments/$protein/aligned.fasta \
           target/evolocity_alignments/$protein/aligned.phylip

    raxml \
        -T 40 -D -F -m PROTCATBLOSUM62 \
        -s target/evolocity_alignments/$protein/aligned.phylip -f E \
        -n raxml_$protein.tree -p 1 \
        > raxml_$protein.log 2>&1

    mv RAxML_* target/evolocity_alignments/$protein/
done
