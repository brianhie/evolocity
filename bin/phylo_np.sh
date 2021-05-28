python bin/fasta2phylip.py \
       target/evolocity_alignments/np/aligned.fasta \
       target/evolocity_alignments/np/aligned.phylip

raxml -T 40 -D -F -m PROTCATBLOSUM62 \
      -s target/evolocity_alignments/np/aligned.phylip -f E \
      -n raxml_np -p 1 \
      > raxml_np.log 2>&1
mv RAxML_*raxml_np* target/evolocity_alignments/np/

FastTree -fastest \
         target/evolocity_alignments/np/aligned.fasta \
         > target/evolocity_alignments/np/fasttree.tree
