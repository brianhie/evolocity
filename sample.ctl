      seqfile = target/cyc/cyc_curated_aligned.phy  * sequence data filename
     treefile = target/cyc/cyc_curated_aligned.phy_phyml_tree.txt   * tree structure file name
      outfile = target/cyc/cyc_curated_models.mlc    * main result file name

        noisy = 9  * 0,1,2,3,9: how much rubbish on the screen
      verbose = 2  * 0: concise; 1: detailed, 2: too much
      runmode = 0  * 0: user tree;  1: semi-automatic;  2: automatic
                   * 3: StepwiseAddition; (4,5):PerturbationNNI; -2: pairwise

      seqtype = 2  * 1:codons; 2:AAs; 3:codons-->AAs

        clock = 0  * 0:no clock, 1:clock; 2:local clock; 3:CombinedAnalysis
       aaDist = 0  * 0:equal, +:geometric; -:linear, 1-6:G1974,Miyata,c,p,v,a
   aaRatefile = ./data/paml/dat/jones.dat  * only used for aa seqs with model=empirical(_F)

                   * dayhoff.dat, jones.dat, wag.dat, mtmam.dat, or your own

        model = 2
                   * models for codons:
                       * 0:one, 1:b, 2:2 or more dN/dS ratios for branches
                   * models for AAs or codon-translated AAs:
                       * 0:poisson, 1:proportional, 2:Empirical, 3:Empirical+F
                       * 6:FromCodon, 7:AAClasses, 8:REVaa_0, 9:REVaa(nr=189)

        icode = 0  * 0:universal code; 1:mammalian mt; 2-10:see below
        Mgene = 0     * codon: 0:rates, 1:separate; 2:diff pi, 3:diff kapa, 4:all diff            
                   * AA: 0:rates, 1:separate

    fix_alpha = 0   * 0: estimate gamma shape parameter; 1: fix it at alpha
        alpha = 0.5 * initial or fixed alpha, 0:infinity (constant rate)
       Malpha = 1   * different alphas for genes
        ncatG = 4   * # of categories in dG of NSsites models
        getSE = 0  * 0: don't want them, 1: want S.E.s of estimates

 RateAncestor = 1  * (0,1,2): rates (alpha>0) or ancestral states (1 or 2)

   Small_Diff = .5e-6
    cleandata = 0  * remove sites with ambiguity data (1:yes, 0:no)?
       method = 1  * Optimization method 0: simultaneous; 1: one branch a time


* Genetic codes: 0:universal, 1:mammalian mt., 2:yeast mt., 3:mold mt.,
* 4: invertebrate mt., 5: ciliate nuclear, 6: echinoderm mt.,
* 7: euplotid mt., 8: alternative yeast nu. 9: ascidian mt.,
* 10: blepharisma nu.
* These codes correspond to transl_table 1 to 11 of GENEBANK.