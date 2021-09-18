declare -a scores=("blosum62" "jtt" "wag" "unit")

#for score in ${scores[@]}
#do
#    nice -n10 python bin/flu_np.py esm1b --evolocity --velocity-score $score \
#         > np_"$score"_evolocity.log 2>&1
#    
#    nice -n10 python bin/flu.py esm1b --evolocity --velocity-score $score \
#         > h1_"$score"_evolocity.log 2>&1
#
#    nice -n10 python bin/gag.py esm1b --evolocity --velocity-score $score \
#         > gag_"$score"_evolocity.log 2>&1
#
#    #nice -n10 python bin/cov.py esm1b --evolocity --velocity-score $score \
#        #> cov_"$score"_evolocity.log 2>&1
#
#    nice -n10 python bin/cyc.py esm1b --evolocity --velocity-score $score \
#         > cyc_"$score"_evolocity.log 2>&1
#
#    nice -n10 python bin/glo.py esm1b --evolocity --velocity-score $score \
#         > glo_"$score"_evolocity.log 2>&1
#
#    nice -n10 python bin/pgk.py esm1b --evolocity --velocity-score $score \
#         > pgk_"$score"_evolocity.log 2>&1
#
#    nice -n10 python bin/eno.py esm1b --evolocity --velocity-score $score \
#         > eno_"$score"_evolocity.log 2>&1
#
#    nice -n10 python bin/ser.py esm1b --evolocity --velocity-score $score \
#         > ser_"$score"_evolocity.log 2>&1
#done

nice -n10 python bin/flu_np.py esm1b-rand --evolocity \
     > np_random_evolocity.log 2>&1

nice -n10 python bin/flu.py esm1b-rand --evolocity \
     > h1_random_evolocity.log 2>&1

nice -n10 python bin/gag.py esm1b-rand --evolocity \
     > gag_random_evolocity.log 2>&1

#nice -n10 python bin/cov.py esm1b-rand --evolocity \
    #> cov_random_evolocity.log 2>&1

#nice -n10 python bin/cyc.py esm1b-rand --evolocity \
#     > cyc_random_evolocity.log 2>&1

nice -n10 python bin/glo.py esm1b-rand --evolocity \
     > glo_random_evolocity.log 2>&1

nice -n10 python bin/pgk.py esm1b-rand --evolocity \
     > pgk_random_evolocity.log 2>&1

nice -n10 python bin/eno.py esm1b-rand --evolocity \
     > eno_random_evolocity.log 2>&1

nice -n10 python bin/ser.py esm1b-rand --evolocity \
     > ser_random_evolocity.log 2>&1
