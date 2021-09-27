####################
## TAPE benchmark ##
####################

python bin/flu_np.py tape --evolocity > np_tape_evolocity.log 2>&1

python bin/flu.py tape --evolocity > h1_tape_evolocity.log 2>&1

python bin/gag.py tape --evolocity > gag_tape_evolocity.log 2>&1

python bin/cov.py tape --evolocity > cov_tape_evolocity.log 2>&1

python bin/cyc.py tape --evolocity > cyc_tape_evolocity.log 2>&1

python bin/glo.py tape --evolocity > glo_tape_evolocity.log 2>&1

python bin/pgk.py tape --evolocity > pgk_tape_evolocity.log 2>&1

python bin/eno.py tape --evolocity > eno_tape_evolocity.log 2>&1

python bin/ser.py tape --evolocity > ser_tape_evolocity.log 2>&1


############################################
## Different velocity functions benchmark ##
############################################

declare -a scores=("blosum62" "jtt" "wag" "unit")

for score in ${scores[@]}
do
    python bin/flu_np.py esm1b --evolocity --velocity-score $score \
         > np_"$score"_evolocity.log 2>&1

    python bin/flu.py esm1b --evolocity --velocity-score $score \
         > h1_"$score"_evolocity.log 2>&1

    python bin/gag.py esm1b --evolocity --velocity-score $score \
         > gag_"$score"_evolocity.log 2>&1

    python bin/cov.py esm1b --evolocity --velocity-score $score \
         > cov_"$score"_evolocity.log 2>&1

    python bin/cyc.py esm1b --evolocity --velocity-score $score \
         > cyc_"$score"_evolocity.log 2>&1
    
    python bin/glo.py esm1b --evolocity --velocity-score $score \
         > glo_"$score"_evolocity.log 2>&1
    
    python bin/pgk.py esm1b --evolocity --velocity-score $score \
         > pgk_"$score"_evolocity.log 2>&1
    
    python bin/eno.py esm1b --evolocity --velocity-score $score \
         > eno_"$score"_evolocity.log 2>&1
    
    python bin/ser.py esm1b --evolocity --velocity-score $score \
         > ser_"$score"_evolocity.log 2>&1
done

###########################################
## Randomly-initialized ESM-1b benchmark ##
###########################################

python bin/flu_np.py esm1b-rand --evolocity \
     > np_random_evolocity.log 2>&1

python bin/flu.py esm1b-rand --evolocity \
     > ha_random_evolocity.log 2>&1

python bin/gag.py esm1b-rand --evolocity \
     > gag_random_evolocity.log 2>&1

python bin/cov.py esm1b-rand --evolocity \
    > cov_random_evolocity.log 2>&1

python bin/cyc.py esm1b-rand --evolocity \
     > cyc_random_evolocity.log 2>&1

python bin/glo.py esm1b-rand --evolocity \
     > glo_random_evolocity.log 2>&1

python bin/pgk.py esm1b-rand --evolocity \
     > pgk_random_evolocity.log 2>&1

python bin/eno.py esm1b-rand --evolocity \
     > eno_random_evolocity.log 2>&1

python bin/ser.py esm1b-rand --evolocity \
     > ser_random_evolocity.log 2>&1
