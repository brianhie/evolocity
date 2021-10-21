###################
## Main analyses ##
###################

python bin/flu_np.py esm1b --evolocity > np_esm1b_evolocity.log 2>&1

python bin/flu.py esm1b --evolocity > h1_esm1b_evolocity.log 2>&1

python bin/gag.py esm1b --evolocity > gag_esm1b_evolocity.log 2>&1
python bin/gag.py esm1b --ancestral > gag_esm1b_ancestral.log 2>&1

python bin/cov.py esm1b --evolocity > cov_esm1b_evolocity.log 2>&1

python bin/cyc.py esm1b --evolocity > cyc_esm1b_evolocity.log 2>&1

python bin/glo.py esm1b --evolocity > glo_esm1b_evolocity.log 2>&1
python bin/glo.py esm1b --ancestral > glo_esm1b_ancestral.log 2>&1

python bin/pgk.py esm1b --evolocity > pgk_esm1b_evolocity.log 2>&1

python bin/eno.py esm1b --evolocity > eno_esm1b_evolocity.log 2>&1

python bin/ser.py esm1b --evolocity > ser_esm1b_evolocity.log 2>&1

#######################
## Benchmarking code ##
#######################

bash bin/benchmark.sh # Requires days of compute!

python bin/benchmark.py

bash bin/benchmark_downsample.sh # Requires days of compute!

python bin/benchmark_downsample.py
