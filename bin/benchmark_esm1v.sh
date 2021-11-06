declare -a models=(1 2 3 4 5)

for model in ${models[@]}
do
    python bin/flu_np.py esm1v"$model" --evolocity \
           > np_esm1v"$model"_evolocity.log 2>&1
    
    python bin/flu.py esm1v"$model" --evolocity \
           > h1_esm1v"$model"_evolocity.log 2>&1
    
    python bin/gag.py esm1v"$model" --evolocity \
           > gag_esm1v"$model"_evolocity.log 2>&1
    
    python bin/cov.py esm1v"$model" --evolocity \
           > cov_esm1v"$model"_evolocity.log 2>&1
    
    python bin/cyc.py esm1v"$model" --evolocity \
           > cyc_esm1v"$model"_evolocity.log 2>&1
    
    python bin/glo.py esm1v"$model" --evolocity \
           > glo_esm1v"$model"_evolocity.log 2>&1
    
    python bin/pgk.py esm1v"$model" --evolocity \
           > pgk_esm1v"$model"_evolocity.log 2>&1
    
    python bin/eno.py esm1v"$model" --evolocity \
           > eno_esm1v"$model"_evolocity.log 2>&1
    
    python bin/ser.py esm1v"$model" --evolocity \
           > ser_esm1v"$model"_evolocity.log 2>&1
done
