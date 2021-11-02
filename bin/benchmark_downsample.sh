declare -a percentages=(75 50 25 10)

# Uniform downsampling benchmarks.
for percentage in ${percentages[@]}
do
    for seed in {2..2}
    do
        CUDA_VISIBLE_DEVICES=0 python bin/flu_np.py esm1b --evolocity --downsample $percentage \
               --seed $seed \
               > np_esm1b_downsample"$percentage".log 2>&1 &
        
        CUDA_VISIBLE_DEVICES=1 python bin/flu.py esm1b --evolocity --downsample $percentage \
               --seed $seed \
               > h1_esm1b_downsample"$percentage".log 2>&1 &
        
        CUDA_VISIBLE_DEVICES=2 python bin/gag.py esm1b --evolocity --downsample $percentage \
               --seed $seed \
               > gag_esm1b_downsample"$percentage".log 2>&1 &
        
        #CUDA_VISIBLE_DEVICES=3 python bin/cov.py esm1b --evolocity --downsample $percentage \
        #       --seed $seed \
        #       > cov_esm1b_downsample"$percentage".log 2>&1 &
        
        CUDA_VISIBLE_DEVICES=3 python bin/cyc.py esm1b --evolocity --downsample $percentage \
               --seed $seed \
               > cyc_esm1b_downsample"$percentage".log 2>&1 &
        
        CUDA_VISIBLE_DEVICES=0 python bin/glo.py esm1b --evolocity --downsample $percentage \
               --seed $seed \
               > glo_esm1b_downsample"$percentage".log 2>&1 &
        
        CUDA_VISIBLE_DEVICES=1 python bin/pgk.py esm1b --evolocity --downsample $percentage \
               --seed $seed \
               > pgk_esm1b_downsample"$percentage".log 2>&1 &
        
        CUDA_VISIBLE_DEVICES=2 python bin/eno.py esm1b --evolocity --downsample $percentage \
               --seed $seed \
               > eno_esm1b_downsample"$percentage".log 2>&1 &
        
        CUDA_VISIBLE_DEVICES=3 python bin/ser.py esm1b --evolocity --downsample $percentage \
               --seed $seed \
               > ser_esm1b_downsample"$percentage".log 2>&1 &
        wait
    done
done

# Weighted downsampling benchmarks.
for percentage in ${percentages[@]}
do
    for seed in {2..2}
    do
        CUDA_VISIBLE_DEVICES=0 python bin/flu_np.py esm1b --evolocity --wdownsample $percentage \
               --seed $seed \
               > np_esm1b_wdownsample"$percentage".log 2>&1 &
        
        CUDA_VISIBLE_DEVICES=1 python bin/flu.py esm1b --evolocity --wdownsample $percentage \
               --seed $seed \
               > h1_esm1b_wdownsample"$percentage".log 2>&1 &
        
        CUDA_VISIBLE_DEVICES=2 python bin/gag.py esm1b --evolocity --wdownsample $percentage \
               --seed $seed \
               > gag_esm1b_wdownsample"$percentage".log 2>&1 &

        #python bin/cov.py esm1b --evolocity --wdownsample $percentage \
        #       --seed $seed \
        #       > cov_esm1b_wdownsample"$percentage".log 2>&1 &

        CUDA_VISIBLE_DEVICES=3 python bin/cyc.py esm1b --evolocity --wdownsample $percentage \
               --seed $seed \
               > cyc_esm1b_wdownsample"$percentage".log 2>&1 &
        
        CUDA_VISIBLE_DEVICES=0 python bin/glo.py esm1b --evolocity --wdownsample $percentage \
               --seed $seed \
               > glo_esm1b_wdownsample"$percentage".log 2>&1 &
        
        CUDA_VISIBLE_DEVICES=1 python bin/pgk.py esm1b --evolocity --wdownsample $percentage \
               --seed $seed \
               > pgk_esm1b_wdownsample"$percentage".log 2>&1 &
        
        CUDA_VISIBLE_DEVICES=2 python bin/eno.py esm1b --evolocity --wdownsample $percentage \
               --seed $seed \
               > eno_esm1b_wdownsample"$percentage".log 2>&1 &
        
        CUDA_VISIBLE_DEVICES=3 python bin/ser.py esm1b --evolocity --wdownsample $percentage \
               --seed $seed \
               > ser_esm1b_wdownsample"$percentage".log 2>&1 &
        wait
    done
done
