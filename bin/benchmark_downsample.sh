declare -a percentages=(75 50 25 10)

for percentage in ${percentages[@]}
do
    #python bin/flu_np.py esm1b --evolocity --downsample $percentage \
    #     > np_esm1b_downsample"$percentage".log 2>&1

    CUDA_VISIBLE_DEVICES=0 python bin/flu.py esm1b --evolocity --downsample $percentage \
         > h1_esm1b_downsample"$percentage".log 2>&1 &

    CUDA_VISIBLE_DEVICES=1 python bin/gag.py esm1b --evolocity --downsample $percentage \
         > gag_esm1b_downsample"$percentage".log 2>&1 &

    #python bin/cov.py esm1b --evolocity --downsample $percentage \
    #     > cov_esm1b_downsample"$percentage".log 2>&1 &

    CUDA_VISIBLE_DEVICES=2 python bin/cyc.py esm1b --evolocity --downsample $percentage \
         > cyc_esm1b_downsample"$percentage".log 2>&1 &

    CUDA_VISIBLE_DEVICES=3 python bin/glo.py esm1b --evolocity --downsample $percentage \
         > glo_esm1b_downsample"$percentage".log 2>&1 &
    
    CUDA_VISIBLE_DEVICES=0 python bin/pgk.py esm1b --evolocity --downsample $percentage \
         > pgk_esm1b_downsample"$percentage".log 2>&1 &
    
    CUDA_VISIBLE_DEVICES=1 python bin/eno.py esm1b --evolocity --downsample $percentage \
         > eno_esm1b_downsample"$percentage".log 2>&1 &
    
    CUDA_VISIBLE_DEVICES=2 python bin/ser.py esm1b --evolocity --downsample $percentage \
         > ser_esm1b_downsample"$percentage".log 2>&1 &

    wait
done

for percentage in ${percentages[@]}
do
    python bin/flu_np.py esm1b --evolocity --wdownsample $percentage \
         > np_esm1b_wdownsample"$percentage".log 2>&1

    CUDA_VISIBLE_DEVICES=0 python bin/flu.py esm1b --evolocity --wdownsample $percentage \
         > h1_esm1b_wdownsample"$percentage".log 2>&1 &

    CUDA_VISIBLE_DEVICES=1 python bin/gag.py esm1b --evolocity --wdownsample $percentage \
         > gag_esm1b_wdownsample"$percentage".log 2>&1 &

    #python bin/cov.py esm1b --evolocity --wdownsample $percentage \
    #     > cov_esm1b_wdownsample"$percentage".log 2>&1 &

    CUDA_VISIBLE_DEVICES=2 python bin/cyc.py esm1b --evolocity --wdownsample $percentage \
         > cyc_esm1b_wdownsample"$percentage".log 2>&1 &

    CUDA_VISIBLE_DEVICES=3 python bin/glo.py esm1b --evolocity --wdownsample $percentage \
         > glo_esm1b_wdownsample"$percentage".log 2>&1 &
    
    CUDA_VISIBLE_DEVICES=0 python bin/pgk.py esm1b --evolocity --wdownsample $percentage \
         > pgk_esm1b_wdownsample"$percentage".log 2>&1 &
    
    CUDA_VISIBLE_DEVICES=1 python bin/eno.py esm1b --evolocity --wdownsample $percentage \
         > eno_esm1b_wdownsample"$percentage".log 2>&1 &
    
    CUDA_VISIBLE_DEVICES=2 python bin/ser.py esm1b --evolocity --wdownsample $percentage \
         > ser_esm1b_wdownsample"$percentage".log 2>&1 &

    wait
done
