#!/bin/bash

export PYTHONHASHSEED=0
#export CUDA_LAUNCH_BLOCKING=1
source ~/bin/anaconda3/bin/activate gsmv
RES_DIR="/home/mohammad/gsmvdev/new/"

runs_ens(){
python3 train.py --exp "ENS_EPS2000" --dataset cifar10 --data_dir ~/Database/Image/ \
   --objective bce --lr_d 0.0005 --lr_g 0.0005 --lr_patience 0.25 \
   --missing_type $2 --missing_rate $3 --hint_rate 0.0 --alpha 0.0 \
   --device cuda:$1 --epoches 2000 --eval_freq 0.05 --batch_size 64 \
   --train_predictor --n_samples $4  --aug_noise_std 0.0 \
   --result_dir $RES_DIR --dump_ens

}


for i in {1..1}
do
    runs_ens 0 mcar_uniform 0.2 128 &
    runs_ens 1 mcar_uniform 0.6 128 &
    runs_ens 2 mcar_rect 0.2 128 &
    runs_ens 3 mcar_rectinv 0.6 128 &
    wait
done

wait
