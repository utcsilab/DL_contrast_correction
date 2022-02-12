#!/bin/bash
# bash script to run multiple simulations one after another to automate stuff
# need to look at how to do parrallel runs on multiple gpus
# for model_type in GAN #UNET
# do
#     for loss in  L1 SSIM # L2 #
#     do
#         python train0.py -e 1000 -ma $model_type -l $loss -mm 'Full_img' -G 4 -lb 1
#     done
# done
# echo "All done"

export epochs="10"
export random_seed="80"
export learn_rate='0.0005'
export disc_learn_rate='0.00001'
export model_arc='GAN'
export model_mode='Full_img'
export patch_size=72
export patch_stride=72
export loss_type='L1'
export Lambda=1
export Lambda_b=0.1
export data_file='repo_text_files'
export disc_epoch=20
export gen_epoch=10
export filter=64
export batch_size=16
export step_size=10
export decay_gamma=0.5
export n_channels=2
export root_dir='/home/sidharth/sid_notebooks/UNET_GAN2_training/'
export GPU_idx=3

# source params.sh 

python train0.py -e ${epochs} -lr ${learn_rate} -dlr ${disc_learn_rate} -df ${data_file} -ma ${model_arc} -l ${loss_type} -mm ${model_mode} -G ${GPU_idx} -lb ${Lambda} -lb_b ${Lambda_b} -de ${disc_epoch} -ge ${gen_epoch} -f ${filter} -b ${batch_size}