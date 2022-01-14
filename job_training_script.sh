#!/bin/bash
# bash script to run multiple simulations one after another to automate stuff
# need to look at how to do parrallel runs on multiple gpus
for model_type in GAN #UNET
do
    for loss in  L1 SSIM # L2 #
    do
        python train0.py -e 1000 -ma $model_type -l $loss -mm 'Full_img' -G 4 -lb 1
    done
done
echo "All done"