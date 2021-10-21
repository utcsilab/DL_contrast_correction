#!/bin/bash
for model_type in GAN #UNET
do
    for loss in  L1 SSIM # L2 #
    do
        python train0.py -e 1000 -ma $model_type -l $loss -mm 'Full_img' -G 0 -lb 0.1
    done
done
echo "All done"