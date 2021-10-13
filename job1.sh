#!/bin/bash
for model_type in UNET GAN
do
    for loss in L1 L2
    do
        python train0.py -e 1000 -ma $model_type -l $loss -mm 'Full_img' -G 3
    done
done
echo "All done"