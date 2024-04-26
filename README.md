# Correcting synthetic MRI contrast-weighted images using deep learning

This repository hosts the official PyTorch implementation of the paper: [Correcting synthetic MRI contrast-weighted images using deep learning](https://www.sciencedirect.com/science/article/pii/S0730725X23002072).


Authored by: Sidharth Kumar, Hamidreza Saber, Odelin Charron, Leorah Freeman, Jonathan I. Tamir

<center><img src="https://github.com/utcsilab/DL_contrast_correction/blob/main/docs/Proposed-model_improvements.png" width="1024"></center>

<u> Figure </u>: *Comparison of contrast correction using different models. The first column is for the ground truth data, the second column is for the synthetic MR image, third row is for the case of the residual model, the fourth row is for the direct contrast correction model and the last column is for the proposed multiplicative model. Each row corresponds to a different subject from the test set and different TI values (in ms) which are as mentioned in the Figure.*



## Abstract
*Synthetic magnetic resonance imaging (MRI) offers a scanning paradigm where a fast multi-contrast sequence can be used to estimate underlying quantitative tissue parameter maps, which are then used to synthesize any desirable clinical contrast by retrospectively changing scan parameters in silico. Two benefits of this approach are the reduced exam time and the ability to generate arbitrary contrasts offline. However, synthetically generated contrasts are known to deviate from the contrast of experimental scans. The reason for contrast mismatch is the necessary exclusion of some unmodeled physical effects such as partial voluming, diffusion, flow, susceptibility, magnetization transfer, and more. The inclusion of these effects in signal encoding would improve the synthetic images, but would make the quantitative imaging protocol impractical due to long scan times. Therefore, in this work, we propose a novel deep learning approach that generates a multiplicative correction term to capture unmodeled effects and correct the synthetic contrast images to better match experimental contrasts for arbitrary scan parameters. The physics inspired deep learning model implicitly accounts for some unmodeled physical effects occurring during the scan. As a proof of principle, we validate our approach on synthesizing arbitrary inversion recovery fast spin-echo scans using a commercially available 2D multi-contrast sequence. We observe that the proposed correction visually and numerically reduces the mismatch with experimentally collected contrasts compared to conventional synthetic MRI. Finally, we show results of a preliminary reader study and find that the proposed method statistically significantly improves in contrast and SNR as compared to synthetic MR images.*

## Installation
The recommended way to run the code is with an Anaconda/Miniconda environment.
First, clone the repository: 

`git clone https://github.com/utcsilab/DL_contrast_correction.git`.

Then, create a new Anaconda environment and install the dependencies:

`conda env create -f environment.yml -n Transfer_GAN`


## Training New Models

To train a new GANl on the dataset, run the following bash script: 

`DL_contrast_correction/job_training_script.sh`

```
export epochs="1000"
export random_seed="80"
export learn_rate='0.001'
export disc_learn_rate='0.00001'
export model_arc='UNET'
export patch_size=72
export patch_stride=72
export loss_type='L2'
export Lambda=0
export Lambda_b=0
export data_file='repo_text_files_0.8'
export disc_epoch=20
export gen_epoch=10
export filter=64
export batch_size=64
export step_size=10000
export decay_gamma=0.5
export n_channels=6
export root_dir='/csiNAS/sidharth/DL_contrast_physics'
export GPU_idx=3


python train0.py -e ${epochs} -lr ${learn_rate} -dlr ${disc_learn_rate} -df ${data_file} -ma ${model_arc} -l ${loss_type} -G ${GPU_idx} -lb ${Lambda} -lb_b ${Lambda_b} -de ${disc_epoch} -ge ${gen_epoch} -f ${filter} -b ${batch_size}
```