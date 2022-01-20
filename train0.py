#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '.')

import torch, os, glob, copy
import numpy as np
from tqdm import tqdm
from dotmap import DotMap
import torch.optim as optim
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.nn import functional as F
from matplotlib import pyplot as plt
from dataloader import *
from torchvision import transforms
from Unet import Unet
from models import Discriminator
from losses import SSIMLoss, generator_loss, discriminator_loss, generator_loss_separately, adversarial_loss, NRMSELoss
from plotter import plotter_GAN
import training_funcs
import argparse

parser = argparse.ArgumentParser(description='Reading args for running the deep network training')
parser.add_argument('-e','--epochs', type=int, default=2, metavar='', help = 'number of epochs to train the network') #positional argument
parser.add_argument('-rs','--random_seed', type=int, default=80, metavar='', help = 'Random reed for the PRNGs of the training') #optional argument
parser.add_argument('-lr','--learn_rate', type=float, default=0.0001, metavar='', help = 'Learning rate for the network') #optional argument
parser.add_argument('-ma','--model_arc', type=str, default='GAN', metavar='',choices=['UNET', 'GAN'], help = 'Choose the type of network to learn')
parser.add_argument('-mm','--model_mode', type=str, default='Full_img', metavar='',choices=['Full_img', 'Patch'], help = 'Choose the mode to train the network either pass full image or patches')
parser.add_argument('-l','--loss_type', type=str, default='L1', metavar='',choices=['SSIM', 'L1', 'L2', 'Perc_L'], help = 'Choose the loss type for the main network')
parser.add_argument('-G','--GPU_idx',  type =int, default=4, metavar='',  help='GPU to Use')
parser.add_argument('-lb','--Lambda', type=float, default=1,metavar='', help = 'variable to weight loss fn w.r.t adverserial loss')
parser.add_argument('-lb_b','--Lambda_b', type=float, default=1,metavar='', help = 'variable to weight loss fn w.r.t perceptual loss')
parser.add_argument('-df','--data_file', type=str, default='mdme_data', metavar='',choices=['mdme_data', 'available_input_data'], help = 'Data on which the model need to be trained')
parser.add_argument('-de','--disc_epoch', type=int, default=10, metavar='', help = 'epochs for training the disc separately') 
parser.add_argument('-ge','--gen_epoch', type=int, default=10, metavar='', help = 'epochs for training the gen separately')
args = parser.parse_args()
# print(args) #print the read arguments

random_seed = args.random_seed  #changed to 80 to see the trianing behaviour on a different set
torch.manual_seed(random_seed)
np.random.seed(random_seed)

# Disaster: trying to make the algorithms reproducible
# torch.use_deterministic_algorithms(True) # if you want to set the use of determnistic algorihtms with all of pytorch, this have issues when using patch based with SSIM (that in itself is not a good idea to use anyway)
torch.backends.cudnn.deterministic = True # Only affects convolution operations
torch.backends.cudnn.benchmark     = False #if you want to replicate the results make this true

# Make pytorch see the same order of GPUs as seen by the nvidia-smi command
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:{}".format(args.GPU_idx) if torch.cuda.is_available() else "cpu")

# Config
hparams             = DotMap()
hparams.logging     = False
# Training parameters
hparams.random_seed = random_seed
hparams.epochs      = args.epochs
hparams.lr          = args.learn_rate
hparams.filter      = 64
hparams.Lambda      = args.Lambda
hparams.Lambda_b    = args.Lambda_b
hparams.device      = device
hparams.batch_size  = 1
hparams.val_split   = 0.2 #to test on subject 15 later on
hparams.step_size   = 10  # Number of epochs to decay with gamma
hparams.decay_gamma = 0.5
hparams.disc_epoch  = args.disc_epoch
hparams.gen_epoch   = args.gen_epoch
# Model parameters
hparams.n_channels  = 1
hparams.n_classes   = hparams.n_channels
hparams.root_dir    = '/home/sidharth/sid_notebooks/UNET_GAN2_training/'
hparams.data_file   = args.data_file #'mdme_data' # 'available_input_data' # 
hparams.model_arc   = args.model_arc #possible options are 'UNET' and 'GAN'
hparams.Disc_train_freq = 0.1 #frequency at which discriminator is trained as compared to the generator
hparams.loss_type   = args.loss_type #loss type to be used in training the model (SSIM, L1, L2, Perc_L), Perc_L is only for UNET right now
hparams.mode        = args.model_mode #options :-'Patch' or 'Full_img'
if hparams.mode == 'Patch':
    hparams.patch_size   = 72 # min value is 32 so that the UNet always have values with dim>1
    hparams.patch_stride = 72

print(hparams) #print the given parameters till now

# Global directory
global_dir = hparams.root_dir  + 'train_results/model_%s_input_data_%s_loss_type_%s_mode_%s'\
    %(hparams.model_arc, hparams.data_file, hparams.loss_type, hparams.mode) 
if not os.path.exists(global_dir):
    os.makedirs(global_dir)
hparams.global_dir = global_dir

# Creating the dataloaders
data_dir = hparams.root_dir + 'repo_text_files/training_samples.txt'
dataset = Exp_contrast_Dataset(data_dir,transform=transforms.Compose([
    Normalize_by_max(),Toabsolute()]))

dataset_size = len(dataset)
train_size = int( (1 - hparams.val_split) * (dataset_size))
test_size = (dataset_size) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader         = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
val_loader           = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.batch_size, shuffle=False)
hparams.train_loader = train_loader
hparams.val_loader   = val_loader
print('Training data length:- ',train_dataset.__len__())
print('Validation data length:- ',val_dataset.__len__())

UNet1 = Unet(in_chans = hparams.n_channels, out_chans=hparams.n_channels,chans=hparams.filter).to(hparams.device)
UNet1.train()

# print('Number of parameters in the generator:- ', np.sum([np.prod(p.shape) for p in UNet1.parameters() if p.requires_grad]))

# Discriminator1 = Discriminator(input_nc = hparams.n_channels).to(hparams.device)
# Discriminator1.train()

import torchvision.models as models
# vgg16 = models.vgg16() #going to use this for the preceptual loss, hence using resnet for the discriminator
#https://discuss.pytorch.org/t/modify-resnet-or-vgg-for-single-channel-grayscale/22762
# above link recommends using resnet instead of vgg as it is more powerful
resnet = models.resnet18()
Discriminator2 = resnet.to(device)

hparams.generator     = UNet1
hparams.discriminator = Discriminator2 #now using the vgg network as the discriminator
if (hparams.model_arc == 'GAN'):
    training_funcs.GAN_training(hparams)
elif(hparams.model_arc == 'UNET'):
    training_funcs.UNET_training(hparams)