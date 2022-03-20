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
parser.add_argument('-e','--epochs', type=int, default=4, metavar='', help = 'number of epochs to train the network') #positional argument
parser.add_argument('-rs','--random_seed', type=int, default=80, metavar='', help = 'Random reed for the PRNGs of the training') #optional argument
parser.add_argument('-lr','--learn_rate', type=float, default=0.0001, metavar='', help = 'Learning rate for the Generator') #optional argument
parser.add_argument('-dlr','--disc_learn_rate', type=float, default=0.00001, metavar='', help = 'Learning rate for the discriminator') #optional argument
parser.add_argument('-ma','--model_arc', type=str, default='UNET', metavar='',choices=['UNET', 'GAN'], help = 'Choose the type of network to learn')
parser.add_argument('-mm','--model_mode', type=str, default='Full_img', metavar='',choices=['Full_img', 'Patch'], help = 'Choose the mode to train the network either pass full image or patches')
parser.add_argument('-ps','--patch_size',type=int,default=72,metavar='',help='size of patches')
parser.add_argument('-pst','--patch_stride',type=int,default=72,metavar='',help='stride of patches')
parser.add_argument('-l','--loss_type', type=str, default='L1', metavar='',choices=['SSIM', 'L1', 'L2'], help = 'Choose the loss type for the main network')
parser.add_argument('-G','--GPU_idx',  type =int, default=2, metavar='',  help='GPU to Use')
parser.add_argument('-lb','--Lambda', type=float, default=0,metavar='', help = 'variable to weight loss fn w.r.t adverserial loss')
parser.add_argument('-lb_b','--Lambda_b', type=float, default=0, metavar='', help = 'variable to weight loss fn w.r.t perceptual loss')
parser.add_argument('-df','--data_file', type=str, default='repo_text_files_10', metavar='', help = 'Data on which the model need to be trained')
parser.add_argument('-de','--disc_epoch', type=int, default=5, metavar='', help = 'epochs for training the disc separately') 
parser.add_argument('-ge','--gen_epoch', type=int, default=5, metavar='', help = 'epochs for training the gen separately')
parser.add_argument('-f','--filter',type=int, default=64, metavar='', help='num of filters for the UNET')
parser.add_argument('-b','--batch_size',type=int,default=5,metavar='',help='batch size for training')
parser.add_argument('-ss','--step_size',type=int,default=10,metavar='',help='Number of epochs to decay with gamma')
parser.add_argument('-dg','--decay_gamma',type=float, default=0.5, metavar='', help = 'gamma decay rate')
parser.add_argument('-nc','--n_channels',type=int,default=1,metavar='',help='number of channels for UNET')
parser.add_argument('-rd','--root_dir', type=str, default='/home/sidharth/sid_notebooks/UNET_GAN2_training/', metavar='', help = 'root directory where all the code is')
args = parser.parse_args()
# saved params.sh and then give that as input to the 

print(args) #print the given parameters till now

torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

# Disaster: trying to make the algorithms reproducible
# torch.use_deterministic_algorithms(True) # if you want to set the use of determnistic algorihtms with all of pytorch, this have issues when using patch based with SSIM (that in itself is not a good idea to use anyway)
torch.backends.cudnn.deterministic = True # Only affects convolution operations
torch.backends.cudnn.benchmark     = False #if you want to replicate the results make this true

# Make pytorch see the same order of GPUs as seen by the nvidia-smi command
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda:{}".format(args.GPU_idx) if torch.cuda.is_available() else "cpu")
args.device      = device

# Global directory where results will be stored for the network training runs
global_dir = args.root_dir  + 'train_results/model_%s_data_%s_loss_%s_mode_%s'\
    %(args.model_arc, args.data_file, args.loss_type, args.model_mode) 
if not os.path.exists(global_dir):
    os.makedirs(global_dir)
args.global_dir = global_dir

# Creating the dataloaders
# have the training text file location in the argparser
train_data_dir = args.root_dir + args.data_file + '/training_samples.txt'
train_dataset = Exp_contrast_Dataset(train_data_dir,transform=transforms.Compose([
    Normalize_by_max(),Toabsolute()]))

val_data_dir = args.root_dir + args.data_file +  '/val_samples.txt'
val_dataset = Exp_contrast_Dataset(val_data_dir,transform=transforms.Compose([
    Normalize_by_max(),Toabsolute()]))

train_loader         = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader           = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
args.train_loader = train_loader
args.val_loader   = val_loader
print('Training data length:- ',train_dataset.__len__())
print('Validation data length:- ',val_dataset.__len__())

UNet1 = Unet(in_chans = 2*args.n_channels, out_chans=args.n_channels,chans=args.filter).to(args.device)
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

args.generator     = UNet1
args.discriminator = Discriminator2 #now using the vgg network as the discriminator
if (args.model_arc == 'GAN'):
    training_funcs.GAN_training(args)
elif(args.model_arc == 'UNET'):
    training_funcs.UNET_training(args)