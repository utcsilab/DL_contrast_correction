#Use (# %%) to convert a normal python file to breakable cells which you can 
# visualize right inside the vscode
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import torch, os
from dotmap import DotMap
from Unet import Unet
from models import Discriminator
from torchvision import transforms
from dataloader import *
from losses import SSIMLoss, generator_loss, discriminator_loss, generator_loss_separately, adversarial_loss, NRMSELoss

os.chdir('/home/sidharth/sid_notebooks/UNET_GAN2_training/train_results/model_GAN_input_data_mdme_data_loss_type_L1_mode_Full_img/learning_rate_0.0001_epochs_20_lambda_1')
saved_results = torch.load('saved_weights.pt')
hparams   =  saved_results['hparams']
# hparams.device = 'cpu' #all gpus are clogged
UNet1 = Unet(in_chans = hparams.n_channels, out_chans=hparams.n_channels,chans=hparams.filter).to(hparams.device)
UNet1.load_state_dict(saved_results['model_state_dict'])
UNet1.eval()
val_loader = hparams.val_loader
local_dir = hparams.global_dir + '/learning_rate_{:.4f}_epochs_{}_lambda_{}'.format(hparams.lr,hparams.epochs,hparams.Lambda) 
print(local_dir)

# for visualizing discriminator learned classification
Discriminator1 = Discriminator(input_nc = hparams.n_channels).to(hparams.device)
Discriminator1.load_state_dict(saved_results['Discriminator_state_dict'])
Discriminator1.eval()
# Discriminator1(input_img[None,...].to(hparams.device)) 


hparams.data_file   =  'mdme_data'#'subject13_data'
data_dir = hparams.root_dir + hparams.data_file
dataset = Exp_contrast_Dataset(data_dir,transform=transforms.Compose([
    Normalize_by_max(),Toabsolute()]))
test_loader           = torch.utils.data.DataLoader(dataset, batch_size=hparams.batch_size, shuffle=False)
print('Test data length:- ',test_loader.__len__())


for index, (input_img, target_img, params) in enumerate(test_loader):
    model_out = UNet1(input_img[None,...].to(hparams.device)) 
    NN_output = model_out.cpu().detach().numpy().squeeze()
    actual_out = target_img.cpu().detach().numpy().squeeze()
    actual_in = input_img.cpu().detach().numpy().squeeze()
    #testing discriminator on real images
    disc_pred_real = Discriminator1(target_img[None,...].to(hparams.device)) 
    output_size = disc_pred_real.size(3)
    real_target = 0.9*(torch.ones(input_img.size(0), 1, output_size, output_size).to(hparams.device))
    D_real_loss = discriminator_loss(disc_pred_real, real_target)
    disc_pred_fake = Discriminator1(model_out) 
    fake_target = (torch.zeros(input_img.size(0), 1, output_size, output_size).to(hparams.device))
    D_fake_loss = discriminator_loss(disc_pred_fake, fake_target)
    print('Discriminator BCE loss: real = {}, fake = {}'.format(D_real_loss, D_fake_loss))
    print('Avg. discriminator out for real and fake:-',torch.mean(disc_pred_real.cpu().detach()),torch.mean(disc_pred_fake.cpu().detach()))
    

    # print('Parameters of contrast:- ','(TE = {}, TR = {}, TI = {})'.format(*params[0]))
    # print('NRMSE between the ground truth and the NN input:- ',function_defs.nrmse(actual_out,actual_in))
    # print('NRMSE between the ground truth and the NN output:- ',function_defs.nrmse(actual_out,NN_output))
    
    # plt.figure(figsize=(16,6))
    # plt.suptitle('Parameters of contrast:- (TE = {}, TR = {}, TI = {}) {}'.format(*params[0],params[1]), fontsize=16)
    # plt.subplot(1,4,1)
    # plt.imshow(np.abs(actual_in),cmap='gray',vmax=0.5,vmin=0)
    # plt.title('Input')
    # plt.colorbar()
    # plt.axis('off')
    # plt.subplot(1,4,2)
    # plt.imshow(np.abs(NN_output),cmap='gray',vmax=0.5,vmin=0)
    # plt.title('Gen Out')
    # plt.axis('off')
    # plt.colorbar()
    # plt.subplot(1,4,3)
    # plt.imshow(np.abs(actual_out),cmap='gray',vmax=0.5,vmin=0)
    # plt.title('Ground Truth')
    # plt.axis('off')
    # plt.colorbar()
    # plt.subplot(1,4,4)
    # plt.imshow(np.abs(NN_output-actual_out),cmap='gray',vmax=0.5*0.5,vmin=0)
    # plt.title('Difference 2X')
    # plt.axis('off')
    # plt.colorbar()
    #     # Save
    # plt.tight_layout()
    # plt.savefig(local_dir + '/test_image_TE = {}, TR = {}, TI = {}_{}.png'.format(*params[0],params[1]), dpi=100)
    # plt.close()


exit()
for index, (input_img, target_img, params) in enumerate(hparams.train_loader):
    print(input_img.size(), target_img.size())

    model_out = UNet1(input_img[None,...].to(hparams.device)) 
    

    NN_output = model_out.cpu().detach().numpy().squeeze()
    actual_out = target_img.cpu().detach().numpy().squeeze()
    actual_in = input_img.cpu().detach().numpy().squeeze()
    # print('Parameters of contrast:- ','(TE = {}, TR = {}, TI = {})'.format(*params))
    # print('NRMSE between the ground truth and the NN input:- ',function_defs.nrmse(actual_out,actual_in))
    # print('NRMSE between the ground truth and the NN output:- ',function_defs.nrmse(actual_out,NN_output))
    
    plt.figure(figsize=(16,6))
    plt.subplot(1,4,1)
    plt.imshow(np.abs(actual_in),cmap='gray')#,vmax=.51,vmin=0)
    plt.title('Input')
#     plt.colorbar()
    plt.axis('off')
    plt.subplot(1,4,2)
    plt.imshow(np.abs(NN_output),cmap='gray')#,vmax=0.5,vmin=0)
    plt.title('Gen Out')
    plt.axis('off')
#     plt.colorbar()
    plt.subplot(1,4,3)
    plt.imshow(np.abs(actual_out),cmap='gray')#,vmax=0.5,vmin=0)
    plt.title('Ground Truth')
    plt.axis('off')
#     plt.colorbar()

    plt.subplot(1,4,4)
    plt.imshow(np.abs(NN_output-actual_out),cmap='gray',vmax=0.5*0.1,vmin=0)
    plt.title('Difference 10X')
    plt.axis('off')
#     plt.colorbar()
    plt.show()

D_loss_real   =  saved_results['D_loss_real']
D_loss_fake   =  saved_results['D_loss_fake']

