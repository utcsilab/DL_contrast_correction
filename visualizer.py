# %% #Use (# %%) to convert a normal python file to breakable cells which you can 
# visualize right inside the vscode
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import torch, os
from dotmap import DotMap
from Unet import Unet

# %%
os.chdir('/home/sidharth/sid_notebooks/UNET_GAN2_training/train_results/model_GAN_input_data_available_input_data_loss_type_SSIM_mode_Full_img/learning_rate_0.0001_epochs_1000_lambda_1')
saved_results = torch.load('saved_weights.pt', map_location='cpu')
hparams   =  saved_results['hparams']
hparams.device = 'cpu' #all gpus are clogged
UNet1 = Unet(in_chans = hparams.n_channels, out_chans=hparams.n_channels,chans=hparams.filter).to(hparams.device)
UNet1.load_state_dict(saved_results['model_state_dict'])
UNet1.eval()
val_loader = hparams.val_loader
# %%
for index, (input_img, target_img, params) in enumerate(val_loader):
    print(input_img.size(), target_img.size())
    model_out = UNet1(input_img[None,...].to(hparams.device)) 

    NN_output = model_out.cpu().detach().numpy().squeeze()
    actual_out = target_img.cpu().detach().numpy().squeeze()
    actual_in = input_img.cpu().detach().numpy().squeeze()
    print('Parameters of contrast:- ','(TE = {}, TR = {}, TI = {})'.format(*params))
    # print('NRMSE between the ground truth and the NN input:- ',function_defs.nrmse(actual_out,actual_in))
    # print('NRMSE between the ground truth and the NN output:- ',function_defs.nrmse(actual_out,NN_output))
    
    plt.figure(figsize=(16,6))
    plt.subplot(1,4,1)
    plt.imshow(np.abs(actual_in),cmap='gray',vmax=.51,vmin=0)
    plt.title('Input')
#     plt.colorbar()
    plt.axis('off')
    plt.subplot(1,4,2)
    plt.imshow(np.abs(NN_output),cmap='gray',vmax=0.5,vmin=0)
    plt.title('Gen Out')
    plt.axis('off')
#     plt.colorbar()
    plt.subplot(1,4,3)
    plt.imshow(np.abs(actual_out),cmap='gray',vmax=0.5,vmin=0)
    plt.title('Ground Truth')
    plt.axis('off')
#     plt.colorbar()

    plt.subplot(1,4,4)
    plt.imshow(np.abs(NN_output-actual_out),cmap='gray',vmax=0.5*0.5,vmin=0)
    plt.title('Difference 2X')
    plt.axis('off')
#     plt.colorbar()
    plt.show()
    break
# %%
for index, (input_img, target_img, params) in enumerate(hparams.train_loader):
    print(input_img.size(), target_img.size())

    model_out = UNet1(input_img[None,...].to(hparams.device)) 
    

    NN_output = model_out.cpu().detach().numpy().squeeze()
    actual_out = target_img.cpu().detach().numpy().squeeze()
    actual_in = input_img.cpu().detach().numpy().squeeze()
    print('Parameters of contrast:- ','(TE = {}, TR = {}, TI = {})'.format(*params))
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
# %%
D_loss_real   =  saved_results['D_loss_real']
D_loss_fake   =  saved_results['D_loss_fake']
# %%
