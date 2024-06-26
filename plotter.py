import numpy as np
import matplotlib.pyplot as plt
from dataloader import *
from torchvision import transforms
import torch
plt.rcParams.update({'font.size': 18})
plt.ioff(); plt.close('all')
"""
The aim of this script was to generate the resultant images after the training is done.
Separate images were generated for the train and val set. The generated images are becoming too large in number
need to only generate a fraction of images to declutter the results. 
"""
def plotter_GAN(hparams,tosave_weights,local_dir,UNet1,train_loader,val_loader):
    #plot the GAN training results
    saved_results = torch.load(tosave_weights)
    G_loss_l1   =  saved_results['G_loss_l1']
    G_loss_adv  =  saved_results['G_loss_adv']
    D_loss_real =  saved_results['D_loss_real']
    D_loss_fake =  saved_results['D_loss_fake']
    G_loss      =  saved_results['G_loss_list']
    D_loss      = saved_results['D_loss_list']
    D_out_fake = saved_results['D_out_fake']
    D_out_real = saved_results['D_out_real']
    val_nrmse_loss = saved_results['val_nrmse_loss']
    val_ssim_loss  = saved_results['val_ssim_loss']
    Lambda      = hparams.Lambda
    fig, ax1 = plt.subplots(figsize=(8,20), nrows=5, ncols=1)
    ax2 = ax1[0].twinx()
    ax1[0].plot(np.mean(G_loss_l1,axis=(1,2)), 'g-')
    ax2.plot(np.mean(G_loss_adv,axis=(1,2)), 'b-')
#     ax1.set_ylim([0, Lambda*.50])
    ax1[0].set_xlabel('Epoch index')
    ax1[0].set_ylabel('{} loss'.format(hparams.loss_type), color='g')
    ax1[0].tick_params(axis='y', colors='g')
    ax2.set_ylabel('Adv loss', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.title('Generator ({} and adv), $\lambda$ = {}'.format(hparams.loss_type ,Lambda))

    ax2 = ax1[1].twinx()
    ax1[1].plot(np.mean(D_loss_real,axis=(1,2)), 'g-')
    ax2.plot(np.mean(D_loss_fake,axis=(1,2)), 'b-')

    ax1[1].set_xlabel('Epoch index')
    ax1[1].set_ylabel('Disc real loss', color='g')
    ax1[1].tick_params(axis='y', colors='g')
    ax2.set_ylabel('Disc fake loss', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.title('Disc Loss (real and fake), $\lambda$ = {}'.format(Lambda))

    ax2 = ax1[2].twinx()
    ax1[2].plot(np.mean(D_loss,axis=(1,2)), 'g-')
    ax2.plot(np.mean(G_loss,axis=(1,2)), 'b-')
#     ax2.xlim([25, 50])
#     ax2.set_ylim([0, 5])

    ax1[2].set_xlabel('Epoch index')
    ax1[2].set_ylabel('Disc loss', color='g')
    ax1[2].tick_params(axis='y', colors='g')
    ax2.set_ylabel('Generator loss', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.title('GAN Loss, $\lambda$ = {}'.format(Lambda))


    ax2 = ax1[3].twinx()
    ax1[3].plot(np.mean(D_out_real,axis=(1,2)), 'g-')
    ax2.plot(np.mean(D_out_fake,axis=(1,2)), 'b-')

    ax1[3].set_xlabel('Epoch index')
    ax1[3].set_ylabel('Disc out real', color='g')
    ax1[3].tick_params(axis='y', colors='g')
    ax2.set_ylabel('Disc out fake', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.title('Disc out, $\lambda$ = {}'.format(Lambda))

    ax2 = ax1[4].twinx()
    ax1[4].plot(np.mean(val_nrmse_loss,axis=1), 'g-')
    ax2.plot(np.mean(val_ssim_loss,axis=1), 'b-')

    ax1[4].set_xlabel('Epoch index')
    ax1[4].set_ylabel('NRMSE', color='g')
    ax1[4].tick_params(axis='y', colors='g')
    ax2.set_ylabel('SSIM', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.title('Avg. NRMSE and SSIM (Val), $\lambda$ = {}'.format(Lambda))

    # Save
    plt.tight_layout()
    plt.savefig(local_dir + '/GAN&DISC_loss_curves.png', dpi=100)
    plt.close()

    if not os.path.exists(local_dir + '/test_images'):
        os.makedirs(local_dir + '/test_images')
    if not os.path.exists(local_dir + '/train_images'):
        os.makedirs(local_dir + '/train_images')
    if not os.path.exists(local_dir + '/val_images'):
        os.makedirs(local_dir + '/val_images')
    img_plotter(hparams, UNet1,val_loader,train_loader,local_dir)
    test_img_plotter(hparams, UNet1, local_dir)





def plotter_UNET(hparams,tosave_weights,local_dir,UNet1,train_loader,val_loader):
    # plot the UNET training results
    saved_results =  torch.load(tosave_weights)
    train_loss    =  saved_results['train_loss']
    val_loss      =  saved_results['val_loss']
    val_nrmse_loss      =  saved_results['val_nrmse_loss']
    val_ssim_loss      =  saved_results['val_ssim_loss']
    Lambda      = hparams.Lambda
    fig, ax1 = plt.subplots(figsize=(8,10), nrows=2, ncols=1)
    ax2 = ax1[0].twinx()
    ax1[0].plot(np.arange(hparams.epochs),np.mean(train_loss,axis=1), 'g-')
    ax2.plot(np.arange(hparams.epochs),np.mean(val_loss,axis=1), 'b-')
#     ax1.set_ylim([0, Lambda*.50])
    ax1[0].set_xlabel('Epoch index')
    ax1[0].set_ylabel('Train loss', color='g')
    ax1[0].tick_params(axis='y', colors='g')
    ax2.set_ylabel('Val loss', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.title('Train ({}) and val loss'.format(hparams.loss_type))


    ax2 = ax1[1].twinx()
    ax1[1].plot(np.arange(hparams.epochs),np.mean(val_nrmse_loss,axis=1), 'g-')
    ax2.plot(np.arange(hparams.epochs),np.mean(val_ssim_loss,axis=1), 'b-')

    ax1[1].set_xlabel('Epoch index')
    ax1[1].set_ylabel('NRMSE', color='g')
    ax1[1].tick_params(axis='y', colors='g')
    ax2.set_ylabel('SSIM', color='b')
    ax2.tick_params(axis='y', colors='b')
    plt.title('Avg. NRMSE and SSIM (Val), $\lambda$ = {}'.format(Lambda))

    # Save
    plt.tight_layout()
    plt.savefig(local_dir + '/UNET_loss_curves.png', dpi=100)
    plt.close()

    if not os.path.exists(local_dir + '/test_images'):
        os.makedirs(local_dir + '/test_images')
    if not os.path.exists(local_dir + '/train_images'):
        os.makedirs(local_dir + '/train_images')
    if not os.path.exists(local_dir + '/val_images'):
        os.makedirs(local_dir + '/val_images')
    img_plotter(hparams, UNet1,val_loader,train_loader,local_dir)
    test_img_plotter(hparams, UNet1, local_dir)



#function for plotting the test images
def test_img_plotter(hparams, UNet1, local_dir):
    test_data_dir = hparams.root_dir + hparams.data_file +  '/test_samples.txt'
    test_dataset = Exp_contrast_Dataset(test_data_dir,transform=transforms.Compose([Normalize_by_max(),Toabsolute()]))

    test_loader    = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)


    for index, (input_img, target_img, params) in enumerate(test_loader):
        if(test_loader.batch_size==1):
            TE, TR, TI = int(params[0][0]),int(params[0][1]),int(params[0][2])
            file_identifier = str(params[1])[20:50]
        else:
            TE, TR, TI = int(params[0][0][0]),int(params[0][1][0]),int(params[0][2][0])
            file_identifier = str(params[1][0])[20:50]

        multiplicative_term = UNet1(input_img.to(hparams.device)) 
        model_out = multiplicative_term*input_img[:,0,None,:,:].to(hparams.device)

        NN_output = model_out[0,...].cpu().detach().numpy().squeeze()
        actual_out = target_img[0,...].cpu().detach().numpy().squeeze()
        actual_in = input_img[0,...].cpu().detach().numpy().squeeze()

        plt.figure(figsize=(16,6))
        plt.suptitle('Parameters of contrast:- (TE = {}, TR = {}, TI = {}) {}'.format(TE, TR, TI, file_identifier), fontsize=16)
        plt.subplot(1,4,1)
        plt.imshow(np.abs(actual_in[0,:,:].squeeze()),cmap='gray',vmax=0.5,vmin=0)
        # plt.imshow(np.abs(actual_in[:,:].squeeze()),cmap='gray',vmax=0.5,vmin=0)#for the case of single channel input
        plt.title('Input')
        # plt.colorbar()
        plt.axis('off')
        plt.subplot(1,4,2)
        plt.imshow(np.abs(NN_output),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Gen Out')
        plt.axis('off')
        # plt.colorbar()
        plt.subplot(1,4,3)
        plt.imshow(np.abs(actual_out),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Ground Truth')
        plt.axis('off')
        # plt.colorbar()
        plt.subplot(1,4,4)
        plt.imshow(np.abs(NN_output-actual_out),cmap='gray',vmax=0.5*0.5,vmin=0)
        plt.title('Difference 2X')
        plt.axis('off')
        # plt.colorbar()
            # Save
        plt.tight_layout()
        plt.savefig(local_dir + '/test_images'+ '/test_image_TE = {}, TR = {}, TI = {}_{}.png'.format(TE, TR, TI, file_identifier) , dpi=100)
        plt.close()

# function for plotting the train and validation images
def img_plotter(hparams, UNet1,val_loader,train_loader,local_dir):
    for index, (input_img, target_img, params) in enumerate(val_loader):
        if(val_loader.batch_size==1):
            TE, TR, TI = int(params[0][0]),int(params[0][1]),int(params[0][2])
            file_identifier = str(params[1])[20:50]
        else:
            TE, TR, TI = int(params[0][0][0]),int(params[0][1][0]),int(params[0][2][0])
            file_identifier = str(params[1][0])[20:50]

        multiplicative_term = UNet1(input_img.to(hparams.device)) 
        model_out = multiplicative_term*input_img[:,0,None,:,:].to(hparams.device)
        model_out = multiplicative_term*input_img[:,0,None,:,:].to(hparams.device)


        # model_out = UNet1(input_img.to(hparams.device)) 
        NN_output = model_out[0,...].cpu().detach().numpy().squeeze()
        actual_out = target_img[0,...].cpu().detach().numpy().squeeze()
        actual_in = input_img[0,...].cpu().detach().numpy().squeeze()

        plt.figure(figsize=(16,6))
        plt.suptitle('Parameters of contrast:- (TE = {}, TR = {}, TI = {}) {}'.format(TE, TR, TI, file_identifier), fontsize=16)
        plt.subplot(1,4,1)
        plt.imshow(np.abs(actual_in[0,:,:].squeeze()),cmap='gray',vmax=0.5,vmin=0)
        # plt.imshow(np.abs(actual_in[:,:].squeeze()),cmap='gray',vmax=0.5,vmin=0)

        plt.title('Input')
        # plt.colorbar()
        plt.axis('off')
        plt.subplot(1,4,2)
        plt.imshow(np.abs(NN_output),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Gen Out')
        plt.axis('off')
        # plt.colorbar()
        plt.subplot(1,4,3)
        plt.imshow(np.abs(actual_out),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Ground Truth')
        plt.axis('off')
        # plt.colorbar()
        plt.subplot(1,4,4)
        plt.imshow(np.abs(NN_output-actual_out),cmap='gray',vmax=0.5*0.5,vmin=0)
        plt.title('Difference 2X')
        plt.axis('off')
        # plt.colorbar()
            # Save
        plt.tight_layout()
        plt.savefig(local_dir + '/val_images'+ '/val_image_TE = {}, TR = {}, TI = {}_{}.png'.format(TE, TR, TI, file_identifier) , dpi=100)
        plt.close()


    # # new train_loader with batch_size = 1
    # train_data_dir = hparams.root_dir + hparams.data_file + '/training_samples.txt'
    # train_dataset = Exp_contrast_Dataset(train_data_dir,transform=transforms.Compose([Normalize_by_max(),Toabsolute()]))
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    for index, (input_img, target_img, params) in enumerate(train_loader):
        if(train_loader.batch_size==1):
            TE, TR, TI = int(params[0][0]),int(params[0][1]),int(params[0][2])
            file_identifier = str(params[1])[20:50]
        else:
            TE, TR, TI = int(params[0][0][0]),int(params[0][1][0]),int(params[0][2][0])
            file_identifier = str(params[1][0])[20:50]

        # model_out = UNet1(input_img.to(hparams.device)) 

        multiplicative_term = UNet1(input_img.to(hparams.device)) 
        model_out = multiplicative_term*input_img[:,0,None,:,:].to(hparams.device)

        NN_output = model_out[0,...].cpu().detach().numpy().squeeze()
        actual_out = target_img[0,...].cpu().detach().numpy().squeeze()
        actual_in = input_img[0,...].cpu().detach().numpy().squeeze()

        plt.figure(figsize=(16,6))
        plt.suptitle('Parameters of contrast:- (TE = {}, TR = {}, TI = {}) {}'.format(TE, TR, TI, file_identifier), fontsize=16)
        plt.subplot(1,4,1)
        plt.imshow(np.abs(actual_in[0,:,:].squeeze()),cmap='gray',vmax=0.5,vmin=0)
        # plt.imshow(np.abs(actual_in[:,:].squeeze()),cmap='gray',vmax=0.5,vmin=0)#for the case of single channel input
        plt.title('Input')
        # plt.colorbar()
        plt.axis('off')
        plt.subplot(1,4,2)
        plt.imshow(np.abs(NN_output),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Gen Out')
        plt.axis('off')
        # plt.colorbar()
        plt.subplot(1,4,3)
        plt.imshow(np.abs(actual_out),cmap='gray',vmax=0.5,vmin=0)
        plt.title('Ground Truth')
        plt.axis('off')
        # plt.colorbar()
        plt.subplot(1,4,4)
        plt.imshow(np.abs(NN_output-actual_out),cmap='gray',vmax=0.5*0.5,vmin=0)
        plt.title('Difference 2X')
        plt.axis('off')
        # plt.colorbar()
            # Save
        plt.tight_layout()
        plt.savefig(local_dir +'/train_images'+ '/train_image_TE = {}, TR = {}, TI = {}_{}.png'.format(TE, TR, TI, file_identifier), dpi=100)
        plt.close()