import torch, os
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from losses import SSIMLoss, generator_loss, discriminator_loss, generator_loss_separately, adversarial_loss, NRMSELoss
from plotter import plotter_GAN, plotter_UNET

def GAN_training(hparams):#separate function for doing generative training
    #load the parameters of interest
    device = hparams.device  
    epochs = hparams.epochs
    lr = hparams.lr
    Lambda = hparams.Lambda
    UNet1 = hparams.generator
    Discriminator1 = hparams.discriminator
    train_loader = hparams.train_loader 
    val_loader   = hparams.val_loader   
    patch_size   = hparams.patch_size
    patch_stride = hparams.patch_stride
    Disc_train_freq = hparams.Disc_train_freq #frequency at which discriminator is trained as compared to the generator

    # choosing betas after talking with Ali, this are required for the case of GANs
    G_optimizer = optim.Adam(UNet1.parameters(), lr=lr, betas=(0.5, 0.999))#right now choosing Adam, other option is SGD
    scheduler = StepLR(G_optimizer, hparams.step_size, gamma=hparams.decay_gamma)
    D_optimizer = optim.Adam(Discriminator1.parameters(), lr=4*lr, betas=(0.5, 0.999))#right now choosing Adam, other option is SGD

    # initialize arrays for storing losses
    train_data_len = train_loader.__len__() # length of training_generator
    # Criterions or losses to choose from
    if (hparams.loss_type=='SSIM'):
        main_loss       = SSIMLoss().to(device)
    elif (hparams.loss_type=='L1'):
        main_loss  = nn.L1Loss()
    elif (hparams.loss_type=='L2'):
        main_loss  = nn.MSELoss() #same as L2 loss

    #lists to store the losses of discriminator and generator
    G_loss_l1, G_loss_adv    = np.zeros((epochs,train_data_len)), np.zeros((epochs,train_data_len)) 
    D_loss_real, D_loss_fake = np.zeros((epochs,train_data_len)), np.zeros((epochs,train_data_len))
    D_out_real, D_out_fake   = np.zeros((epochs,train_data_len)), np.zeros((epochs,train_data_len))
    G_loss_list, D_loss_list = np.zeros((epochs,train_data_len)), np.zeros((epochs,train_data_len))

    if (hparams.mode=='Patch'):
        unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_stride) # Unfold kernel
    # Loop over epochs
    for epoch in tqdm(range(epochs), total=epochs, leave=True):
        disc_epoch = 10 #discriminator will be trained 10 times as much as generator and it will be trained first
        # this does not work as I expected hence, both need to be trained simultaneoulsy
        for disc_epoch in range(disc_epoch):
            for index, (input_img, target_img, params) in enumerate(train_loader):
                if (hparams.mode=='Patch'):
                    unfolded_in, unfolded_out = unfold(input_img[None,...]), unfold(target_img[None,...])
                    patches_in,  patches_out  = unfolded_in.reshape(1,patch_size,patch_size,-1), unfolded_out.reshape(1,patch_size,patch_size,-1)
                    patches_in,  patches_out  = patches_in.permute(3,0,1,2), patches_out.permute(3,0,1,2)
                    input_img, target_img = patches_in.to(device), patches_out.to(device) # Transfer to GPU
                else:
                    input_img, target_img = input_img[None,...], target_img[None,...]
                # this works for both
                input_img, target_img = input_img.to(device), target_img.to(device) # Transfer to GPU

                generated_image = UNet1(input_img)
                G = Discriminator1(generated_image)

                # ground truth labels real and fake
                output_size = G.size(3)
                #using soft targets
                real_target = 0.9*(torch.ones(input_img.size(0), 1, output_size, output_size).to(device))
                fake_target = (torch.zeros(input_img.size(0), 1, output_size, output_size).to(device))


                disc_inp_fake = generated_image.detach()
                D_fake = Discriminator1(disc_inp_fake)
                D_fake_loss = discriminator_loss(D_fake, fake_target)
                #Disc real loss
                disc_inp_real = target_img                
                D_real = Discriminator1(disc_inp_real)
                D_real_loss = discriminator_loss(D_real,  real_target)

                # average discriminator loss
                D_total_loss = (D_real_loss + D_fake_loss) / 2
                # compute gradients and run optimizer step
                D_total_loss.backward()
                D_optimizer.step()
                D_loss_list[epoch,index] = D_loss_list[epoch,index] + D_total_loss.cpu().detach().numpy()
                D_loss_real[epoch,index] = D_loss_real[epoch,index] + D_real_loss.cpu().detach().numpy()
                D_loss_fake[epoch,index] = D_loss_fake[epoch,index] + D_fake_loss.cpu().detach().numpy()
        D_loss_list[epoch,:] =  D_loss_list[epoch,:]/disc_epoch #avg loss over disc_epoch training of discriminator
        D_loss_real[epoch,:] =  D_loss_real[epoch,:]/disc_epoch
        D_loss_fake[epoch,:] =  D_loss_fake[epoch,:]/disc_epoch


        for index, (input_img, target_img, params) in enumerate(train_loader):
            if (hparams.mode=='Patch'):
                unfolded_in, unfolded_out = unfold(input_img[None,...]), unfold(target_img[None,...])
                patches_in,  patches_out  = unfolded_in.reshape(1,patch_size,patch_size,-1), unfolded_out.reshape(1,patch_size,patch_size,-1)
                patches_in,  patches_out  = patches_in.permute(3,0,1,2), patches_out.permute(3,0,1,2)
                input_img, target_img = patches_in.to(device), patches_out.to(device) # Transfer to GPU
            else:
                input_img, target_img = input_img[None,...], target_img[None,...]
            # this works for both
            input_img, target_img = input_img.to(device), target_img.to(device) # Transfer to GPU
            # generator forward pass
            #
            # Train generator with real labels, train generator before the discriminator
            #
            generated_image = UNet1(input_img)
            G = Discriminator1(generated_image)

            # ground truth labels real and fake
            output_size = G.size(3)
            real_target = 0.9*(torch.ones(input_img.size(0), 1, output_size, output_size).to(device))
            fake_target = (torch.zeros(input_img.size(0), 1, output_size, output_size).to(device))

            gen_loss = adversarial_loss(G, real_target)
            #the 1 tensor need to be changed based on the max value in the input images
            if (hparams.loss_type=='SSIM'):
                loss_val = main_loss(generated_image, target_img, torch.tensor([1]).to(device))
            else:
                loss_val = main_loss(generated_image, target_img)
            G_loss = gen_loss + (Lambda* loss_val)  
            G_loss_list[epoch,index] = G_loss.cpu().detach().numpy()
            G_loss_l1[epoch,index], G_loss_adv[epoch,index] = loss_val.cpu().detach().numpy(), gen_loss.cpu().detach().numpy()   
            #storing discriminator outputs 
            D_out_fake[epoch,index] = np.mean(G.cpu().detach().numpy())             
            G_real = Discriminator1(target_img)
            D_out_real[epoch,index] = np.mean(G_real.cpu().detach().numpy())
            # compute gradients and run optimizer step
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()

            # Generator training ends
        # Scheduler
        scheduler.step()
    # Save models
    local_dir = hparams.global_dir + '/learning_rate_{:.4f}_epochs_{}_lambda_{}'.format(hparams.lr,hparams.epochs,hparams.Lambda) 
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
    tosave_weights = local_dir +'/saved_weights.pt' 
    torch.save({
        'epoch': epoch,
        'model_state_dict': UNet1.state_dict(),
        'optimizer_state_dict': G_optimizer.state_dict(),
        'Discriminator_state_dict':Discriminator1.state_dict(),
        'G_loss_list': G_loss_list,
        'G_loss_l1': G_loss_l1,
        'G_loss_adv': G_loss_adv,
        'D_loss_list': D_loss_list,
        'D_loss_real': D_loss_real,
        'D_loss_fake': D_loss_fake,
        'D_out_real':D_out_real,
        'D_out_fake':D_out_fake,
        'hparams': hparams}, tosave_weights)
    plotter_GAN(hparams,tosave_weights,local_dir,UNet1,train_loader,val_loader)


def UNET_training(hparams):
    device       = hparams.device  
    epochs       = hparams.epochs
    lr           = hparams.lr
    UNet1        = hparams.generator
    train_loader = hparams.train_loader 
    val_loader   = hparams.val_loader   
    patch_size   = hparams.patch_size
    G_optimizer = optim.Adam(UNet1.parameters(), lr=lr)#right now choosing Adam, other option is SGD
    scheduler = StepLR(G_optimizer, hparams.step_size, gamma=hparams.decay_gamma)
    # initialize arrays for storing losses
    train_data_len = train_loader.__len__() # length of training_generator
    val_data_len = val_loader.__len__()
    # Criterions
    # Criterions or losses to choose from
    if (hparams.loss_type=='SSIM'):
        main_loss       = SSIMLoss().to(device)
    elif (hparams.loss_type=='L1'):
        main_loss  = nn.L1Loss()
    elif (hparams.loss_type=='L2'):
        main_loss  = nn.MSELoss() #same as L2 loss
    
    train_loss = np.zeros((epochs,train_data_len)) #lists to store the losses of discriminator and generator
    val_loss = np.zeros((epochs,val_data_len)) #lists to store the losses of discriminator and generator
    if (hparams.mode=='Patch'):
        unfold = torch.nn.Unfold(kernel_size=hparams.patch_size, stride=hparams.patch_stride) # Unfold kernel
    # Loop over epochs
    for epoch in tqdm(range(epochs), total=epochs, leave=True):
        for index, (input_img, target_img, params) in enumerate(train_loader):
            if (hparams.mode=='Patch'):
                unfolded_in, unfolded_out = unfold(input_img[None,...]), unfold(target_img[None,...])
                patches_in,  patches_out  = unfolded_in.reshape(1,patch_size,patch_size,-1), unfolded_out.reshape(1,patch_size,patch_size,-1)
                patches_in,  patches_out  = patches_in.permute(3,0,1,2), patches_out.permute(3,0,1,2)
                input_img, target_img = patches_in.to(device), patches_out.to(device) # Transfer to GPU
            else:
                input_img, target_img = input_img[None,...], target_img[None,...]
            # Transfer to GPU
            input_img, target_img = input_img.to(device), target_img.to(device)

            generated_image = UNet1(input_img)

            #the 1 tensor need to be changed based on the max value in the input images
            if (hparams.loss_type=='SSIM'):
                loss_val = main_loss(generated_image, target_img, torch.tensor([1]).to(device))
            else:
                loss_val = main_loss(generated_image, target_img)

            # compute gradients and run optimizer step
            G_optimizer.zero_grad()
            loss_val.backward()
            G_optimizer.step()
            train_loss[epoch,index] = loss_val.cpu().detach().numpy()
        # Scheduler
        scheduler.step()
        for index, (input_img, target_img, params) in enumerate(val_loader):
            if (hparams.mode=='Patch'):
                unfolded_in, unfolded_out = unfold(input_img[None,...]), unfold(target_img[None,...])
                patches_in,  patches_out  = unfolded_in.reshape(1,patch_size,patch_size,-1), unfolded_out.reshape(1,patch_size,patch_size,-1)
                patches_in,  patches_out  = patches_in.permute(3,0,1,2), patches_out.permute(3,0,1,2)
                input_img, target_img = patches_in.to(device), patches_out.to(device) # Transfer to GPU
            else:
                input_img, target_img = input_img[None,...], target_img[None,...]
            # Transfer to GPU
            input_img, target_img = input_img.to(device), target_img.to(device)

            generated_image = UNet1(input_img)

            #the 1 tensor need to be changed based on the max value in the input images
            if (hparams.loss_type=='SSIM'):
                loss_val = main_loss(generated_image, target_img, torch.tensor([1]).to(device))
            else:
                loss_val = main_loss(generated_image, target_img)
            val_loss[epoch,index] = loss_val.cpu().detach().numpy()
    # Save models
    local_dir = hparams.global_dir + '/learning_rate_{:.4f}_epochs_{}_lambda_{}'.format(hparams.lr,hparams.epochs,hparams.Lambda) 
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
    tosave_weights = local_dir +'/saved_weights.pt' 
    torch.save({
        'epoch': epoch,
        'model_state_dict': UNet1.state_dict(),
        'optimizer_state_dict': G_optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'hparams': hparams}, tosave_weights)
    plotter_UNET(hparams,tosave_weights,local_dir,UNet1,train_loader,val_loader)
