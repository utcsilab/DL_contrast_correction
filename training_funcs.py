import torch, os
from torch import nn
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from losses import SSIMLoss, generator_loss, discriminator_loss, generator_loss_separately, adversarial_loss, NRMSELoss, VGGPerceptualLoss
from plotter import plotter_GAN, plotter_UNET
import sys
from Unet import Unet

def binary_acc(disc_out, actual_out):#function for calculating accuracy of discriminator
    m = nn.Sigmoid()#sigmoid is removed from the discriminator def to automatically handle the edge cases
    output = m(disc_out)
    disc_prediction = output>0.5
    actual_out = actual_out*torch.ones(disc_prediction.shape)
    compare = actual_out == disc_prediction
    out = torch.sum(compare)/torch.prod(torch.tensor(list(actual_out.size())))
    return out


def GAN_training(hparams):#separate function for doing generative training
    #load the parameters of interest
    device = hparams.device  
    epochs = hparams.epochs
    lr = hparams.learn_rate
    disc_lr = hparams.disc_learn_rate
    Lambda = hparams.Lambda
    Lambda_b = hparams.Lambda_b
    UNet1 = hparams.generator
    Discriminator1 = hparams.discriminator
    train_loader = hparams.train_loader 
    val_loader   = hparams.val_loader   
    local_dir = hparams.global_dir + '/gen_lr_{:.5f}_disc_lr_{:.5f}_epochs_{}_lambda_{}_gen_epoch_{}_disc_epoch_{}_Lambda_b{}'.format(hparams.learn_rate,hparams.disc_learn_rate,hparams.epochs,hparams.Lambda,hparams.gen_epoch,hparams.disc_epoch,Lambda_b) 
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
    # choosing betas after talking with Ali, this are required for the case of GANs
    G_optimizer = optim.Adam(UNet1.parameters(), lr=lr, betas=(0.5, 0.999))
    G_scheduler = StepLR(G_optimizer, hparams.step_size, gamma=hparams.decay_gamma)
    D_optimizer = optim.Adam(Discriminator1.parameters(), lr=disc_lr, betas=(0.5, 0.999))
    D_scheduler = StepLR(D_optimizer, hparams.step_size, hparams.decay_gamma)
    # initialize arrays for storing losses
    train_data_len = train_loader.__len__() # length of training_generator
    val_data_len   = val_loader.__len__()   # length of val_generator 
    # Criterions or losses to choose from
    if (hparams.loss_type=='SSIM'):
        main_loss  = SSIMLoss().to(device)
    elif (hparams.loss_type=='L1'):
        main_loss  = nn.L1Loss()
    elif (hparams.loss_type=='L2'):
        main_loss  = nn.MSELoss() #same as L2 loss
    VGG_loss  = VGGPerceptualLoss().to(device) #perceptual loss 

    disc_epoch = hparams.disc_epoch #discriminator will be trained these many times
    gen_epoch  = hparams.gen_epoch #generator will be trained for these many iterations 

    #lists to store the losses of discriminator and generator
    G_loss_l1, G_loss_adv    = np.zeros((epochs,gen_epoch,train_data_len)), np.zeros((epochs,gen_epoch,train_data_len)) 
    D_loss_real, D_loss_fake = np.zeros((epochs,disc_epoch,train_data_len)), np.zeros((epochs,disc_epoch,train_data_len))
    D_out_real, D_out_fake   = np.zeros((epochs,gen_epoch,train_data_len)), np.zeros((epochs,gen_epoch,train_data_len))
    G_loss_list, D_loss_list = np.zeros((epochs,gen_epoch,train_data_len)), np.zeros((epochs,disc_epoch,train_data_len))
    D_out_acc                = np.zeros((epochs,disc_epoch,train_data_len))
    accuracy_results         = np.zeros((epochs,disc_epoch))
    val_nrmse_loss = np.zeros((epochs,val_data_len))
    val_ssim_loss  = np.zeros((epochs,val_data_len))
    SSIM       = SSIMLoss().to(device)
    NRMSE      = NRMSELoss()
    # Loop over epochs
    for epoch in tqdm(range(epochs), total=epochs, leave=True):
        # at each epoch I re-initiate the discriminator optimizer
        for disc_epoch_idx in range(disc_epoch):#first training the discriminator
            for index, (input_img, target_img, params) in enumerate(train_loader):
                target_img = target_img[None,...]
                # Transfer to GPU
                input_img, target_img = input_img.to(device), target_img.to(device)
                target_img = target_img.permute(1,0,2,3)# to make it work with batch size > 1
                # the network will only estimate the correction multiplicative term for now
                multiplicative_term = UNet1(input_img)
                generated_image = multiplicative_term * input_img[:,0,None,:,:]
                G = Discriminator1(generated_image)

                # ground truth labels real and fake
                real_target = torch.ones(list(G.size())).to(device)
                fake_target = torch.zeros(list(G.size())).to(device)

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
                for p in Discriminator1.parameters():#clipping the critic's weights
                    p.data.clamp_(-0.01, 0.01)
                D_out_acc[epoch,disc_epoch_idx,index] = (binary_acc(D_real.cpu(), True) + binary_acc(D_fake.cpu(), False))
                D_loss_list[epoch,disc_epoch_idx,index] =  D_total_loss.cpu().detach().numpy()
                D_loss_real[epoch,disc_epoch_idx,index] =  D_real_loss.cpu().detach().numpy()
                D_loss_fake[epoch,disc_epoch_idx,index] =  D_fake_loss.cpu().detach().numpy()
            accuracy_results[epoch,disc_epoch_idx] = np.sum(D_out_acc[epoch,disc_epoch_idx,:])/(2*train_data_len)
            # D_scheduler.step()
        # D_loss_list[epoch,:] =  D_loss_list[epoch,:]/disc_epoch #avg loss over disc_epoch training of discriminator
        # D_loss_real[epoch,:] =  D_loss_real[epoch,:]/disc_epoch
        # D_loss_fake[epoch,:] =  D_loss_fake[epoch,:]/disc_epoch

        for gen_epoch_idx in range(gen_epoch):
            for index, (input_img, target_img, params) in enumerate(train_loader):
                target_img = target_img[None,...]
                # Transfer to GPU
                input_img, target_img = input_img.to(device), target_img.to(device)
                target_img = target_img.permute(1,0,2,3)# to make it work with batch size > 1
                multiplicative_term = UNet1(input_img)
                generated_image = multiplicative_term * input_img[:,0,None,:,:]
                G = Discriminator1(generated_image)

                # ground truth labels real and fake
                real_target = (torch.ones(list(G.size())).to(device))
                fake_target = torch.zeros(list(G.size())).to(device)
        
                gen_loss = adversarial_loss(G, real_target)
                #the 1 tensor need to be changed based on the max value in the input images
                # by default perceptual loss is added to all losses
                if (hparams.loss_type=='SSIM'):
                    loss_val = main_loss(generated_image, target_img, torch.tensor([1]).to(device)) + Lambda_b*VGG_loss(generated_image, target_img)
                else:
                    loss_val = main_loss(generated_image, target_img) + Lambda_b*VGG_loss(generated_image, target_img)
                G_loss = Lambda*gen_loss + loss_val  
                # compute gradients and run optimizer step
                G_optimizer.zero_grad()
                G_loss.backward()
                G_optimizer.step()
                # store loss values
                G_loss_list[epoch,gen_epoch_idx,index] = G_loss.cpu().detach().numpy()
                G_loss_l1[epoch,gen_epoch_idx,index], G_loss_adv[epoch,gen_epoch_idx,index] = loss_val.cpu().detach().numpy(), gen_loss.cpu().detach().numpy()   
                #storing discriminator outputs 
                D_out_fake[epoch,gen_epoch_idx,index] = np.mean(G.cpu().detach().numpy())             
                G_real = Discriminator1(target_img)
                D_out_real[epoch,gen_epoch_idx,index] = np.mean(G_real.cpu().detach().numpy())

            # G_loss_list[epoch,:] = G_loss_list[epoch,:]/gen_epoch 
            # G_loss_l1[epoch,:], G_loss_adv[epoch,:] = G_loss_l1[epoch,:]/gen_epoch, G_loss_adv[epoch,:]/gen_epoch
            # #storing discriminator outputs 
            # D_out_fake[epoch,:] = D_out_fake[epoch,:]/gen_epoch          
            # D_out_real[epoch,:] = D_out_fake[epoch,:]/gen_epoch 
            # Generator training ends
        # Scheduler should be in the generator epochs or the overall epochs need to check this
        # G_scheduler.step()
        # saving the validation set results, now 
        for index, (input_img, target_img, params) in enumerate(val_loader):
            target_img = target_img[None,...]
            # Transfer to GPU
            input_img, target_img = input_img.to(device), target_img.to(device)
            target_img = target_img.permute(1,0,2,3)# to make it work with batch size > 1

            multiplicative_term = UNet1(input_img)
            generated_image = multiplicative_term * input_img[:,0,None,:,:]
            # SSIM def is defined in a way so that the network tries to minimize it
            val_ssim_loss[epoch,index] = 1 - SSIM(generated_image, target_img, torch.tensor([1]).to(device))
            val_nrmse_loss[epoch,index] = NRMSE(generated_image, target_img)
        #saving trained model at each epoch
        torch.save(UNet1.state_dict(), local_dir + '/model_epoch%d.pt' % epoch)
    # Save models
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
        'D_out_acc':D_out_acc,
        'val_nrmse_loss':val_nrmse_loss,
        'val_ssim_loss':val_ssim_loss,
        'hparams': hparams}, tosave_weights)
    
    sourceFile = open(local_dir +'/params_used.txt', 'w')
    for arg in vars(hparams):
        print(arg, '=', getattr(hparams, arg), file = sourceFile)
        if(arg=='val_loader'):
            break
    # print(hparams, file = sourceFile)
    sourceFile.close()
    plotter_GAN(hparams,tosave_weights,local_dir,UNet1,train_loader,val_loader)


def UNET_training(hparams):
    device       = hparams.device  
    epochs       = hparams.epochs
    lr           = hparams.learn_rate
    UNet1        = hparams.generator
    train_loader = hparams.train_loader 
    val_loader   = hparams.val_loader   
    Lambda_b = hparams.Lambda_b
    G_optimizer  = optim.Adam(UNet1.parameters(), lr=lr)#right now choosing Adam, other option is SGD
    # G_optimizer  = optim.SGD(UNet1.parameters(), lr=lr)#right now choosing Adam, other option is SGD
    scheduler    = StepLR(G_optimizer, hparams.step_size, gamma=hparams.decay_gamma)
    # initialize arrays for storing losses
    train_data_len = train_loader.__len__() # length of training_generator
    val_data_len = val_loader.__len__()
    # Criterions or losses to choose from
    if (hparams.loss_type=='SSIM'):
        main_loss  = SSIMLoss().to(device)
    elif (hparams.loss_type=='L1'):
        main_loss  = nn.L1Loss()
    elif (hparams.loss_type=='L2'):
        main_loss  = nn.MSELoss() #same as L2 loss
    elif (hparams.loss_type=='Perc_L'):#perceptual loss based on vgg
        main_loss  = VGGPerceptualLoss().to(device)
    VGG_loss  = VGGPerceptualLoss().to(device)
    NRMSE      = NRMSELoss()
    SSIM       = SSIMLoss().to(device)
    train_loss = np.zeros((epochs,train_data_len)) #lists to store the losses of discriminator and generator
    val_loss = np.zeros((epochs,val_data_len)) #lists to store the losses of discriminator and generator
    train_nrmse_loss = np.zeros((epochs,train_data_len))
    val_nrmse_loss = np.zeros((epochs,val_data_len))
    train_ssim_loss = np.zeros((epochs,train_data_len))
    val_ssim_loss = np.zeros((epochs,val_data_len))
    best_val_loss = 1000000000000 #variable to store the best val loss
    local_dir = hparams.global_dir + '/learning_rate_{:.5f}_epochs_{}_lambda_{}_loss_type{}_Lambda_b{}'.format(hparams.learn_rate,hparams.epochs,hparams.Lambda,hparams.loss_type,Lambda_b) 
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir)
    best_UNet = Unet(in_chans = hparams.n_channels, out_chans=1,chans=hparams.filter, num_pool_layers = 4,drop_prob=0.0).to(hparams.device)

    # Loop over epochs
    for epoch in tqdm(range(epochs), total=epochs, leave=True):
        UNet1.train()
        for index, (input_img, target_img, params) in enumerate(train_loader):
            target_img = target_img[None,...]
            # Transfer to GPU
            input_img, target_img = input_img.to(device), target_img.to(device)
            target_img = target_img.permute(1,0,2,3)# to make it work with batch size > 1

            multiplicative_term = UNet1(input_img)
            generated_image = multiplicative_term * input_img[:,0,None,:,:]


            #the 1 tensor need to be changed based on the max value in the input images
            # by default now every loss will have the perceptual loss included
            if (hparams.loss_type=='SSIM'):
                loss_val = main_loss(generated_image, target_img, torch.tensor([1]).to(device)) + Lambda_b*VGG_loss(generated_image, target_img)
            else:
                loss_val = main_loss(generated_image, target_img) + Lambda_b*VGG_loss(generated_image, target_img)

            # compute gradients and run optimizer step
            G_optimizer.zero_grad()
            loss_val.backward()
            G_optimizer.step()
            train_loss[epoch,index] = loss_val.cpu().detach().numpy()
        # Scheduler
        # scheduler.step()# this is hurting the learning process
        UNet1.eval()
        for index, (input_img, target_img, params) in enumerate(val_loader):
            target_img = target_img[None,...]
            # Transfer to GPU
            input_img, target_img = input_img.to(device), target_img.to(device)
            target_img = target_img.permute(1,0,2,3)# to make it work with batch size > 1

            multiplicative_term = UNet1(input_img)
            generated_image = multiplicative_term * input_img[:,0,None,:,:]


            #the 1 tensor need to be changed based on the max value in the input images
            if (hparams.loss_type=='SSIM'):
                loss_val = main_loss(generated_image, target_img, torch.tensor([1]).to(device))
            else:
                loss_val = main_loss(generated_image, target_img)
            val_loss[epoch,index] = loss_val.cpu().detach().numpy()
            val_ssim_loss[epoch,index] = 1 - SSIM(generated_image, target_img, torch.tensor([1]).to(device))
            val_nrmse_loss[epoch,index] = NRMSE(generated_image, target_img)
        if (np.mean(val_loss[epoch,:]) < best_val_loss):
            # import time
            # start_time = time.time()
            best_epoch = epoch+1
            # best_UNet = UNet1.clone()
            best_UNet.load_state_dict(UNet1.state_dict())
            best_UNet.eval()
            best_val_loss = np.mean(val_loss[epoch,:])
            # best_weights = local_dir +'/best_weights.pt' 
            # torch.save({
            #     'best_epoch': epoch+1,
            #     'model_state_dict': UNet1.state_dict(),
            #     'optimizer_state_dict': G_optimizer.state_dict(),
            #     'hparams':hparams,
            #     }, best_weights)
            # print("--- %s seconds ---" % (time.time() - start_time))

    # Save models
    import time
    start_time = time.time()
    tosave_weights = local_dir +'/saved_weights.pt' 
    torch.save({
        'epoch': epoch,
        'model_state_dict': UNet1.state_dict(),
        'best_state_dict': best_UNet.state_dict(),
        'best_epoch': best_epoch,
        'optimizer_state_dict': G_optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_nrmse_loss':val_nrmse_loss,
        'val_ssim_loss':val_ssim_loss,
        'hparams': hparams}, tosave_weights)
    print("--- %s seconds ---" % (time.time() - start_time))
    sourceFile = open(local_dir +'/params_used.txt', 'w')
    print('best_epoch', '=', best_epoch, file = sourceFile)
    for arg in vars(hparams):
        print(arg, '=', getattr(hparams, arg), file = sourceFile)
        if(arg=='val_loader'):
            break
    # print(hparams, file = sourceFile)
    sourceFile.close()
    plotter_UNET(hparams,tosave_weights,local_dir,UNet1,train_loader,val_loader)
