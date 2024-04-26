import os, torch, h5py, glob, random
import numpy as np
from torch.utils.data import Dataset

# Multicoil fastMRI dataset with various options
class Dataset1(Dataset):
    def __init__(self, datafiles):
        self.datafiles  = datafiles

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, idx):
        #generates onse sample of data
        ID = self.datafiles[idx]
        X = torch.load('available_data/input/'+ID)
        y = torch.load('available_data/output/'+ID)
        return X,y

""" defining a new class that can read the files from the file location itself, and also
can perform the required transform, as data size is less, transferring all files directly to the GPU

"""
"""
Need to change this dataloader class to be made similar to the way Jon suggested
Maintain a text file for all the training/val/test samples and then take the text file as input to the dataloader
"""
# this class should only take the txt file location
class Exp_contrast_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory location of the text file containing files that need to be in the dataloader.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        with open(root_dir, "r") as text_file:
            files = text_file.readlines()
            files = [file.rstrip() for file in files]
        self.datafiles = files
        # Initialize preloaded data arrays
        self.X_array, self.y_array, self.params_array = [], [], []
        self.T1_array, self.T2_array, self.PD_array = [], [], []
        # self.X_random = np.random.randn(2,288,288) # for giving the same random input always
        # self.X_random = np.random.rand(2,288,288) # for giving the same random input always

        # Walk through all files and preload data
        os.chdir('/csiNAS/sidharth')#change current working directory
        for datafile in self.datafiles:
            loaded_data = torch.load(datafile)
            local_X = np.asarray(loaded_data['input'])
            local_y = np.asarray(loaded_data['output'])
            local_T1 = np.asarray(loaded_data['T1_map'])
            local_T2 = np.asarray(loaded_data['T2_map'])
            local_PD = np.asarray(loaded_data['PD_map'])
            local_TE, local_TR, local_TI = \
                    np.asarray(loaded_data['TE']), np.asarray(loaded_data['TR']), \
                    np.asarray(loaded_data['TI'])
            local_params = [local_TE, local_TR, local_TI]

            # Place in preloaded arrays
            self.X_array.append(local_X)
            self.y_array.append(local_y)
            self.T1_array.append(local_T1)
            self.T2_array.append(local_T2)
            self.PD_array.append(local_PD)
            self.params_array.append(local_params)

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, idx):
        #first check if idx is list or integer
        #if interger then current code works
        # if list then for loop over the list and combine them into tensor and return 
        X = self.X_array[idx]
        y = self.y_array[idx]
        T1_map = self.T1_array[idx]
        T2_map = self.T2_array[idx]
        PD_map = self.PD_array[idx]
        PD_map_real = PD_map.real
        PD_map_imag = PD_map.imag
        params = self.params_array[idx]
        ID = self.datafiles[idx]
        if self.transform is not None:
            X = self.transform(X)
            y = self.transform(y)
            T1_map = self.transform(T1_map)
            T2_map = self.transform(T2_map)
            PD_map_real = self.transform(PD_map_real)
            PD_map_imag = self.transform(PD_map_imag)
        if self.target_transform is not None:# this is for the random horizontal and vertical flips
            rand_toss = np.random.choice(4, 1)
            if (rand_toss == 0):
                X = self.target_transform[0](X)
                y = self.target_transform[0](y)
                T1_map = self.target_transform[0](T1_map)
                T2_map = self.target_transform[0](T2_map)
                PD_map_real = self.target_transform[0](PD_map_real)
                PD_map_imag = self.target_transform[0](PD_map_imag)
            elif (rand_toss == 1):
                X = self.target_transform[1](X)
                y = self.target_transform[1](y)
                T1_map = self.target_transform[1](T1_map)
                T2_map = self.target_transform[1](T2_map)
                PD_map_real = self.target_transform[1](PD_map_real)
                PD_map_imag = self.target_transform[1](PD_map_imag)
            elif (rand_toss == 2):
                X = self.target_transform[2](X)
                y = self.target_transform[2](y)
                T1_map = self.target_transform[2](T1_map)
                T2_map = self.target_transform[2](T2_map)
                PD_map_real = self.target_transform[2](PD_map_real)
                PD_map_imag = self.target_transform[2](PD_map_imag)
        #normalizing with a TI_max value
        TI_max = 3000 #choosen heuristically, none of the TI values are larger than 2500ms
        TI_array = np.ones((X.shape))*params[2]/TI_max
        X = np.stack((X,T1_map,T2_map,PD_map_real,PD_map_imag,TI_array))
        # X = X + 0.1*self.X_random #added 
        # X = self.X_random # X = X + 0.1*self.X_random #added 
        # X = y[None,:,:] # input and output same
        return X.astype(np.float32), y.astype(np.float32), [params,ID]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, object):
        #convert to tensor 
        return torch.from_numpy(object).float()

class horizontal_flip(object):
    def __call__(self, object):
        return np.fliplr(object).copy()#flip left right

class vertical_flip(object):
    def __call__(self, object):
        return np.flipud(object).copy()#flip upside down

class vert_hori_flip(object):
    def __call__(self, object):
        return np.flipud(np.fliplr(object)).copy()#flip upside down and also left right

class rand_rotate(object):
    """ random rotations about the z axis, not much usefull for this case here"""
    def __call__(self, object):
        #convert to tensor 
        rand_toss = np.random.choice(4, 1)
        if(rand_toss == 0):
            return object
        elif(rand_toss == 1):
            return np.rot90(object, 1).copy()
        elif(rand_toss == 2):
            return np.rot90(object, 2).copy()
        elif(rand_toss == 3):
            return np.rot90(object, 3).copy()


class Toreal(object):
    """Convert 1 channel complex values to 2 channel real values."""
    def __call__(self, object):
        return np.concatenate( ( np.real(object)[None,:,:],np.imag(object)[None,:,:]),axis=0)

class Toabsolute(object):
    """Convert 1 channel complex values to absolute values."""
    def __call__(self, object):
        return np.abs(object)


class Normalize_by_max(object):
    """to normalize by the max value."""

    def __call__(self, object):
        #return normalized object 
        return object/np.max(np.abs(object))

class Normalize_by_99(object):
    """To normalize by the 99th and 1st percentile"""
    def __call__(self, object):
        # Normalize the image using percentiles
        lo, hi = np.percentile(np.abs(object), (1, 99))
        return (object - lo) / (hi - lo)

class Normalize_by_90(object):
    """to normalize by the max value."""

    def __call__(self, object):
        #convert to tensor 
        abs_object = np.abs(object)
        max_val_overall = np.max(abs_object)
        max_val_center = np.max(abs_object[100:200,100:200])
        # print(np.min(abs_object),round(np.mean(abs_object),5),round(np.max(abs_object),5))
        # if (max_val_overall > 3*max_val_center):
        #     object[np.abs(object)>max_val_center] = 0
        # object[abs_object>0.1*max_val_overall] = 0
        object = object/np.percentile(object,99)
        object[np.abs(object)>1] = 0
        return object


class Normalize_by_WM(object):
    """to normalize by the max value."""
    def __call__(self, object):
        avg_WM_signal = np.mean(np.abs(object[170:220,100:150]))
        object = object/avg_WM_signal
        object[np.abs(object)>4] = 0
        return object