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

        # Walk through all files and preload data
        os.chdir('/home/sidharth/sid_notebooks/')#change current working directory
        for datafile in self.datafiles:
            loaded_data = torch.load(datafile)
            local_X = np.asarray(loaded_data['input'])
            local_y = np.asarray(loaded_data['output'])
            local_TE, local_TR, local_TI = \
                    np.asarray(loaded_data['TE']), np.asarray(loaded_data['TR']), \
                    np.asarray(loaded_data['TI'])
            local_params = [local_TE, local_TR, local_TI]

            # Place in preloaded arrays
            self.X_array.append(local_X)
            self.y_array.append(local_y)
            self.params_array.append(local_params)

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, idx):
        #first check if idx is list or integer
        #if interger then current code works
        # if list then for loop over the list and combine them into tensor and return 
        X = self.X_array[idx]
        y = self.y_array[idx]
        params = self.params_array[idx]
        ID = self.datafiles[idx]
        if self.transform is not None:
            X = self.transform(X)
            y = self.transform(y)
        TI_array = np.ones((X.shape))*params[2]
        X = np.stack((X,TI_array))
        return X.astype(np.float32), y.astype(np.float32), [params,ID]


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, object):
        #convert to tensor 
        return torch.from_numpy(object).float()

class vertical_flip(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, object):
        return np.rot90(object, 2).copy()

class horizontal_flip(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, object):
        return np.rot90(object, 1).copy()

class horizontal_flip2(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, object):
        return np.rot90(object, 3).copy()

class rand_rotate(object):
    """Convert ndarrays in sample to Tensors."""

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