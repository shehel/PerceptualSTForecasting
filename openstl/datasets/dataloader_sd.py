import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

import h5py
import random
from pathlib import Path

from clearml import Dataset
import pdb
from scipy.stats import bernoulli
from scipy.ndimage import convolve
# implement a load_h5_file function

import os


perm = [[0,1,2,3],
        [1,2,3,0],
        [2,3,0,1],
        [3,0,1,2]
        ]

class StandardScaler():
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)


    def transform(self, data):
        return (data - self.mean) / self.std


    def inverse_transform(self, data):
        return (data * self.std) + self.mean
        
import pickle
class SDDataset(Dataset):
    def __init__(self, root_dir: str, s_file_filter: int = None, e_file_filter: int = None,  test: bool = False,
                 pre_seq_length: int = 12, aft_seq_length: int = 12, perm_bool=False):
        self.root_dir = root_dir
        self.s_file_filter = s_file_filter
        self.e_file_filter = e_file_filter
        self.test = test
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.perm_bool = perm_bool
        self.mean = 0
        self.std = 0
        self.data, self.scaler, self.static_ch, self.pixel_list, self.pixel_list_val = self._load_dataset()
        # add a empty channel for static data
        self.static_ch = np.expand_dims(self.static_ch, axis=0)
        self.perm = True
        self.dir = 0
        

    def _load_dataset(self):
        ptr = np.load(os.path.join(self.root_dir, 'his_images.npz'))
        # add 2 empty dimensions at the beginning to mean and std
        # put mean and std to device
        
        mean= (ptr['mean'][0,62,61:62])
        std= (ptr['std'][0,62,61:62])
        # convert mean and std to torch tensors and put it in cuda

        mean = torch.from_numpy(mean).cuda()
        std = torch.from_numpy(std).cuda()
        
        scaler = StandardScaler(mean, std)
        # load pickle file called occupied_pixels_px.pkl
        pixels = pickle.load(open(os.path.join(self.root_dir, 'occupied_pixels_px.pkl'), 'rb'))
        # convert # loss to numpy array but map third tuple of every element to number based on N,E,S,W corresponding to 0,1,2,3
        array_pixels = {0:[], 1:[], 2:[], 3:[]}
        for i in pixels.keys():
            temp_loss = []
            dir = 0 if i[2] == 'N' else 1 if i[2] == 'E' else 2 if i[2] == 'S' else 3  
            temp_loss.append(i[0])
            # 19 is the offset for padding to 128 in sd
            temp_loss.append(i[1]+19)
            #temp_loss.append(dir)
            #array_pixels.append(temp_loss)
            array_pixels[dir].append(temp_loss)
        
        array_pixels_val = []
        for i in pixels.keys():
            temp_loss = []
            dir = 0 if i[2] == 'N' else 1 if i[2] == 'E' else 2 if i[2] == 'S' else 3  
            temp_loss.append(i[0])
            # 19 is the offset for padding to 128 in sd
            temp_loss.append(i[1]+19)
            temp_loss.append(dir)
            array_pixels_val.append(temp_loss)
        array_pixels_val = np.array(array_pixels_val)
        
        #array_pixels = np.array(array_pixels)
        # pixel list is a list of tuples of the form (0, 69, 'N'). I want to extract only the first two elements of the tuple.
        # make it a numpy array of shape (num_pixels, 2) and add 19 to each element like [:,1]+=19
        # add 2 empty dimensions at the beginning to static_ch
        

        return ptr['data'][self.s_file_filter:self.e_file_filter], scaler, ptr['static_ch'], array_pixels, array_pixels_val

    def __len__(self):
        return len(self.data) - (self.pre_seq_length + self.aft_seq_length)

    def __getitem__(self, idx: int):
        if idx + self.pre_seq_length + self.aft_seq_length > len(self.data):
            raise IndexError("Index out of bounds")
        two_hours = self.data[idx:idx + self.pre_seq_length + self.aft_seq_length]
        dir_sel = self.dir
        if self.perm:
            dir_sel = random.randint(0,3)
            two_hours = two_hours[:,perm[dir_sel]]
        dynamic_input_batch, target_batch = two_hours[:self.pre_seq_length], two_hours[self.pre_seq_length:]
        if self.perm:
            target_batch = target_batch[:,0:1]
        #static_batch = self.data[idx]  # Example static data, modify as needed
        return dynamic_input_batch, target_batch, self.pixel_list[dir_sel]

def train_collate_fn(batch):
    dynamic_input_batch, target_batch, static_batch = zip(*batch)
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    target_batch = np.stack(target_batch, axis=0)
    #static_batch = np.concatenate(static_batch, axis=0)
    
    # append batch index to every element of static_batch. Static_batch consists of 32 lists each with several lists. 
    # I want to append batch index to every element of the inner lists.
    static_batch = [[np.append(static_batch[i][j], i) for j in range(len(static_batch[i]))] for i in range(len(static_batch))]

       
    # stack all static_batches which includes list of lists in into a single numpy array
    static_batch = np.concatenate(static_batch, axis=0)
    # make static_batch a int32 array
    static_batch = static_batch.astype(np.int32)

    dynamic_input_batch = torch.from_numpy(dynamic_input_batch).float()
    target_batch = torch.from_numpy(target_batch).float()
    static_batch = torch.from_numpy(static_batch)


    return dynamic_input_batch, target_batch, static_batch
def load_data(batch_size, val_batch_size, data_root, num_workers=0, pre_seq_length=None, aft_seq_length=None,
              in_shape=None, distributed=False, use_prefetcher=False, use_augment=False):
    try:
        #data_root = Dataset.get(dataset_id="20fef9fe5f0b49319a7f380ae16d5d1e").get_local_copy() # berlin_full
        #data_root = Dataset.get(dataset_id="6ecb9b57d2034556829ebeb9c8a99d63").get_local_copy() # berlin_full
        data_root = Dataset.get(dataset_id="24135ccf7b1740158b7ca43deca2c114").get_local_copy()
        #data_root = Dataset.get(dataset_id="efd30aa3795f4f498fb4f966a4aec93b").get_local_copy()
    except:
        print("Could not find dataset in clearml server. Exiting!")
    
    filters = []
    for cat in ['train', 'val', 'test']:
        idx = np.load(os.path.join(data_root, 'idx_' + cat + '.npy'))
        filters.append(idx)
    train_set = SDDataset(data_root, 0, filters[1][0], test=False, pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
    val_set = SDDataset(data_root, filters[1][0], filters[2][0],  test=True, pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
    test_set = SDDataset(data_root, filters[2][0], filters[2][-1]+1, test=True, pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)
   # test_set=val_set

    # For simplicity, the test dataloader is the same as the validation one.
    dataloader_train = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size, shuffle=True,
                                                   pin_memory=True, drop_last=True,
                                                   num_workers=num_workers,
                                                   collate_fn = train_collate_fn)
    dataloader_vali = torch.utils.data.DataLoader(val_set,
                                                  batch_size=val_batch_size, shuffle=False,
                                                  pin_memory=True, drop_last=True,
                                                  num_workers=num_workers,
                                                  collate_fn = train_collate_fn
                                                  )
    dataloader_test = torch.utils.data.DataLoader(test_set,
                                                  batch_size=val_batch_size, shuffle=False,
                                                  pin_memory=True, drop_last=True,
                                                  num_workers=num_workers,
                                                  collate_fn = train_collate_fn)

    return dataloader_train, dataloader_vali, dataloader_test


    return dataloader_train, dataloader_vali, dataloader_test

if __name__ == '__main__':
    # Example usage
    batch_size = 32
    val_batch_size = 32
    data_root = 'sdv0'
    pre_seq_length = 12
    aft_seq_length = 12

    dataloader_train, dataloader_vali, dataloader_test = load_data(batch_size, val_batch_size, data_root, pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length)

    # Example for iterating over the train loader
    for dynamic_input_batch, target_batch, static_batch in dataloader_train:
        # Process batches
        pass
