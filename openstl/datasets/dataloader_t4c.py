import torch
import numpy as np
from torch.utils.data import Dataset

import h5py
import random
from pathlib import Path

from clearml import Dataset
import pdb
from scipy.stats import bernoulli
# implement a load_h5_file function
def load_h5_file(file_path, sl = None, to_torch = False) -> np.ndarray:
    """Given a file path to an h5 file assumed to house a tensor, load that
    tensor into memory and return a pointer.

    Parameters
    ----------
    file_path: str
        h5 file to load
    sl: Optional[slice]
        slice to load (data is written in chunks for faster access to rows).
    """
    # load
    with h5py.File(str(file_path) if isinstance(file_path, Path) else file_path, "r") as fr:
        data = fr.get("array")
        if sl is not None:
            data = np.array(data[sl])
        else:
            data = np.array(data)
        if to_torch:
            data = torch.from_numpy(data)
            data = data.to(dtype=torch.float)
        return data

MAX_TEST_SLOT_INDEX = 240

def create_binary_mask(frame, epsilon=1e-5):
    """
    Create a binary mask by sampling pixels from the input frame with probability proportional to their intensity values.

    Parameters:
        frame (numpy.ndarray): Input 128x128 frame containing values in the range 0-255.
        epsilon (float): A small value to add to the normalized intensity values to make the function numerically stable.
                         Default is 1e-5.

    Returns:
        numpy.ndarray: Binary mask of the same shape as the input frame with randomly activated pixels.
    """
    # Normalize the frame to have values in the range [0, 1]
    normalized_frame = frame / 255.0
    mask = np.random.binomial(1, normalized_frame, size=normalized_frame.shape)

    # Add epsilon to the normalized intensity values to ensure numerical stability

    # Create a binary mask using Bernoulli sampling with probabilities equal to the normalized intensity values

    return mask.astype(int)

def find_largest(matrix, topk):
    # Flatten the matrix to a 1D array
    flat_matrix = matrix.flatten()

    # Step 2: Flatten the matrix and get sorting indices
    flattened = matrix.flatten()
    sorted_indices = np.argsort(-np.abs(flattened))  # Sort by magnitude, descending

    # Step 3: Select top 1000 largest indices
    top_indices = sorted_indices[:topk]

    # Step 4: Randomly sample 500 from top 1000 indices
    random_500_indices = np.random.choice(top_indices, 20, replace=False)

    # Step 5: Create the mask
    mask = np.zeros_like(matrix)
    np.put(mask, random_500_indices, 1)
    return mask
def activate_code(probability):
    """
    Activate a piece of code with a defined probability using Bernoulli sampling from scipy.

    :param probability: The probability of activation (between 0 and 1).
    :return: True if the code should be activated, False otherwise.
    """
    rv = bernoulli(probability)
    return rv.rvs() == 1

class T4CDataset(Dataset):
    """Taxibj <https://arxiv.org/abs/1610.00081>`_ Dataset"""

    def __init__(self, 
                 root_dir: str,
                 file_filter: str = None,
                 test:bool = False,
                 pre_seq_length: int = 12,
                 aft_seq_length: int = 1,
                 perm_bool = False,
                 ):
        self.root_dir = root_dir
        self.file_filter = file_filter
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.perm = perm_bool
        self.mean = 0
        self.std = 1
        self.test = test
        self.m = [0.1197112,  1.73881751, 0.11084138, 1.7988615,  0.1094989,  1.81207202, 0.10324037, 1.71015936]
        self.s = [1.6551339 , 5.82894155, 1.54915965, 6.13199342, 1.51485388, 6.19980411, 1.47532519, 5.83794635]
        # make self.mean and self.std so that it has the shape [None,None,:,None,None
        self.m = np.array(self.m)[None,:,None,None]
        self.s = np.array(self.s)[None,:,None,None] 
        if self.file_filter is None:
            self.file_filter = "**/training/*8ch.h5"

        self.static_filter = "**/*_static.h5"
        self.static_dict = {}
        self.file_data = []
        self.file_list = []
        self.probability = 0.07

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        return data, labels

    def _load_dataset(self):
        self.file_list = list(Path(self.root_dir).rglob(self.file_filter))
        self.file_list.sort()
        self.len = len(self.file_list) * MAX_TEST_SLOT_INDEX
        static_list = list(Path(self.root_dir).rglob(self.static_filter))


        for city in static_list:
             self.static_dict[city.parts[-2]] = load_h5_file(city)
        for i, file in enumerate(self.file_list):
             self.file_data.append(load_h5_file(file))

    def _load_h5_file(self, fn, sl):
        return load_h5_file(fn, sl=sl)

    def __len__(self):
        return self.len


    def __getitem__(self, idx: int):
        if idx > self.__len__():
            raise IndexError("Index out of bounds")

        file_idx = idx // MAX_TEST_SLOT_INDEX
        start_hour = idx % MAX_TEST_SLOT_INDEX

        
        #two_hours = self._load_h5_file(self.file_list[file_idx], sl=slice(start_hour, start_hour + self.pre_seq_length * 2 + 1))
        two_hours = self.file_data[file_idx][start_hour:start_hour + self.pre_seq_length * 2 + 1]
        #two_hours = two_hours
        #two_hours = (two_hours - np.min(two_hours)) * (200 / (np.max(two_hours) - np.min(two_hours)))
        #
        #two_hours = (two_hours - two_hours.min()) / (two_hours.max() - two_hours.min())

        # Rescale values to be between 50 and 200
        #two_hours = 0 + (two_hours * (20 - 0))
        two_hours = np.transpose(two_hours, (0, 3, 1, 2))

        # TODO
        if self.test:
            random_int_x = 312
            random_int_y = 68
        else:
            random_int_x = 312
            random_int_y = 68

            # random_int_x = random.randint(0, 300)
            # random_int_y = random.randint(0, 300)
        two_hours = two_hours[:,:,random_int_x:random_int_x + 128, 
                    random_int_y:random_int_y+128, ]

        #two_hours = (two_hours - self.m) / self.s
        #two_hours = two_hours/255
    
        dynamic_input, output_data = two_hours[:self.pre_seq_length], two_hours[self.pre_seq_length:self.pre_seq_length+self.aft_seq_length]
        static_ch = self.static_dict[self.file_list[file_idx].parts[-3]]
        #static_ch = static_ch/255
        # get mean of of dynamic input across first axis
        inp_mean = np.mean(dynamic_input, axis=0)
        # remove mean from output data
        #output_data = output_data - inp_mean
        #static_ch = inp_mean[4,:,:]
        output_data = output_data[:,0::1,:,:]
        static_ch = static_ch[0, random_int_x:random_int_x+128, random_int_y:random_int_y+128]
        #static_ch = static_ch/static_ch.max()
        #static_ch = np.ones((128,128))
        if self.test:
            static_ch = np.where(static_ch > 0, 1,0)
            #a = 1
        else:
            #static_ch = find_largest(inp_mean[4], 1000)
            static_ch = np.where(static_ch > 0, 1,0)
            #a = 1


        static_ch = static_ch[np.newaxis, np.newaxis, :, :]
        

        # zero out all but [:,:,52,76] in output_data
        # dynamic_input[:,:,0:64,:] = 0
        # dynamic_input[:,:,65:,:] = 0
        # dynamic_input[:,:,:,0:64] = 0
        # dynamic_input[:,:,:,65:] = 0

        # zero out all but a 5x5 patch around 64,64 in static_ch
        #static_ch = static_ch[0,0,59:69,59:69]

        return dynamic_input, output_data, static_ch

def train_collate_fn(batch):
    dynamic_input_batch, target_batch, static_batch = zip(*batch)
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    static_batch = np.stack(static_batch, axis=0)
    target_batch = np.stack(target_batch, axis=0)
    dynamic_input_batch = torch.from_numpy(dynamic_input_batch).float()
    target_batch = torch.from_numpy(target_batch).float()
    static_batch = torch.from_numpy(static_batch).float()


    return dynamic_input_batch, target_batch, static_batch


def load_data(batch_size, val_batch_size, data_root,
              num_workers=0, pre_seq_length=None, aft_seq_length=None,
              in_shape=None, distributed=False, use_prefetcher=False,use_augment=False):

    try:
        #data_root = Dataset.get(dataset_id="20fef9fe5f0b49319a7f380ae16d5d1e").get_local_copy() # berlin_full
        #data_root = Dataset.get(dataset_id="6ecb9b57d2034556829ebeb9c8a99d63").get_local_copy() # berlin_full
        data_root = Dataset.get(dataset_id="59b1fd80e3274676aeba314c832bbd85").get_local_copy()
        #data_root = Dataset.get(dataset_id="efd30aa3795f4f498fb4f966a4aec93b").get_local_copy()
    except:
        print("Could not find dataset in clearml server. Exiting!")
    train_filter = "**/training/*8ch.h5"
    val_filter = "**/validation/*8ch.h5"
    train_set = T4CDataset(data_root, train_filter, pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length, test=False)
    val_set = T4CDataset(data_root, val_filter, pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length, test=True)
    test_set = T4CDataset(data_root, val_filter, pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length, test=True)


    train_set._load_dataset()
    val_set._load_dataset()
    test_set._load_dataset()

    
    #test_set.file_list = [Path('/home/jeschneider/Documents/data/raw/MOSCOW/validation/2019-01-29_MOSCOW_8ch.h5')]
    #test_set.file_list = [Path('/data/raw/ANTWERP/training/2019-06-25_ANTWERP_8ch.h5')]
    #test_set.file_list = [Path('/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw/ANTWERP/training/2020-04-25_ANTWERP_8ch.h5')]
    #test_set.file_list = [Path('/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw/BERLIN/training/2019-06-25_BERLIN_8ch.h5')]
    test_set.file_list = [Path('/home/jeschneider/Documents/data/raw/BERLIN/training/2019-06-25_BERLIN_8ch.h5'),]
                          #Path('/home/jeschneider/Documents/data/raw/MOSCOW/validation/2020-06-25_MOSCOW_8ch.h5'),
                          #  Path('/home/jeschneider/Documents/data/raw/ISTANBUL/training/2019-06-25_ISTANBUL_8ch.h5'),
                          #  Path('/home/jeschneider/Documents/data/raw/ANTWERP/training/2019-06-25_ANTWERP_8ch.h5'),]
    #test_set.file_list = [Path('/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw/MOSCOW/validation/2020-06-25_MOSCOW_8ch.h5')]
    #test_set.file_list = [Path('/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw/MOSCOW/validation/2019-04-06_MOSCOW_8ch.h5')]
    #test_set.file_list = [Path('/data/raw/MOSCOW/validation/2019-01-29_MOSCOW_8ch.h5')]
    #test_set.file_list = [Path('/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw/MOSCOW/validation/2019-01-29_MOSCOW_8ch.h5')]

    test_set.len = 240*len(test_set.file_list)
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

if __name__ == '__main__':
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=4,
                  val_batch_size=4,
                  data_root='7daysv2',
                  num_workers=4,
                  pre_seq_length=12, aft_seq_length=1)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
