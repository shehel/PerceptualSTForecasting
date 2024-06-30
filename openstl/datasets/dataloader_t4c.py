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
    # doubt this works
    # seed reset every time for determinism
    np.random.seed(40)
    # Flatten the matrix to a 1D array
    flat_matrix = matrix.flatten()

    # Step 2: Flatten the matrix and get sorting indices
    flattened = matrix.flatten()
    sorted_indices = np.argsort(-np.abs(flattened))  # Sort by magnitude, descending

    sort_lim = int(topk/0.25)
    # Step 3: Select top 1000 largest indices
    top_indices = sorted_indices[:sort_lim]

    # Step 4: Randomly sample 500 from top 1000 indices
    random_500_indices = np.random.choice(top_indices, topk, replace=False)
    #random_500_indices = top_indices[:topk]
    # Step 5: Create the mask
    mask = np.zeros_like(matrix)
    np.put(mask, random_500_indices, 1)
    return mask

# def find_largest(matrix, topk):
#     """
#     Create a mask of 1s and 0s where the topk values in the matrix are set to 1 and the rest to 0.

#     Parameters:
#         matrix (numpy.ndarray): Input 128x128 matrix.
#         topk (int): The number of highest values to be set to 1 in the mask.

#     Returns:
#         numpy.ndarray: A mask of the same shape as the input matrix with topk values set to 1 and the rest to 0.
#     """
#     # Flatten the matrix and get the indices of the topk values
#     flat_matrix = matrix.flatten()
#     top_indices = np.argpartition(-flat_matrix, topk)[:topk]

#     # Create a mask with all zeros
#     mask = np.zeros_like(flat_matrix)

#     # Set the topk indices to 1
#     mask[top_indices] = 1
#     pdb.set_trace()
#     # Reshape the mask to the original matrix shape
#     mask = mask.reshape(matrix.shape)

#     return mask

def activate_code(probability):
    """
    Activate a piece of code with a defined probability using Bernoulli sampling from scipy.

    :param probability: The probability of activation (between 0 and 1).
    :return: True if the code should be activated, False otherwise.
    """
    rv = bernoulli(probability)
    return rv.rvs() == 1

from datetime import datetime
# Step 1: Extract day of the week from filenames
def get_day_of_week(filename):
    date_str = filename.name.split('_')[0]
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    return date_obj.strftime("%A")

perm = [[0,1,2,3,4,5,6,7],
        [2,3,4,5,6,7,0,1],
        [4,5,6,7,0,1,2,3],
        [6,7,0,1,2,3,4,5]
        ]
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
        self.pixel_list = np.array([
            [64, 64],
            [64, 65],
            [36, 83],
            [63, 86],
            [67, 94],
            [58, 49],
            [50, 37],
            [42, 95],
            [60, 90]
        ])


        self.perm = False

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        return data, labels

    def _load_dataset(self, set="train"):
        self.file_list = list(Path(self.root_dir).rglob(self.file_filter))
        #jself.file_list = self.file_list[:1]
        self.file_list.sort()

        self.len = len(self.file_list) * MAX_TEST_SLOT_INDEX
        static_list = list(Path(self.root_dir).rglob(self.static_filter))

        self.weekday_mean = load_h5_file(Path(self.root_dir) / "BERLIN/weekday_mean_trainval.h5")
        # float 32
        self.weekday_mean = self.weekday_mean.astype(np.float32)
        self.weekend_mean = load_h5_file(Path(self.root_dir) / "BERLIN/weekend_mean_trainval.h5")
        self.weekend_mean = self.weekend_mean.astype(np.float32)
        # self.weekday_std = load_h5_file(Path(self.root_dir) / "BERLIN/weekday_std.h5")
        # # float 32da
        # self.weekday_std = self.weekday_std.astype(np.float32)
        # self.weekend_std = load_h5_file(Path(self.root_dir) / "BERLIN/weekend_std.h5")
        # self.weekend_std = self.weekend_std.astype(np.float32)

        for city in static_list:
             self.static_dict[city.parts[-2]] = load_h5_file(city)
        for i, file in enumerate(self.file_list):
             self.file_data.append(load_h5_file(file).astype(np.float32))
        last_city = load_h5_file(self.file_list[-1]).astype(np.float32)
        inp_mean = np.mean(last_city[:,:,:,0::2], axis=0)[:,:]
        # move last dim to first
        inp_mean = np.moveaxis(inp_mean, -1, 0)
        # expand inp_mean to have shape [1,128,128]
        #inp_mean = inp_mean[np.newaxis,:,:]
        
        self.static_dict[self.file_list[-1].parts[-3]] = inp_mean

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
        #two_hours = two_hours.astype(np.float32)
        day = get_day_of_week(self.file_list[file_idx])

        # if day == "Saturday" or day == "Sunday":
        #     two_hours = (two_hours - self.weekend_mean[start_hour:start_hour + self.pre_seq_length * 2 + 1])#/(self.weekend_std[start_hour:start_hour + self.pre_seq_length * 2 + 1]+1e-6)
        # else:
        #     two_hours = (two_hours - self.weekday_mean[start_hour:start_hour + self.pre_seq_length * 2 + 1])#/(self.weekday_std[start_hour:start_hour + self.pre_seq_length * 2 + 1]+1e-6)

        #two_hours = two_hours
        #two_hours = (two_hours - np.min(two_hours)) * (200 / (np.max(two_hours) - np.min(two_hours)))
        #
        #two_hours = (two_hours - two_hours.min()) / (two_hours.max() - two_hours.min())

        # Rescale values to be between 50 and 200
        #two_hours = 0 + (two_hours * (20 - 0))
        two_hours = np.transpose(two_hours, (0, 3, 1, 2))

        if self.perm:
            dir_select = random.randint(0,3)
            #dir_select = 2

            two_hours = two_hours[:,perm[dir_select],:,:]

        # sample two quantiles between 0 and 1. It should be a multiple of 5 and shouldn't include 0.5. Lower quantile should be smaller than 0.5 and higher quantile should be larger than 0.5
        if self.test:
            low_quantile = 0.05
            high_quantile = 0.95

        else:
            low_quantile = random.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])

            #high_quantile = random.choice([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
            high_quantile = 1 - low_quantile
        low_quantile = np.repeat(low_quantile, 32*32).reshape(1,32,32)
        high_quantile = np.repeat(high_quantile, 32*32).reshape(1,32,32)
        m_quantile = np.repeat(0.5, 32*32).reshape(1,32,32)
        #quantile = random.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
        # make it into a vector

        quantiles = np.array([low_quantile, m_quantile, high_quantile])
            # random_int_x = random.randint(0, 300)
            # random_int_y = random.randint(0, 300)
        #two_hours = two_hours[:,:,random_int_x:random_int_x + 64,
        #            random_int_y:random_int_y+64, ]

        #two_hours = (two_hours - self.m) / self.s
        #two_hours = two_hours/255
    
        dynamic_input, output_data = two_hours[:self.pre_seq_length], two_hours[self.pre_seq_length:self.pre_seq_length+self.aft_seq_length]
        if dynamic_input[:,4,:,:].max() > 255:
            pdb.set_trace()
        static_ch = self.static_dict[self.file_list[file_idx].parts[-3]]
        #static_ch = static_ch/255
        # get mean of of dynamic input across first axis
        #inp_mean = np.mean(dynamic_input[:,0::2], axis=0)
        # remove mean from output data
        #output_data = output_data - inp_mean
        if self.perm:
            inp_static_ch = static_ch[dir_select,:,:]
            # add 3 more channels but make them all zeros
            inp_static_ch = np.stack([inp_static_ch, np.zeros_like(inp_static_ch), np.zeros_like(inp_static_ch), np.zeros_like(inp_static_ch)], axis=0)
            output_data = output_data[:,0::1,:,:]
        else:
            output_data = output_data[:,0::1,:,:]
            #inp_static_ch = inp_mean[2,:,:]
            # TODO make it dynamic
            inp_static_ch = static_ch[:,:,:]
        #inp_static_ch = inp_mean[:,:,:]

        #static_ch = static_ch[0, random_int_x:random_int_x+64, random_int_y:random_int_y+64]
        #static_ch = static_ch/static_ch.sum()
        static_ch = np.zeros((4,128,128))
        static_ch[:,64-32:64+32,64-32:64+32] = inp_static_ch[:,64-32:64+32,64-32:64+32]
        static_ch = np.abs(static_ch)
        #pxs = 500
        # ranges

        if self.test:
            # static_ch_m = find_largest(static_ch, pxs)
            # static_ch = static_ch * static_ch_m
            #static_ch = np.where(static_ch > 15, 1,0)
            static_ch = np.where((static_ch > 0) & (static_ch < 1), 1, 0)

            a = 1
        else:
            # static_ch_m = find_largest(static_ch, pxs)
            # static_ch = static_ch * static_ch_m

            #static_ch = np.where((static_ch > 15) & (static_ch < 20), 1, 0)
            #static_ch = np.where(static_ch > 0, 1,0)
            a = 1

        
        #static_ch[:,self.pixel_list[:, 0], self.pixel_list[:, 1]] = 1
        static_ch = static_ch[np.newaxis, :,:, :]
        

        # zero out all but [:,:,52,76] in output_data
        # dynamic_input[:,:,0:64,:] = 0
        # dynamic_input[:,:,65:,:] = 0
        # dynamic_input[:,:,:,0:64] = 0
        # dynamic_input[:,:,:,65:] = 0

        # zero out all but a 5x5 patch around 64,64 in static_ch
        #static_ch = static_ch[0,0,59:69,59:69]
        return dynamic_input, output_data, static_ch, quantiles

def train_collate_fn(batch):
    dynamic_input_batch, target_batch, static_batch, quantiles_batch = zip(*batch)
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    static_batch = np.stack(static_batch, axis=0)
    quantiles_batch = np.stack(quantiles_batch, axis=0)
    target_batch = np.stack(target_batch, axis=0)
    dynamic_input_batch = torch.from_numpy(dynamic_input_batch).float()
    target_batch = torch.from_numpy(target_batch).float()
    static_batch = torch.from_numpy(static_batch).float()
    ranges = [0,0,1,5,10,15]
    ranges_l = [255,1,5,10,15,255]
    # sample between 0 and 5
    rng = random.randint(0,5)
    # use torch equivalent of np.where((static_batch > ranges[rng]) & (static_batch < ranges[rng+1]), 1, 0)
    static_batch = torch.where((static_batch > ranges[rng]) & (static_batch < ranges_l[rng]), 1, 0)
    quantiles_batch = torch.from_numpy(quantiles_batch).float()

    return dynamic_input_batch, target_batch, static_batch, quantiles_batch
def test_collate_fn(batch):
    dynamic_input_batch, target_batch, static_batch, quantiles_batch = zip(*batch)
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    static_batch = np.stack(static_batch, axis=0)
    quantiles_batch = np.stack(quantiles_batch, axis=0)
    target_batch = np.stack(target_batch, axis=0)
    dynamic_input_batch = torch.from_numpy(dynamic_input_batch).float()
    target_batch = torch.from_numpy(target_batch).float()
    static_batch = torch.from_numpy(static_batch).float()
    # sample between 0 and 5
    # use torch equivalent of np.where((static_batch > ranges[rng]) & (static_batch < ranges[rng+1]), 1, 0)
    quantiles_batch = torch.from_numpy(quantiles_batch).float()

    return dynamic_input_batch, target_batch, static_batch, quantiles_batch


def load_data(batch_size, val_batch_size, data_root,
              num_workers=0, pre_seq_length=None, aft_seq_length=None,
              in_shape=None, distributed=False, use_prefetcher=False,use_augment=False):
    try:
        #data_root = Dataset.get(dataset_id="20fef9fe5f0b49319a7f380ae16d5d1e").get_local_copy() # berlin_full
        #data_root = Dataset.get(dataset_id="6ecb9b57d2034556829ebeb9c8a99d63").get_local_copy() # berlin_full
        data_root = Dataset.get(dataset_id="0ac6a015bb804f30a092c089fa7b28ea").get_local_copy()
        #data_root = Dataset.get(dataset_id="446207d29ccd48368e3a5a3d63d2feaa").get_local_copy()
        #data_root = Dataset.get(dataset_id="efd30aa3795f4f498fb4f966a4aec93b").get_local_copy()
    except:
        print("Could not find dataset in clearml server. Exiting!")
    train_filter = "**/training/*8ch.h5"
    val_filter = "**/validation/*8ch.h5"
    test_filter = "**/test/*8ch.h5"
    train_set = T4CDataset(data_root, train_filter, pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length, test=False)
    val_set = T4CDataset(data_root, val_filter, pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length, test=True)
    test_set = T4CDataset(data_root, test_filter, pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length, test=True)


    train_set._load_dataset()
    val_set._load_dataset()
    test_set._load_dataset()

    #test_set.file_list = [Path('/home/jeschneider/Documents/data/raw/MOSCOW/validation/2019-01-29_MOSCOW_8ch.h5')]
    #test_set.file_list = [Path('/data/raw/ANTWERP/training/2019-06-25_ANTWERP_8ch.h5')]
    #test_set.file_list = [Path('/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw/ANTWERP/training/2020-04-25_ANTWERP_8ch.h5')]
    #test_set.file_list = [Path('/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw/BERLIN/training/2019-06-25_BERLIN_8ch.h5')]
    #test_set.file_list = [Path('/home/jeschneider/Documents/data/raw/BERLIN/training/2019-06-25_BERLIN_8ch.h5'),]
                          #Path('/home/jeschneider/Documents/data/raw/MOSCOW/validation/2020-06-25_MOSCOW_8ch.h5'),
                          #  Path('/home/jeschneider/Documents/data/raw/ISTANBUL/training/2019-06-25_ISTANBUL_8ch.h5'),
                          #  Path('/home/jeschneider/Documents/data/raw/ANTWERP/training/2019-06-25_ANTWERP_8ch.h5'),]
    #test_set.file_list = [Path('/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw/MOSCOW/validation/2020-06-25_MOSCOW_8ch.h5')]
    #test_set.file_list = [Path('/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw/MOSCOW/validation/2019-04-06_MOSCOW_8ch.h5')]
    #test_set.file_list = [Path('/data/raw/MOSCOW/validation/2019-01-29_MOSCOW_8ch.h5')]
    #test_set.file_list = [Path('/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw/MOSCOW/validation/2019-01-29_MOSCOW_8ch.h5')]

    dataloader_train = torch.utils.data.DataLoader(train_set,
                                                   batch_size=batch_size, shuffle=True,
                                                   pin_memory=True, drop_last=True,
                                                   num_workers=num_workers,
                                                   collate_fn = train_collate_fn)
    dataloader_vali = torch.utils.data.DataLoader(val_set,
                                                  batch_size=val_batch_size, shuffle=False,
                                                  pin_memory=True, drop_last=True,
                                                  num_workers=num_workers,
                                                  collate_fn = test_collate_fn
                                                  )
    dataloader_test = torch.utils.data.DataLoader(test_set,
                                                  batch_size=val_batch_size, shuffle=False,
                                                  pin_memory=True, drop_last=True,
                                                  num_workers=num_workers,
                                                  collate_fn = test_collate_fn)

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
