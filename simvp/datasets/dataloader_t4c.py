import torch
import numpy as np
from torch.utils.data import Dataset

import h5py
import random
from pathlib import Path

from clearml import Dataset
import pdb
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

class T4CDataset(Dataset):
    """Taxibj <https://arxiv.org/abs/1610.00081>`_ Dataset"""

    def __init__(self, 
                 root_dir: str,
                 file_filter: str = None,
                 test:bool = False
                 ):
        self.root_dir = root_dir
        self.file_filter = file_filter
        self.test = test

        if self.file_filter is None:
            self.file_filter = "**/training/*8ch.h5"

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

    def _load_h5_file(self, fn, sl):
        return load_h5_file(fn, sl=sl)

    def __len__(self):
        return self.len


    def __getitem__(self, idx: int):
        if idx > self.__len__():
            raise IndexError("Index out of bounds")

        file_idx = idx // MAX_TEST_SLOT_INDEX
        start_hour = idx % MAX_TEST_SLOT_INDEX

        
        two_hours = self._load_h5_file(self.file_list[file_idx], sl=slice(start_hour, start_hour + 12 * 2 + 1))
        two_hours = np.transpose(two_hours, (0, 3, 1, 2))

        if self.test:
            random_int_x = 10
            random_int_y = 40
        else:
            random_int_x = random.randint(0, 300)
            random_int_y = random.randint(0, 300)
        two_hours = two_hours[:,:,random_int_x:random_int_x + 128, 
                    random_int_y:random_int_y+128, ]

        #input_data, output_data = prepare_test(two_hours)
        dynamic_input, output_data = two_hours[:12], two_hours[12:13]
        #print (dynamic_input[:, 123-10, 61-40, 5])
        #print (output_data[:,123-10, 61-40, 5])

        output_data = output_data[:,1::2,:,:]

        return dynamic_input, output_data 

def train_collate_fn(batch):
    dynamic_input_batch, target_batch = zip(*batch)
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    target_batch = np.stack(target_batch, axis=0)
    dynamic_input_batch = torch.from_numpy(dynamic_input_batch).float()
    target_batch = torch.from_numpy(target_batch).float()


    return dynamic_input_batch, target_batch

def load_data(batch_size, val_batch_size, data_root,
              num_workers=0):

    data_root = "7daysv2"
    try:
        data_root = Dataset.get(dataset_project="t4c", dataset_name=data_root).get_local_copy()
    except:
        print("Could not find dataset in clearml server. Exiting!")
    train_filter = "**/training/*8ch.h5"
    val_filter = "**/validation/*8ch.h5"
    train_set = T4CDataset(data_root, train_filter)
    val_set = T4CDataset(data_root, val_filter)
    test_set = T4CDataset(data_root, val_filter, test=True)

    train_set._load_dataset()
    val_set._load_dataset()
    test_set._load_dataset()

    
    test_set.file_list = [Path('/home/shehel/ml/NeurIPS2021-traffic4cast/data/raw/MOSCOW/validation/2019-01-29_MOSCOW_8ch.h5')]
    test_set.len = 240
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
