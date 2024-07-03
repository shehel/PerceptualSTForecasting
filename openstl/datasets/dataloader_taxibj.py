import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pdb

from openstl.datasets.utils import create_loader


class TaxibjDataset(Dataset):
    """Taxibj <https://arxiv.org/abs/1610.00081>`_ Dataset"""

    def __init__(self, X, Y, use_augment=False, test=False):
        super(TaxibjDataset, self).__init__()
        self.X = (X+1) / 2  # channel is 2
        self.Y = (Y+1) / 2
        self.use_augment = use_augment
        self.mean = 0
        self.std = 1
        self.test = test
        self.perm = False
        self.static_ch = self.X.mean(axis=(0,1),keepdims=True)


    def _augment_seq(self, seqs):
        """Augmentations as a video sequence"""
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index, ::]).float()
        labels = torch.tensor(self.Y[index, ::]).float()
        if self.use_augment:
            len_data = data.shape[0]  # 4
            seqs = self._augment_seq(torch.cat([data, labels], dim=0))
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]

        if self.test:
            low_quantile = 0.05
            high_quantile = 0.95

        else:
            #low_quantile = random.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45])

            #high_quantile = random.choice([0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
            #high_quantile = 1 - low_quantile
            low_quantile = 0.05
            high_quantile = 0.95

        low_quantile = np.repeat(low_quantile, 16*16).reshape(1,16,16)
        high_quantile = np.repeat(high_quantile, 16*16).reshape(1,16,16)
        m_quantile = np.repeat(0.5, 16*16).reshape(1,16,16)
        #quantile = random.choice([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
        # make it into a vector

        quantiles = np.array([low_quantile, m_quantile, high_quantile])
        # create a array of ones of shape 1,1,32,32
        #static_ch = np.ones((1, 1, 32, 32))
        return data, labels, self.static_ch[0], quantiles
def train_collate_fn(batch):
    dynamic_input_batch, target_batch, static_batch, quantiles_batch = zip(*batch)
    dynamic_input_batch = np.stack(dynamic_input_batch, axis=0)
    static_batch = np.stack(static_batch, axis=0)
    quantiles_batch = np.stack(quantiles_batch, axis=0)
    target_batch = np.stack(target_batch, axis=0)
    dynamic_input_batch = torch.from_numpy(dynamic_input_batch).float()
    target_batch = torch.from_numpy(target_batch).float()
    static_batch = torch.from_numpy(static_batch).float()

    ranges = [0,0,0.01,0.02,0.05,0.1]
    ranges_l = [0.5,0.01,0.02,0.05,0.1,0.5]
    # sample between 0 and 5
    # do below only once out of 5 times
    #if random.randint(0, 5) == 0:
    #rng = random.randint(0, 5)
    # use torch equivalent of np.where((static_batch > ranges[rng]) & (static_batch < ranges[rng+1]), 1, 0)
    #static_batch = torch.where((static_batch > ranges[rng]) & (static_batch < ranges_l[rng]), 1, 0)
    #else:
    static_batch = torch.ones_like(static_batch)
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
    # make static_batch ones in the shape of static_batch
    static_batch = torch.ones_like(static_batch)
    # sample between 0 and 5
    # use torch equivalent of np.where((static_batch > ranges[rng]) & (static_batch < ranges[rng+1]), 1, 0)
    quantiles_batch = torch.from_numpy(quantiles_batch).float()

    return dynamic_input_batch, target_batch, static_batch, quantiles_batch


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=None, aft_seq_length=None, in_shape=None,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):

    dataset = np.load(os.path.join(data_root, 'taxibj/dataset.npz'))
    X_train, Y_train, X_test, Y_test = dataset['X_train'], dataset[
        'Y_train'], dataset['X_test'], dataset['Y_test']
    assert X_train.shape[1] == pre_seq_length and Y_train.shape[1] == aft_seq_length
    train_set = TaxibjDataset(X=X_train, Y=Y_train, use_augment=use_augment, test=False)
    test_set = TaxibjDataset(X=X_test, Y=Y_test, use_augment=False, test=True)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher, collate_fn=train_collate_fn)
    dataloader_vali = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher, collate_fn=test_collate_fn)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher, collate_fn=test_collate_fn)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    dataloader_train, _, dataloader_test = \
        load_data(batch_size=16,
                  val_batch_size=4,
                  data_root='../../data/',
                  num_workers=4,
                  pre_seq_length=4, aft_seq_length=4)

    print(len(dataloader_train), len(dataloader_test))
    for item in dataloader_train:
        print(item[0].shape, item[1].shape)
        break
    for item in dataloader_test:
        print(item[0].shape, item[1].shape)
        break
