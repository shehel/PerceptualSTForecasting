# Copyright (c) CAIRI AI Lab. All rights reserved

import pdb
import cv2
import os
import logging
import platform
import random
import subprocess
import sys
import warnings
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Tuple

import torch
import torchvision
from torch.nn import functional as F
from torch import nn
import torch.multiprocessing as mp
from torch import distributed as dist
from math import exp
import openstl
from .config_utils import Config

import pdb

def normalize_image(image, range_min, range_max):
    # Normalize the image pixel values to the range [0, 1]
    return (image - range_min) / (range_max - range_min)

def gaussian(window_size, sigma):
    gauss = torch.tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# def create_window_3D(window_size, channel):
#     _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
#     _2D_window = _1D_window.mm(_1D_window.t())
#     _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
#     window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
#     return window

def create_window_3D(window_size, channel):
    # Create a 3D window filled with ones
    _3D_window = torch.ones((window_size, 1, 1)).float().unsqueeze(0).unsqueeze(0)
    window = _3D_window.expand(channel, 1, window_size, 1, 1).contiguous()
    return window/window.sum()

def _ssim_3D(img1, img2, window, window_size, channel, size_average=False):
    #padding = window_size // 2
    #mu1 = F.conv3d(img1, window, padding=padding, groups=channel)
    #mu2 = F.conv3d(img2, window, padding=padding, groups=channel)

    padding = (window_size - 1) # Fixing the padding calculation here

    mu1 = F.conv3d(img1, window, padding='same', groups=channel) # Applying padding only to the temporal dimension
    mu2 = F.conv3d(img2, window, padding='same', groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding ='same', groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding ='same', groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding ='same', groups = channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    eps = 1e-8
    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1 + eps)


# Ensure variances are non-negative
    sigma1_sq = torch.clamp(sigma1_sq, min=eps)
    sigma2_sq = torch.clamp(sigma2_sq, min=eps)

# Contrast component with added stability
    contrast = (2 * torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C2) / (sigma1_sq + sigma2_sq + C2 + eps)

# Structure component with added stability
    structure_denominator = torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C2 / 2 + eps
    structure = (sigma12 + C2 / 2) / structure_denominator

# Overall SSIM
    ssim_map = structure #* contrast

    if size_average:
        calc = ssim_map.mean()
        # if torch.isnan(calc) or torch.isinf(calc):
        #     pdb.set_trace()

    else:
        calc = ssim_map.mean(1).mean(1).mean(1)
        # if torch.isnan(calc) or torch.isinf(calc):
        #     pdb.set_trace()

    return calc


class SSIM3D(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        # switch 1 and 2 dimensions for img1 and img2 so that data becomes batchsize, channels, timesteps, height, width
        img1 = img1.permute(0, 2, 1, 3, 4)
        img2 = img2.permute(0, 2, 1, 3, 4)
        (_, channel, _, _, _) = img1.size()

        if torch.isnan(img1.sum()):
            pdb.set_trace()
        if channel == self.channel and self.window.device == img1.device and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)

            if img1.is_cuda:
                window = window.to(img1.device)
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        # clip img1 and img2 to be between 0 and 255
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)
        #img1 = normalize_image(img1, 0, 255)
        #img2 = normalize_image(img2, 0, 255)

        calc = _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)
        # check if calc is torch nan or inf and call pdb
        return calc
def row_standardization(matrix):
    # Calculate mean and standard deviation along axis 1 (rows)
    mean = torch.mean(matrix, dim=1, keepdim=True)
    std = torch.std(matrix, dim=1, keepdim=True)

    # Apply row-wise standardization
    standardized_matrix = (matrix - mean) / (std + 1e-7)  # Adding a small epsilon to avoid division by zero

    return standardized_matrix

def calculate_ratios(image_timeseries):
    # image_timeseries shape: (batch, 6, 1, 128, 128)

    # Step 1: Sum the pixel values for each timestep
    pixel_sums_per_timestep = torch.sum(image_timeseries, dim=(2, 3, 4))  # Shape: (batch, 6)

    # Step 2: Calculate the ratios
    total_pixel_sum = torch.sum(pixel_sums_per_timestep, dim=1, keepdim=True)  # Shape: (batch, 1)

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    ratios = pixel_sums_per_timestep / (total_pixel_sum + epsilon)

    return ratios

class DifferentialDivergenceLoss(nn.Module):
    def __init__(self, tau=1, epsilon=1e-8, w1=1, w2 =1, w3=1, w4=1, w5=2000):
        super(DifferentialDivergenceLoss, self).__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.w1, self.w2, self.w3, self.w4, self.w5 = w1, w2, w3, w4, w5
        #self.main_loss = nn.L1Loss()
        self.main_loss = nn.MSELoss()
        self.ssim = SSIM3D(window_size=4)

    def forward(self, pred, true):
        # mae loss using functional
        #std_loss = self.ssim(pred, true)
        mse_loss = self.main_loss(pred, true)
        # sum_1 = torch.sum(pred, dim=(3,4))[:,:,0]
        # sum_2 = torch.sum(true, dim=(3,4))[:,:,0]
        # sum_1 = row_standardization(sum_1)
        # sum_2 = row_standardization(sum_2)
        # sum_loss = self.main_loss(sum_1, sum_2)
        true_ratios = calculate_ratios(true)
        predicted_ratios = calculate_ratios(pred)
        sum_loss = self.main_loss(true_ratios,predicted_ratios)


        #pred_prob = F.softmax(sum_1, dim=1)
        #true_prob = F.softmax(sum_2, dim=1)

        # do kl div loss on pred_prob and true_prob using F.kl_div
        #sum_loss = F.kl_div(torch.log(pred_prob + self.epsilon), true_prob, reduction='batchmean')

        std_loss = F.mse_loss(torch.std(pred, dim=1), torch.std(true, dim=1))

        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = true[:, 1:] - true[:, :-1]
        #pred_diff = pred_diff.view(pred_diff.shape[0], pred_diff.shape[1], -1)
        #true_diff = true_diff.reshape(true_diff.shape[0], true_diff.shape[1], -1)

        #pred_prob = F.softmax(pred_diff / self.tau, dim=2)
        #true_prob = F.softmax(true_diff / self.tau, dim=2)
        reg_mse = F.mse_loss(pred_diff, true_diff)
        reg_std = F.mse_loss(torch.std(pred_diff, dim=1), torch.std(true_diff,dim=1))

        # get KL between pred_prob and true_prob
        #sum_loss = torch.sum(true_prob * torch.log(true_prob / pred_prob), dim=1).mean()

        #sum_loss = self.main_loss(pred_prob, true_prob)

        train_loss = self.w1 * mse_loss + self.w2 * reg_mse + self.w3 * reg_std + self.w4 * std_loss + self.w5*sum_loss
        # check if train loss is nan
        if torch.any(torch.isnan(train_loss)):
            pdb.set_trace()
        total_loss = mse_loss + reg_mse + reg_std + std_loss + self.w5 * sum_loss

        return train_loss, total_loss,  mse_loss, reg_mse, reg_std, std_loss, sum_loss*self.w5
def set_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def setup_multi_processes(cfg):
    """Setup multi-processing environment variables."""
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        current_method = mp.get_start_method(allow_none=True)
        if current_method is not None and current_method != mp_start_method:
            warnings.warn(
                f'Multi-processing start method `{mp_start_method}` is '
                f'different from the previous setting `{current_method}`.'
                f'It will be force set to `{mp_start_method}`. You can change '
                f'this behavior by changing `mp_start_method` in your config.')
        mp.set_start_method(mp_start_method, force=True)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if 'OMP_NUM_THREADS' not in os.environ and cfg['num_workers'] > 1:
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and cfg['num_workers'] > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


def collect_env():
    """Collect the information of the running environments."""
    env_info = {}
    env_info['sys.platform'] = sys.platform
    env_info['Python'] = sys.version.replace('\n', '')

    cuda_available = torch.cuda.is_available()
    env_info['CUDA available'] = cuda_available

    if cuda_available:
        from torch.utils.cpp_extension import CUDA_HOME
        env_info['CUDA_HOME'] = CUDA_HOME

        if CUDA_HOME is not None and os.path.isdir(CUDA_HOME):
            try:
                nvcc = os.path.join(CUDA_HOME, 'bin/nvcc')
                nvcc = subprocess.check_output(
                    '"{}" -V | tail -n1'.format(nvcc), shell=True)
                nvcc = nvcc.decode('utf-8').strip()
            except subprocess.SubprocessError:
                nvcc = 'Not Available'
            env_info['NVCC'] = nvcc

        devices = defaultdict(list)
        for k in range(torch.cuda.device_count()):
            devices[torch.cuda.get_device_name(k)].append(str(k))
        for name, devids in devices.items():
            env_info['GPU ' + ','.join(devids)] = name

    gcc = subprocess.check_output('gcc --version | head -n1', shell=True)
    gcc = gcc.decode('utf-8').strip()
    env_info['GCC'] = gcc

    env_info['PyTorch'] = torch.__version__
    env_info['PyTorch compiling details'] = torch.__config__.show()
    env_info['TorchVision'] = torchvision.__version__
    env_info['OpenCV'] = cv2.__version__

    env_info['openstl'] = openstl.__version__

    return env_info


def print_log(message):
    print(message)
    logging.info(message)


def output_namespace(namespace):
    configs = namespace.__dict__
    message = ''
    for k, v in configs.items():
        message += '\n' + k + ': \t' + str(v) + '\t'
    return message


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True


def get_dataset(dataname, config):
    from openstl.datasets import dataset_parameters
    from openstl.datasets import load_data
    config.update(dataset_parameters[dataname])
    return load_data(**config)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_throughput(model, input_dummy):

    def get_batch_size(H, W):
        max_side = max(H, W)
        if max_side >= 128:
            bs = 10
            repetitions = 1000
        else:
            bs = 100
            repetitions = 100
        return bs, repetitions

    if isinstance(input_dummy, tuple):
        input_dummy = list(input_dummy)
        _, T, C, H, W = input_dummy[0].shape
        bs, repetitions = get_batch_size(H, W)
        _input = torch.rand(bs, T, C, H, W).to(input_dummy[0].device)
        input_dummy[0] = _input
        input_dummy = tuple(input_dummy)
    else:
        _, T, C, H, W = input_dummy.shape
        bs, repetitions = get_batch_size(H, W)
        input_dummy = torch.rand(bs, T, C, H, W).to(input_dummy.device)
    total_time = 0
    with torch.no_grad():
        for _ in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            if isinstance(input_dummy, tuple):
                _ = model(*input_dummy)
            else:
                _ = model(input_dummy)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * bs) / total_time
    return Throughput


def load_config(filename:str = None):
    """load and print config"""
    print('loading config from ' + filename + ' ...')
    try:
        configfile = Config(filename=filename)
        config = configfile._cfg_dict
    except (FileNotFoundError, IOError):
        config = dict()
        print('warning: fail to load the config!')
    return config


def update_config(args, config, exclude_keys=list()):
    """update the args dict with a new config"""
    assert isinstance(args, dict) and isinstance(config, dict)
    for k in config.keys():
        if args.get(k, False):
            if args[k] != config[k] and k not in exclude_keys and args[k] is not None:
                print(f'overwrite config key -- {k}: {config[k]} -> {args[k]}')
            else:
                args[k] = config[k]
        else:
            args[k] = config[k]
    return args


def weights_to_cpu(state_dict: OrderedDict) -> OrderedDict:
    """Copy a model state_dict to cpu.

    Args:
        state_dict (OrderedDict): Model weights on GPU.

    Returns:
        OrderedDict: Model weights on GPU.
    """
    state_dict_cpu = OrderedDict()
    for key, val in state_dict.items():
        state_dict_cpu[key] = val.cpu()
    # Keep metadata in state_dict
    state_dict_cpu._metadata = getattr(  # type: ignore
        state_dict, '_metadata', OrderedDict())
    return state_dict_cpu


def init_dist(launcher: str, backend: str = 'nccl', **kwargs) -> None:
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.
    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.
    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()


def _init_dist_pytorch(backend: str, **kwargs) -> None:
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_mpi(backend: str, **kwargs) -> None:
    local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    if 'MASTER_PORT' not in os.environ:
        # 29500 is torch.distributed default port
        os.environ['MASTER_PORT'] = '29500'
    if 'MASTER_ADDR' not in os.environ:
        raise KeyError('The environment variable MASTER_ADDR is not set')
    os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
    os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    dist.init_process_group(backend=backend, **kwargs)


def get_dist_info() -> Tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def reduce_tensor(tensor):
    rt = tensor.data.clone()
    dist.all_reduce(rt.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return rt
