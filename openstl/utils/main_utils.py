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

from torch.autograd import Variable
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pdb


class QuantileRegressionLoss(nn.Module):
    def __init__(self, quantile_weights):
        super(QuantileRegressionLoss, self).__init__()
        self.pinball_loss = PinballLoss()
        self.quantile_weights = quantile_weights

    def forward(self, pred, target, mask, quantiles):
        total_loss = 0
        individual_losses = []
        middle_index = len(self.quantile_weights) // 2
        for i, q_weight in enumerate(self.quantile_weights):
            q_pred = pred[:, i, :, :, :].squeeze()
            # skip if quantile is 0.5
            if quantiles[0,i] == 0.5:
                continue
            q_quantile = quantiles[:, i:i+1]
            loss = q_weight * self.pinball_loss(q_pred, target.squeeze(), q_quantile, mask)
            individual_losses.append(loss)
            total_loss += loss
        mae_loss = torch.mean(torch.abs(pred[:, middle_index, :, :, :] - target))
        individual_losses.append(mae_loss)
        total_loss += mae_loss
        return total_loss, individual_losses

class PinballLoss():
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, output, target, quantile, mask):
        try:
            assert output.shape == target.shape, "Output and target must have the same shape."
            assert output.shape[0] == quantile.shape[0], "Quantile must match the batch size of output and target."
        except:
            pdb.set_trace()
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        error = error*mask

        smaller_index = error < 0
        bigger_index = error > 0

        # Apply quantile to each batch element
        quantile_expanded = quantile.view(-1, 1, 1, 1, 1).expand_as(error)  # Adjust dimensions to match error

        try:
            loss[smaller_index] = quantile_expanded[smaller_index] * torch.abs(error[smaller_index])
        except:
            pdb.set_trace()
        loss[bigger_index] = (1 - quantile_expanded[bigger_index]) * torch.abs(error[bigger_index])

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class PinballLossFixed():

  def __init__(self, quantile=0.10, reduction='mean'):
      self.quantile = quantile
      assert 0 < self.quantile
      assert self.quantile < 1
      self.reduction = reduction

  def __call__(self, output, target, mask):
      assert output.shape == target.shape
      loss = torch.zeros_like(target, dtype=torch.float)
      error = (output - target)
      error = error*mask
      smaller_index = error < 0
      bigger_index = 0 < error
      loss[smaller_index] = self.quantile * ((abs(error))[smaller_index])
      loss[bigger_index] = (1-self.quantile) * ((abs(error))[bigger_index])

      if self.reduction == 'sum':
        loss = loss.sum()
      if self.reduction == 'mean':
        loss = loss.mean()

      return loss

def mis_loss_func(
    y_pred: torch.tensor, y_true: torch.tensor, alpha: float
) -> torch.tensor:
    """Calculate MIS loss

    Args:
        y_pred (torch.tensor): Predicted values
        y_true (torch.tensor): True values
        alpha (float): 1-confidence interval (e.g. 0.05 for 95% confidence interval)

    Returns:
        torch.tensor: output losses
    """
    alpha = torch.tensor(alpha)
    alpha = alpha.view(-1, 1, 1, 1, 1)  # Reshape alpha to match the batch dimension and broadcast
    lower = y_pred[:, 0]
    upper = y_pred[:, 1]
    loss = upper - lower
    loss = torch.max(loss, loss + (2 / alpha) * (lower - y_true))
    loss = torch.max(loss, loss + (2 / alpha) * (y_true - upper))
    loss = torch.mean(loss)

    return loss



# As per definition in Dewolf paper
def eval_quantiles(lower, upper, trues,mask, time_step=1):
    N = mask.sum()*time_step

    icp = torch.sum((trues > lower) & (trues < upper)).float() / N
    diffs = torch.abs(upper - lower)
    mil = torch.sum(diffs) / N

    return icp, mil

class IntervalScores(nn.Module):
    def __init__(self, quantile_weights):
        super(IntervalScores, self).__init__()
        self.quantile_loss_fn = QuantileRegressionLoss(quantile_weights)
        self.mis_loss_fn = WeightedMISLoss(quantile_weights)

    def forward(self, pred, true, mask, quantiles, train_run=True, loss_type='quantile'):
        pred = pred * torch.unsqueeze(mask, 1)
        if loss_type == 'quantile':
            total_loss, individual_losses = self.quantile_loss_fn(pred, true, mask, quantiles)
        elif loss_type == 'mis':
            total_loss, individual_losses = self.mis_loss_fn(pred, true, mask, quantiles)
        else:
            raise ValueError("Invalid loss_type. Choose 'quantile' or 'mis'.")
        return total_loss, individual_losses

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


class WeightedMISLoss(nn.Module):
    def __init__(self, quantile_weights):
        super(WeightedMISLoss, self).__init__()
        self.quantile_weights = quantile_weights

    def forward(self, y_pred, y_true, mask, quantiles):
        total_loss = 0
        individual_losses = []

        num_quantiles = len(self.quantile_weights)
        middle_index = num_quantiles // 2

        for i in range(middle_index):
            lower_idx = i
            upper_idx = num_quantiles - 1 - i

            lower = y_pred[:, lower_idx, :, :, :]
            upper = y_pred[:, upper_idx, :, :, :]

            alpha = 1 - (quantiles[:, upper_idx] - quantiles[:, lower_idx])

            loss = mis_loss_func(torch.stack([lower, upper], dim=1), y_true, alpha)
            loss = loss * mask
            loss = torch.mean(loss)

            weight = (self.quantile_weights[lower_idx] + self.quantile_weights[upper_idx]) / 2
            weighted_loss = weight * loss

            individual_losses.append(weighted_loss)
            total_loss += weighted_loss

        # also include a MAE loss
        # TODO: use mask
        mae_loss = torch.mean(torch.abs(y_pred[:, middle_index, :, :, :] - y_true))
        individual_losses.append(mae_loss)
        total_loss = (total_loss + mae_loss) #/ (middle_index + 1)
        return total_loss, individual_losses
