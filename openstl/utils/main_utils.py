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

from . import soft_dtw
from . import path_soft_dtw


class QuantileRegressionLoss(nn.Module):
    def __init__(self, params):
        super(QuantileRegressionLoss, self).__init__()

        self.q_lo_loss = PinballLoss()
        self.q_hi_loss = PinballLoss()

        self.q_lo_weight = params['q_lo_weight']
        self.q_hi_weight = params['q_hi_weight']
        self.mse_weight = params['mse_weight']

    def forward(self, pred, target, static_ch, quantile):
        lo_loss = self.q_lo_weight * self.q_lo_loss(pred[:,0,:,:,:].squeeze(), target.squeeze(), quantile[:,0:1], static_ch)
        hi_loss = self.q_hi_weight * self.q_hi_loss(pred[:,2,:,:,:].squeeze(), target.squeeze(), quantile[:,2:3], static_ch)
        #m_loss = self.mse_weight * self.mse_loss(pred[:,1,:,:,:].squeeze(), target.squeeze(), static_ch)
        #m_loss = self.mse_weight * torch.mean(F.mse_loss(pred[:,1].squeeze(), target.squeeze(), reduction='none') * static_ch)
        loss = lo_loss + hi_loss
        return loss

class PinballLoss():
    def __init__(self, reduction='mean'):
        self.reduction = reduction

    def __call__(self, output, target, quantile, static_ch):
        try:
            assert output.shape == target.shape, "Output and target must have the same shape."
            assert output.shape[0] == quantile.shape[0], "Quantile must match the batch size of output and target."
        except:
            pdb.set_trace()
        loss = torch.zeros_like(target, dtype=torch.float)
        error = output - target
        error = error*static_ch

        smaller_index = error < 0
        bigger_index = error > 0

        # Apply quantile to each batch element
        quantile_expanded = quantile.view(-1, 1, 1, 1, 1).expand_as(error)  # Adjust dimensions to match error

        loss[smaller_index] = quantile_expanded[smaller_index] * torch.abs(error[smaller_index])
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

  def __call__(self, output, target, static_ch):
      assert output.shape == target.shape
      loss = torch.zeros_like(target, dtype=torch.float)
      error = (output - target)
      error = error*static_ch
      smaller_index = error < 0
      bigger_index = 0 < error
      loss[smaller_index] = self.quantile * ((abs(error))[smaller_index])
      loss[bigger_index] = (1-self.quantile) * ((abs(error))[bigger_index])

      if self.reduction == 'sum':
        loss = loss.sum()
      if self.reduction == 'mean':
        loss = loss.mean()

      return loss
def ccc(gold, pred):
    #gold = torch.squeeze(gold, dim=2)
    #pred = torch.squeeze(pred, dim=2)

    gold_mean = torch.mean(gold, dim=1, keepdim=True)
    pred_mean = torch.mean(pred, dim=1, keepdim=True)

    covariance = torch.mean((gold - gold_mean) * (pred - pred_mean), dim=1, keepdim=True)

    gold_var = torch.mean((gold - gold_mean) ** 2, dim=1, keepdim=True)
    pred_var = torch.mean((pred - pred_mean) ** 2, dim=1, keepdim=True)

    ccc = 2. * covariance / (gold_var + pred_var + (gold_mean - pred_mean) ** 2 + torch.finfo(torch.float32).eps)
    return ccc

def ccc_loss(gold, pred, mask):
    return (1. - torch.mean(ccc(gold, pred) * mask[:]))

def mis_loss_func(
    y_pred: torch.tensor, y_true: torch.tensor, interval: float
) -> torch.tensor:
    """Calculate MIS loss

    Args:
        y_pred (torch.tensor): Predicted values
        y_true (torch.tensor): True values
        interval (float): confidence interval (e.g. 0.95 for 95% confidence interval)

    Returns:
        torch.tensor: output losses
    """
    alpha = (interval[:,0]) + (1- interval[:,2])
    alpha = alpha.view(-1, 1, 1, 1, 1)  # Reshape alpha to match the batch dimension and broadcast
    lower = y_pred[:, 0]
    upper = y_pred[:, 2]
    loss = upper - lower
    loss = torch.max(loss, loss + (2 / alpha) * (lower - y_true))
    loss = torch.max(loss, loss + (2 / alpha) * (y_true - upper))
    loss = torch.mean(loss)

    return loss

def eval_quantiles(lower, upper, trues, preds, static_ch):
    N = static_ch.sum()*4

    icp = torch.sum((trues > lower) & (trues < upper)).float() / N
    diffs = torch.maximum(torch.zeros_like(upper), upper - lower)
    mil = torch.sum(diffs) / N
    # rmil = 0.0
    # diffs_flat = diffs.view(-1)
    # trues_flat = trues.view(-1)
    # preds_flat = preds.view(-1)

    # for i in range(N):
    #     if trues_flat[i] != preds_flat[i]:
    #         rmil += diffs_flat[i] / torch.abs(trues_flat[i] - preds_flat[i])

    # rmil = rmil / N
    # clc = torch.exp(-rmil * (icp - 0.95))

    return icp, mil
class DifferentialDivergenceLoss(nn.Module):
    def __init__(self, tau=1, epsilon=1e-8, w1=1, w2 =1, w3=1, w4=1, w5=1):
        super(DifferentialDivergenceLoss, self).__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.w1, self.w2, self.w3, self.w4, self.w5 = w1, w2, w3, w4, w5
        #self.main_loss = nn.L1Loss()
        self.main_loss = nn.MSELoss()
        #self.ssim = SSIM(window_size=4)
        self.q_loss = QuantileRegressionLoss(params={"q_lo": 0.05, "q_hi": 0.95, "q_lo_weight": 1, "q_hi_weight": 1, "mse_weight": 1})
        #self.pixels = [(42, 23), (45, 26), (43, 24), (28, 28), (22, 25), (24, 26), (8, 71), (26, 27), (44, 25), (27, 28), (19, 24), (25, 27), (18, 24), (20, 24), (41, 22), (21, 25), (23, 26), (65, 90), (48, 34), (66, 91), (46, 27), (53, 42), (57, 47), (54, 43), (62, 56), (51, 39), (69, 109), (49, 36), (50, 37), (70, 111), (60, 53), (56, 46), (48, 33), (67, 104), (52, 40), (7, 71), (71, 114), (58, 48), (72, 127), (61, 54), (67, 93), (29, 28), (40, 22), (41, 23), (68, 106), (62, 57), (6, 70), (61, 55), (64, 89)]
        self.pixel_list = [(64, 64),
            (64, 65),
            (36, 83),
            (63, 86),
            (67, 94),
            (58, 49),
            (50, 37),
            (42, 95),
            (60, 90)]
        #self.pixels = [(25, 15), (30, 24), (28, 21), (16, 2), (22, 11), (24, 14), (21, 10), (29, 22), (19, 7), (26, 16), (33, 58), (16, 1), (18, 5), (34, 59), (29, 23), (17, 4), (30, 25), (31, 27), (20, 8), (13, 31), (27, 19), (31, 28), (27, 18), (31, 26), (35, 61), (17, 3), (23, 12), (0, 62), (32, 57), (31, 29), (32, 32), (23, 13), (36, 63), (32, 31), (31, 30), (18, 6), (32, 33), (15, 0), (26, 17), (32, 34), (28, 20), (31, 54), (35, 60), (32, 35), (21, 9), (3, 62), (20, 9), (7, 51), (31, 52), (32, 36)]
        # Create indices for fixed positions
        self.pixels = self.pixel_list[:]
        self.pixels = torch.tensor(self.pixels, dtype=torch.long)
        self.cent_loss = PinballLossFixed(quantile=0.5)


    def forward(self, pred, true, static_ch, quantiles, train_run=True):
        #true_std = torch.std(true, dim=1, keepdim=False)
        #true_std = true_std/((torch.max(true_std)+self.epsilon))

        # add an axis to static_ch at 1 and multiply it by pred to get new pred
        pred = pred * torch.unsqueeze(static_ch, 1)
        mse_loss = self.cent_loss(pred[:,1,:,:,:].squeeze(), true.squeeze(), static_ch)
        sum_loss = self.q_loss(pred, true, static_ch[:,], quantiles)
        std_loss = mis_loss_func(pred, true, quantiles)
        reg_mse, reg_std = eval_quantiles(pred[:,0], pred[:,2], true, pred[:,1], static_ch)
        # mae loss using functional
        #std_loss = self.ssim(pred, true)
        #mse_loss = dilate_loss(pred[:,:,0,64:65,64], true[:,:,0,64:65,64], alpha=0.1, gamma=0.001, device=pred.device)[0]
        #mse_loss = 1- ssim3D(pred, true, static_ch=static_ch)

        #sampled_true, sampled_pred = sample_top_pixels_modified(true, pred, static_ch)
        # binatrize static_ch
        #static_ch = torch.where(static_ch > 0, torch.ones_like(static_ch), torch.zeros_like(static_ch))
        #mse_loss, reg_mse, reg_std = self.main_loss(pred[:,:,]*static_ch, true[:,:]*static_ch)
        #sum_loss, std_loss, mse_loss = ssim3D(pred, true, static_ch=static_ch)
        #sum_loss = mse_of_spatial_std(pred[:,:,0], true[:,:,0], static_ch)
        #reg_std = self.main_loss(pred*static_ch, true*static_ch)
        # pred = pred * static_ch
        # true = true * static_ch
        #sum_loss = self.main_loss(pred*static_ch, true*static_ch)
        #mse_loss = 1 - ssim(pred[:,:,0], true[:,:,0], data_range=255, size_average=True, win_size=7) # return a scalar
        # sum_1 = torch.sum(pred, dim=(3,4))[:,:,0]
        # sum_2 = torch.sum(true, dim=(3,4))[:,:,0]
        # sum_1 = row_standardization(sum_1)
        # sum_2 = row_standardization(sum_2)
        # sum_loss = self.main_loss(sum_1, sum_2)
        #true_ratios = calculate_ratios(true)
        #predicted_ratios = calculate_ratios(pred)
        #sum_loss = self.main_loss(true_ratios,predicted_ratios)
        #pred = pred * static_ch
        #true = true * static_ch
        #pdb.set_trace()
        # sum_loss = torch.mean(F.mse_loss(pred, true, reduction='none') * static_ch)
        # mae_loss = torch.mean(F.l1_loss(pred, true, reduction='none') * static_ch)
        # if train_run==False:
        #     print ("L1 ", mae_loss)
        #     print ("MSE")
        #     for i in self.pixel_list:
        #         mse_px = F.mse_loss(pred[:,:,:,i[0],i[1]], true[:,:,:,i[0],i[1]])
        #         print (i, mse_px)

        #     print ("STD")
        #     for i in self.pixel_list:
        #         std_px = F.mse_loss(torch.std(pred[:,:,:,i[0],i[1]], dim=1), torch.std(true[:,:,:,i[0],i[1]], dim=1))
        #         print (i, std_px)
        # #sum_loss = mse_loss
        # if train_run==False:
        #     print ("DILATE")
        #     # for i in self.pixel_list:

        #     #     sampled_true, sampled_pred = sample_pixels_efficient(true, pred, [i])
        #     #     sampled_true = (sampled_true - sampled_true.mean(dim=1, keepdim=True)) / (sampled_true.std(dim=1, keepdim=True) + self.epsilon)
        #     #     sampled_pred = (sampled_pred - sampled_pred.mean(dim=1, keepdim=True)) / (sampled_pred.std(dim=1, keepdim=True) + self.epsilon)
        #     #     reg_mse, reg_std = dilate_loss(sampled_pred, sampled_true, alpha=0.1, gamma=0.001, device=pred.device)
        #     #     print (i, reg_mse, reg_std, ((reg_mse)*0.001 + reg_std))
        #     sampled_true, sampled_pred = sample_pixels_efficient(true, pred, self.pixel_list)
        #     sampled_true = (sampled_true - sampled_true.mean(dim=1, keepdim=True)) / (sampled_true.std(dim=1, keepdim=True) + self.epsilon)
        #     sampled_pred = (sampled_pred - sampled_pred.mean(dim=1, keepdim=True)) / (sampled_pred.std(dim=1, keepdim=True) + self.epsilon)
        #     reg_mse, reg_std = dilate_loss(sampled_pred, sampled_true, alpha=0.1, gamma=0.001, device=pred.device)

        # else:
        #     # sampled_true, sampled_pred = sample_pixels_efficient(true, pred, self.pixels)
        #     # reg_mse, reg_std = dilate_loss(sampled_pred, sampled_true, alpha=0.1, gamma=0.001, device=pred.device)

        #     reg_mse = mse_loss
        #     reg_std = mse_loss
        # #pred_prob = F.softmax(sum_1, dim=1)
        # #true_prob = F.softmax(sum_2, dim=1)

        # # do kl div loss on pred_prob and true_prob using F.kl_div
        # #sum_loss = F.kl_div(torch.log(pred_prob + self.epsilon), true_prob, reduction='batchmean')

        # # std_loss = F.mse_loss(torch.std(pred, dim=1), torch.std(true, dim=1), reduction='none')
        # # std_loss = torch.mean(std_loss * static_ch[:,0])

        # #sum_loss = std_loss
        # #sum_loss = F.mse_loss(torch.sum(pred, dim=1), torch.sum(true,dim=1))
        # #reg_std = modified_total_variation_loss(pred[:,:,0], true[:,:,0])#F.mse_loss(torch.std(pred, dim=1), torch.std(true, dim=1))
        # #reg_std = modified_total_variation_loss(mu1[:,:,0]*static_ch[:,0],
        #  #                                       mu2[:,:,0]*static_ch[:,0])#F.mse_loss(torch.std(pred, dim=1), torch.std(true, dim=1))
        # #std_loss = SSIM(pred[:,:,0], true[:,:,0])#F.mse_loss(torch.std(pred, dim=1), torch.std(true, dim=1))
        # #sum_loss = mse_of_spatial_cov(pred[:,:,0], true[:,:,0])#F.mse_loss(torch.std(pred, dim=1), torch.std(true, dim=1))
        # pred_diff = pred[:, 1:,:,:,:] - pred[:, :-1,:,:,:]
        # true_diff = true[:, 1:,:,:,:] - true[:, :-1,:,:,:]
        # # multiply pred_diff and true_diff with static_ch
        # pred_diff = pred_diff * static_ch[:, :,:,:,:]
        # true_diff = true_diff * static_ch[:, :,:,:,:]
        # #pdb.set_trace()
        # std_loss = F.mse_loss(torch.std(pred, dim=1), torch.std(true, dim=1), reduction='none')
        # std_loss = torch.mean(std_loss * static_ch[:,0])

        #std_loss = ccc_loss(true,pred, static_ch)
                # square the difference and sum over all pixels
        # take the absolute value of the difference
        
        # if train_run==True:
        #     # get the abs value of the diff and sum across 1st dimension preserving the shape
        #     #pdb.set_trace()
        #     pred_diff1 = ((pred_diff)**2).sum(dim=1, keepdim=True)
        #     true_diff1 = ((true_diff)**2).sum(dim=1, keepdim=True)
        #     # pred_diff1 = ((pred_diff)**2).sum(dim=1, keepdim=True)
        #     # true_diff1 = ((true_diff)**2).sum(dim=1, keepdim=True)
        #     check_loss = F.mse_loss(pred_diff1, true_diff1, reduction='none')
        #     check_loss = torch.mean(check_loss * static_ch)

        #     sum_loss = F.mse_loss(pred_diff1, true_diff1)

        #     #pred_diff = (pred_diff**2).sum(dim=1, keepdim=True)
        #     #true_diff = (true_diff**2).sum(dim=1, keepdim=True)



            
        #     #pred_diff = (pred_diff**2).sum()/(static_ch[0].sum()*pred.shape[0]*pred.shape[1])
        #     #true_diff = (true_diff**2).sum()/(static_ch[0].sum()*pred.shape[0]*pred.shape[1])
        # # get the abs value of the diff
        # # take mean squared error of the difference
        #     #sum_loss = F.mse_loss(pred_diff, true_diff, reduction='mean')
        # else:
        #     print ("Sum Total")
        #     pred_diffx = (pred_diff**2).sum(dim=1, keepdim=True)
        #     true_diffx = (true_diff**2).sum(dim=1, keepdim=True)
        #     pred_diffy = (torch.abs(pred_diff)).sum(dim=1, keepdim=True)
        #     true_diffy = (torch.abs(true_diff)).sum(dim=1, keepdim=True)

            
        #     # get the abs value of the diff
        #     sum_l1_loss = F.l1_loss(pred_diffx, true_diffx)

        #     sum_loss = F.mse_loss(pred_diffx, true_diffx)
        #     print (sum_loss)
        #     print (sum_l1_loss)
        #     c_loss = ccc_loss(true[:,:,:,self.pixels[:,0], self.pixels[:,1]],pred[:,:,:,self.pixels[:,0], self.pixels[:,1]],
        #                     static_ch[:,:,:,self.pixels[:,0], self.pixels[:,1]])
        #     print (c_loss)

        # if train_run==False:

        #     print ("Sum")
        #     for i in self.pixel_list:
        #         pred_diff = pred[:, 1:,:,i[0],i[1]] - pred[:, :-1,:,i[0],i[1]]
        #         true_diff = true[:, 1:,:,i[0],i[1]] - true[:, :-1,:,i[0],i[1]]
        #         # square the difference and sum over all pixels
        #         #pred_diff = torch.abs(pred_diff).sum(dim=1, keepdim=True)
        #         #true_diff = torch.abs(true_diff).sum(dim=1, keepdim=True)
        #         pred_diff = (pred_diff**2).sum(dim=1, keepdim=True)
        #         true_diff = (true_diff**2).sum(dim=1, keepdim=True)
        #         # get the abs value of the diff
        #         print (i, F.mse_loss(pred_diff, true_diff))
        #         print (i, F.l1_loss(pred_diff, true_diff))
        #         # squared error

        #true_diff = true_diff
        #pred_diff = pred_diff.view(pred_diff.shape[0], pred_diff.shape[1], -1)
        #true_diff = true_diff.reshape(true_diff.shape[0], true_diff.shape[1], -1)
        #pred_prob = F.softmax(pred_diff / self.tau, dim=2)
        #true_prob = F.softmax(true_diff / self.tau, dim=2)

        # reg_mse = torch.mean(F.mse_loss(pred_diff, true_diff, reduction='none') * static_ch[:, :, :, :, :])
        # reg_std = F.mse_loss(torch.std(pred_diff, dim=1), torch.std(true_diff,dim=1), reduction='none')
        # reg_std = torch.mean(reg_std * static_ch[:, 0, :, :, :])


        # get KL between pred_prob and true_prob
        #sum_loss = torch.sum(true_prob * torch.log(true_prob / pred_prob), dim=1).mean()

        #sum_loss = self.main_loss(pred_prob, true_prob)

        train_loss = self.w1 * (mse_loss) + self.w2 * (reg_mse) + self.w3 * (reg_std) + self.w4 * std_loss + self.w5*sum_loss
        # check if train loss is nan
        if torch.any(torch.isnan(train_loss)):
            pdb.set_trace()
        total_loss = mse_loss + reg_mse + reg_std + std_loss + self.w5 * sum_loss
        return [train_loss, total_loss,  self.w1*mse_loss, self.w2*reg_mse, self.w3*reg_std, self.w4*std_loss, sum_loss*self.w5]

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
