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

# def dilate_loss(outputs, targets, alpha, gamma, device):
# 	# outputs, targets: shape (batch_size, N_output, 1)
# 	batch_size, N_output = outputs.shape[0:2]
# 	loss_shape = 0
# 	softdtw_batch = soft_dtw.SoftDTWBatch.apply
# 	D = torch.zeros((batch_size, N_output,N_output )).to(device)
# 	for k in range(batch_size):
# 		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
# 		D[k:k+1,:,:] = Dk
# 	loss_shape = softdtw_batch(D,gamma)

# 	path_dtw = path_soft_dtw.PathDTWBatch.apply
# 	path = path_dtw(D,gamma)
# 	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
# 	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output)
# 	loss = alpha*loss_shape+ (1-alpha)*loss_temporal
# 	return loss, loss_shape, loss_temporal

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

# def create_window_3D(window_size, channel):
#     # Create a 3D mean kernel where each value is 1 / (window_size * window_size * window_size)
#     _3D_window = torch.ones((window_size, window_size, window_size)) / (window_size ** 3)
#     _3D_window = _3D_window.float().unsqueeze(0).unsqueeze(0)

#     # Expand the kernel across the specified number of channels
#     window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())

#     return window
# def _ssim_3D(img1, img2, window, window_size, channel, size_average=True, static_ch=None):
#     mu1 = F.conv3d(img1, window, padding=window_size//2, groups=channel)
#     mu2 = F.conv3d(img2, window, padding=window_size//2, groups=channel)
#     #ssim_map = F.mse_loss(mu1*static_ch, mu2*static_ch, reduction='mean')
#     #return ssim_map
#     mu1_sq = mu1.pow(2)

#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
#     sigma12 = F.conv3d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

#     C1 = 0.01 ** 2
#     C2 = 0.03 ** 2



#     ssim_map_comp1 = F.mse_loss(mu1*static_ch, mu2*static_ch, reduction='mean')
#     # take the difference squared of squar root of sigma1 and sigma2 and divide by sigma1*sigma2 + C2
#     ssim_map_comp2 = ((torch.sqrt(sigma1_sq)-torch.sqrt(sigma2_sq))**2)+C1/(sigma1_sq + sigma2_sq + C1)
#     ssim_map_comp3 = (sigma12 + C2)/(torch.sqrt(sigma1_sq)*torch.sqrt(sigma2_sq) + C2)
#     #pdb.set_trace()
#     # ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
#     # #ssim_map = ssim_map * static_ch
#     # if size_average:
#     #     return ssim_map.mean()
#     # else:
#     #     return ssim_map.mean(1).mean(1).mean(1)
#     return ssim_map_comp1#*0.2+(1-ssim_map_comp2.mean())+(1-ssim_map_comp3.mean())

# def _ssim_3D(img1, img2, window, window_size, channel, size_average=True, static_ch=None):
#     if static_ch is None:
#         raise ValueError("static_ch must be provided")

#     mu1 = F.conv3d(img1, window, padding=window_size//2, groups=channel)
#     mu2 = F.conv3d(img2, window, padding=window_size//2, groups=channel)

#     mu1_sq = mu1.pow(2)
#     mu2_sq = mu2.pow(2)
#     mu1_mu2 = mu1 * mu2

#     sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
#     sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
#     sigma12 = F.conv3d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

#     # Adding small constants for stability
#     C1 = 1e-6
#     C2 = 1e-6

#     # Checking for non-negative sigma values
#     sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
#     sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

#     ssim_map_comp1 = F.mse_loss(mu1 * static_ch, mu2 * static_ch, reduction='mean')

#     ssim_map_comp2 = (torch.sqrt(sigma1_sq) - torch.sqrt(sigma2_sq))**2 / (sigma1_sq + sigma2_sq + C1)
#     ssim_map_comp3 = 1 - (sigma12 + C2) / (torch.sqrt(sigma1_sq) * torch.sqrt(sigma2_sq) + C2)

#     return 0.2 * ssim_map_comp1 + ssim_map_comp2.mean() + ssim_map_comp3.mean()
def _ssim_3D(img1, img2, window, window_size, channel, size_average=True, static_ch=None):
    if static_ch is None:
        raise ValueError("static_ch must be provided")

    mu1 = F.conv3d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv3d(img2, window, padding=window_size//2, groups=channel)
    m_loss =modified_total_variation_loss(mu1[:,0,:], mu2[:,0,:])

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    # Adding small constants for stability
    sigma1_sq = torch.clamp(sigma1_sq, min=1e-8)
    sigma2_sq = torch.clamp(sigma2_sq, min=1e-8)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2/2
    #ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    #ssim_map_1 = (2 * mu1_mu2 + C1)/(mu1_sq + mu2_sq + C1)
    #ssim_map_2 = ((2 * torch.sqrt(sigma1_sq+1e-8) * torch.sqrt(sigma2_sq+1e-8)) + C2)/(sigma1_sq + sigma2_sq + C2)
    #ssim_map_3 = (sigma12 + C3)/((torch.sqrt(sigma1_sq+1e-8) * torch.sqrt(sigma2_sq+1e-8)) + C3)
    # mask each of the components iwth static_ch
    #ssim_map_1 = ssim_map_1 * static_ch
    #ssim_map_2 = ssim_map_2 * static_ch
    #ssim_map_3 = ssim_map_3 * static_ch
    #return #1-ssim_map_1.mean(), 1-ssim_map_2.mean(), 1-ssim_map_3.mean()
    ssim_map_1 = F.mse_loss(mu1 * static_ch, mu2 * static_ch, reduction='mean')
    ssim_map_2 = F.mse_loss(torch.sqrt(sigma1_sq) * static_ch, torch.sqrt(sigma2_sq) * static_ch, reduction='mean')
    # binarize img1 and img2
    #img1 = torch.where(img1 > 0, torch.ones_like(img1), torch.zeros_like(img1))
    #img2 = torch.where(img2 > 0, torch.ones_like(img2), torch.zeros_like(img2))
    ssim_map_3 = F.l1_loss(img1, img2, reduction='mean')
    return ssim_map_1, ssim_map_2, ssim_map_3


def ssim3D(img1, img2, window_size=5, size_average=True, static_ch = None):
    # Permute dimensions to have 'channels' at index 1
    img1 = img1.permute(0, 2, 1, 3, 4)
    img2 = img2.permute(0, 2, 1, 3, 4)

    # Normalize the images
    #img1 = (img1 - 0) / (255)

    #img2 = (img2 - 0) / (255)

    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim_3D(img1, img2, window, window_size, channel, size_average, static_ch)

def normalize_image(image, range_min, range_max):
    # Normalize the image pixel values to the range [0, 1]
    return (image - range_min) / (range_max - range_min)

def mse_of_spatial_std(input_tensor, target_tensor, kernel_size=3, epsilon=1e-8):
    # Check if both input and target tensor have the same dimensions
    assert input_tensor.shape == target_tensor.shape

    # Step 1: Compute the mean for each patch
    kernel = torch.ones(input_tensor.shape[1], 1, kernel_size, kernel_size, device=input_tensor.device) / (kernel_size ** 2)
    mean_input = F.conv2d(input_tensor, kernel, padding=kernel_size//2, groups=input_tensor.shape[1])
    mean_target = F.conv2d(target_tensor, kernel, padding=kernel_size//2, groups=target_tensor.shape[1])

    # Step 2: Compute the variance for each patch
    var_input = F.conv2d((input_tensor - mean_input) ** 2, kernel, padding=kernel_size//2, groups=input_tensor.shape[1])
    var_target = F.conv2d((target_tensor - mean_target) ** 2, kernel, padding=kernel_size//2, groups=target_tensor.shape[1])

    # Step 3: Compute the standard deviation for each patch
    std_input = torch.sqrt(var_input + epsilon)
    std_target = torch.sqrt(var_target + epsilon)

    # Step 4: Compute the MSE of standard deviations
    mse_loss = F.mse_loss(std_input, std_target)
    return mse_loss

def mse_of_spatial_cov(input_tensor, target_tensor, kernel_size=7, epsilon=1e-8):
    # Check if both input and target tensor have the same dimensions
    assert input_tensor.shape == target_tensor.shape

    # Step 1: Compute the mean for each patch
    kernel = torch.ones(input_tensor.shape[1], 1, kernel_size, kernel_size, device=input_tensor.device) / (kernel_size ** 2)
    mean_input = F.conv2d(input_tensor, kernel, padding=kernel_size//2, groups=input_tensor.shape[1])
    mean_target = F.conv2d(target_tensor, kernel, padding=kernel_size//2, groups=target_tensor.shape[1])

    # Step 2: Compute the covariance for each patch
    cov_input = F.conv2d((input_tensor - mean_input) * (input_tensor - mean_input), kernel, padding=kernel_size//2, groups=input_tensor.shape[1])
    cov_target = F.conv2d((target_tensor - mean_target) * (target_tensor - mean_target), kernel, padding=kernel_size//2, groups=target_tensor.shape[1])

    # Step 3: Compute the MSE of covariances
    mse_loss = F.mse_loss(cov_input, cov_target)

    return mse_loss
def row_standardization(matrix):
    # Calculate mean and standard deviation along axis 1 (rows)
    mean = torch.mean(matrix, dim=1, keepdim=True)
    std = torch.std(matrix, dim=1, keepdim=True)

    # Apply row-wise standardization
    standardized_matrix = (matrix - mean) / (std + 1e-7)  # Adding a small epsilon to avoid division by zero

    return standardized_matrix

def calculate_ratios(image_timeseries):
    # image_timeseries shape: (batch, 6, 1, 128, 128)

    # binarize image_timeseries
    #image_timeseries = torch.where(image_timeseries > 0, torch.ones_like(image_timeseries), torch.zeros_like(image_timeseries))
    # Step 1: Sum the pixel values for each timestep
    pixel_sums_per_timestep = torch.sum(image_timeseries, dim=(2, 3, 4))  # Shape: (batch, 6)

    # Step 2: Calculate the ratios
    #total_pixel_sum = torch.sum(pixel_sums_per_timestep, dim=1, keepdim=True)  # Shape: (batch, 1)

    # Add a small epsilon to avoid division by zero
    epsilon = 1e-8
    ratios = pixel_sums_per_timestep #/ (total_pixel_sum + epsilon)

    return ratios

def modified_total_variation_loss(predicted, true):
    """Compute the modified total variation loss.

    Parameters:
        predicted (torch.Tensor): 4D tensor representing the predicted images (B, C, H, W)
        true (torch.Tensor): 4D tensor representing the true images (B, C, H, W)

    Returns:
        torch.Tensor: scalar tensor representing the modified TV loss.
    """

    diff_h_predicted = predicted[:, :, 1:, :] - predicted[:, :, :-1, :]
    diff_w_predicted = predicted[:, :, :, 1:] - predicted[:, :, :, :-1]

    diff_h_true = true[:, :, 1:, :] - true[:, :, :-1, :]
    diff_w_true = true[:, :, :, 1:] - true[:, :, :, :-1]

    pred_diff = predicted[:, 1:] - true[:, :-1]
    true_diff = true[:, 1:] - true[:, :-1]


    mse_diff_h = F.mse_loss(diff_h_predicted, diff_h_true, reduction='mean')
    mse_diff_w = F.mse_loss(diff_w_predicted, diff_w_true, reduction='mean')
    mse_diff_t = F.mse_loss(pred_diff, true_diff, reduction='mean')

    modified_tv_loss = mse_diff_h + mse_diff_w + mse_diff_t

    return modified_tv_loss

def dilate_loss(outputs, targets, alpha, gamma, device):
	# outputs, targets: shape (batch_size, N_output, 1)
	batch_size, N_output = outputs.shape[0:2]
	loss_shape = 0
	softdtw_batch = soft_dtw.SoftDTWBatch.apply
	D = torch.zeros((batch_size, N_output,N_output )).to(device)
	for k in range(batch_size):
		Dk = soft_dtw.pairwise_distances(targets[k,:,:].view(-1,1),outputs[k,:,:].view(-1,1))
		D[k:k+1,:,:] = Dk
	loss_shape = softdtw_batch(D,gamma)

	path_dtw = path_soft_dtw.PathDTWBatch.apply
	path = path_dtw(D,gamma)
	Omega =  soft_dtw.pairwise_distances(torch.range(1,N_output).view(N_output,1)).to(device)
	loss_temporal =  torch.sum( path*Omega ) / (N_output*N_output)
	loss = alpha*loss_shape+ (1-alpha)*loss_temporal
	return loss, loss_shape, loss_temporal

# Modified function to return output of shape (500, 5, 1)
def sample_top_pixels_modified(true, pred, mask, pixels=50):
    """
    Samples the top 50 highest height, width pixels based on mask values from true and pred tensors.
    
    Parameters:
    - true (torch.Tensor): The ground truth tensor of shape (batch_size, timestep, 1, height, width).
    - pred (torch.Tensor): The prediction tensor of shape (batch_size, timestep, 1, height, width).
    - mask (torch.Tensor): The mask tensor of shape (batch_size, 1, 1, height, width).
    
    Returns:
    - sampled_true (torch.Tensor): Sampled ground truth tensor of shape (batch_size * 50, timestep, 1).
    - sampled_pred (torch.Tensor): Sampled prediction tensor of shape (batch_size * 50, timestep, 1).
    """
    
    # Step 1: Flatten the mask to (batch_size, height * width)
    mask_flattened = mask.view(mask.size(0), -1)
    
    # Step 2: Sort the flattened mask and get top 50 indices
    _, top_indices = torch.topk(mask_flattened, pixels, dim=-1)
    
    # Step 3: Gather the data corresponding to top 50 indices
    top_indices_expanded = top_indices.unsqueeze(1).unsqueeze(2).expand(-1, true.size(1), -1, -1)
    
    true_flattened = true.view(true.size(0), true.size(1), 1, true.size(3) * true.size(4))
    pred_flattened = pred.view(pred.size(0), pred.size(1), 1, pred.size(3) * pred.size(4))
    
    gathered_true = torch.gather(true_flattened, -1, top_indices_expanded)
    gathered_pred = torch.gather(pred_flattened, -1, top_indices_expanded)
    
    # Step 4: Reshape to (batch_size * 50, timestep, 1)
    sampled_true = gathered_true.view(-1, true.size(1), 1)
    sampled_pred = gathered_pred.view(-1, pred.size(1), 1)
    
    return sampled_true, sampled_pred


class DilateLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=0.001, device=None):
        super(DilateLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.device = device

    def forward(self, pred, true, static_ch):
        sampled_true, sampled_pred = sample_top_pixels_modified(true, pred, static_ch, pixels=50)
        return dilate_loss(sampled_pred, sampled_true, self.alpha, self.gamma, self.device)[0]

class DifferentialDivergenceLoss(nn.Module):
    def __init__(self, tau=1, epsilon=1e-8, w1=1, w2 =1, w3=1, w4=0, w5=1):
        super(DifferentialDivergenceLoss, self).__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.w1, self.w2, self.w3, self.w4, self.w5 = w1, w2, w3, w4, w5
        #self.main_loss = nn.L1Loss()
        self.main_loss = nn.MSELoss()
        #self.ssim = SSIM(window_size=4)

    def forward(self, pred, true, static_ch):
        # mae loss using functional
        #std_loss = self.ssim(pred, true)
        #mse_loss = dilate_loss(pred[:,:,0,64:65,64], true[:,:,0,64:65,64], alpha=0.1, gamma=0.001, device=pred.device)[0]
        #mse_loss = 1- ssim3D(pred, true, static_ch=static_ch)
        #sampled_true, sampled_pred = sample_top_pixels_modified(true, pred, static_ch, pixels=25)
       # mse_loss = dilate_loss(sampled_pred, sampled_true, alpha=0.1, gamma=0.001, device=pred.device)[0]
        # binatrize static_ch
        #static_ch = torch.where(static_ch > 0, torch.ones_like(static_ch), torch.zeros_like(static_ch))
        #mse_loss, reg_mse, reg_std = self.main_loss(pred[:,:,]*static_ch, true[:,:]*static_ch)
        mse_loss, reg_mse, reg_std = ssim3D(pred, true, static_ch=static_ch)
        # pred = pred * static_ch
        # true = true * static_ch
        #sum_loss = self.main_loss(pred*static_ch, true*static_ch)
        #mse_loss = 1 - ssim(pred[:,:,0], true[:,:,0], data_range=255, size_average=True, win_size=7) # return a scalar
        # sum_1 = torch.sum(pred, dim=(3,4))[:,:,0]
        # sum_2 = torch.sum(true, dim=(3,4))[:,:,0]
        # sum_1 = row_standardization(sum_1)
        # sum_2 = row_standardization(sum_2)
        # sum_loss = self.main_loss(sum_1, sum_2)
        true_ratios = calculate_ratios(true)
        predicted_ratios = calculate_ratios(pred)
        #sum_loss = self.main_loss(true_ratios,predicted_ratios)
        # set mean loss to the mean of temporal means
        sum_loss = torch.mean(pred, dim=(1)).mean()

        #pred_prob = F.softmax(sum_1, dim=1)
        #true_prob = F.softmax(sum_2, dim=1)

        # do kl div loss on pred_prob and true_prob using F.kl_div
        #sum_loss = F.kl_div(torch.log(pred_prob + self.epsilon), true_prob, reduction='batchmean')

        std_loss = F.mse_loss(torch.std(pred, dim=1), torch.std(true, dim=1))
        #sum_loss = std_loss
        #sum_loss = F.mse_loss(torch.sum(pred, dim=1), torch.sum(true,dim=1))
        #reg_std = modified_total_variation_loss(pred[:,:,0], true[:,:,0])#F.mse_loss(torch.std(pred, dim=1), torch.std(true, dim=1))
        #reg_std = modified_total_variation_loss(mu1[:,:,0]*static_ch[:,0],
         #                                       mu2[:,:,0]*static_ch[:,0])#F.mse_loss(torch.std(pred, dim=1), torch.std(true, dim=1))
        #std_loss = SSIM(pred[:,:,0], true[:,:,0])#F.mse_loss(torch.std(pred, dim=1), torch.std(true, dim=1))
        #sum_loss = mse_of_spatial_cov(pred[:,:,0], true[:,:,0])#F.mse_loss(torch.std(pred, dim=1), torch.std(true, dim=1))

        # pred_diff = pred[:, 1:] - pred[:, :-1]
        # true_diff = true[:, 1:] - true[:, :-1]
        #pred_diff = pred_diff.view(pred_diff.shape[0], pred_diff.shape[1], -1)
        #true_diff = true_diff.reshape(true_diff.shape[0], true_diff.shape[1], -1)
        #pred_prob = F.softmax(pred_diff / self.tau, dim=2)
        #true_prob = F.softmax(true_diff / self.tau, dim=2)
        # reg_mse = mse_loss#F.mse_loss(pred_diff, true_diff)
        #reg_std = F.mse_loss(torch.std(pred_diff, dim=1), torch.std(true_diff,dim=1))

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
