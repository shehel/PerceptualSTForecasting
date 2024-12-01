import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from timm.utils import AverageMeter

from openstl.models import SimVP_Model, UNet_Model, UNetQ_Model
from openstl.utils import reduce_tensor, IntervalScores
from .base_method import Base_method
from openstl.core.optim_scheduler import get_optim_scheduler

import pdb
import math


class UNet(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch= self._init_optimizer(steps_per_epoch)
        self.criterion = IntervalScores(quantile_weights=[1,1,1])
        self.loss_type = 'mis'
        # set 1 to be a torch.tensor and move it to gpu


    def _init_optimizer(self, steps_per_epoch):
        opt_gen, sched_gen, epoch_gen = get_optim_scheduler(
            self.args, self.args.epoch, self.model, steps_per_epoch)
        return opt_gen, sched_gen, epoch_gen
    def _build_model(self, args):
        resid_model = UNetQ_Model(**args).to(self.device)
        return resid_model

    def _predict(self, batch_x, batch_y=None, test=False, **kwargs):
        """Forward the model"""
        if self.args.aft_seq_length == self.args.pre_seq_length:

            pred_y, _ = self.model(batch_x)
        elif self.args.aft_seq_length < self.args.pre_seq_length:
            pred_y, translated = self.model(batch_x)
            pred_y = pred_y[:, :self.args.aft_seq_length]
        elif self.args.aft_seq_length > self.args.pre_seq_length:
            pred_y = []
            d = self.args.aft_seq_length // self.args.pre_seq_length
            m = self.args.aft_seq_length % self.args.pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y, pred_y

    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()

        pinball_m = AverageMeter()

        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y, batch_static, batch_quantiles in train_pbar:

            data_time_m.update(time.time() - end)


            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y, batch_static, batch_quantiles = batch_x.to(self.device), batch_y.to(self.device), batch_static.to(self.device), batch_quantiles.to(self.device)
            runner.call_hook('before_train_iter')

            with self.amp_autocast():

                pred_y, _ = self._predict([batch_x, batch_quantiles])
                # prepend batch_y[:,0,:,:,:] to pred_y along dimension 1


                loss,_ = self.criterion(pred_y[:,:,:,:,:,:], batch_y[:,:,:,:,:], batch_static[:,:,:], batch_quantiles[:,:], loss_type=self.loss_type)


            if epoch >= 0:
                if self.loss_scaler is not None:
                    if torch.any(torch.isnan(loss)) or torch.any(torch.isinf(loss)):
                        raise ValueError("Inf or nan loss value. Please use fp32 training!")
                    self.loss_scaler(
                        loss, self.model_optim,
                        clip_grad=self.args.clip_grad, clip_mode=self.args.clip_mode,
                        parameters=self.model.parameters())
                else:
                    loss.backward()
                    self.clip_grads(self.model.parameters())
                    self.model_optim.step()

            torch.cuda.synchronize()
            num_updates += 1

            #loss, total_loss, mse_loss,mse_div,std_div,reg_loss = self.criterion(pred_y[:,:,2:3,:,:], batch_y[:,:,4:5,:,:])
            if not self.dist:
                pinball_m.update(loss.item(), batch_x.size(0))

            if self.dist:
                pinball_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'pinball loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()
        return num_updates, pinball_m, eta
