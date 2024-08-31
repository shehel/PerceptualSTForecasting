import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from timm.utils import AverageMeter

from openstl.models import SimVP_Model, SimVPQ_Model, SimVPQCond_Model, SimVPQFiLM_Model, SimVPQFiLMC_Model, SimVPQCondC_Model
from openstl.utils import reduce_tensor, IntervalScores, FourQuantileRegressionLoss
from .base_method import Base_method

import pdb
class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion = IntervalScores()
        self.criterion2 = FourQuantileRegressionLoss(params={"q_lo_weight": 1, "q_hi_weight": 1, "m_weight": 1, "lo2_weight": 1, "hi2_weight": 1})

    def _build_model(self, args):

        model = SimVPQCondC_Model(**args)
        model = model.to(self.device)
        return model

    def _predict(self, batch_x, batch_y=None, **kwargs):
        """Forward the model"""
        if self.args.aft_seq_length == self.args.pre_seq_length:
            pred_y, translated = self.model(batch_x)
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
        return pred_y, translated

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
                #pred_y, _ = self._predict([batch_x, batch_quantiles[:,1:2]])
                #pred_y_lo, _ = self._predict([batch_x, batch_quantiles[:,0:1]])
                interval = (batch_quantiles[:,2]-batch_quantiles[:,0])
                pred_y, _ = self._predict([batch_x, interval])
                # interval1 = (batch_quantiles[:,0,:,:,:,:]-batch_quantiles[:,6])
                # interval2 = (batch_quantiles[:,1,:,:,:,:]-batch_quantiles[:,5])
                # interval3 = (batch_quantiles[:,2,:,:,:,:]-batch_quantiles[:,4])
                # pred_y1, _ = self._predict([batch_x, interval1])
                # pred_y2, _ = self._predict([batch_x, interval2])
                # pred_y3, _ = self._predict([batch_x, interval3])
                #pred_y_hi, _ = self._predict([batch_x, batch_quantiles[:,2:3]])
                # # create a new dimension at axis 1
                # pred_y_lo = pred_y_lo.unsqueeze(1)
                # pred_y_m = pred_y_m.unsqueeze(1)
                # pred_y_hi = pred_y_hi.unsqueeze(1)

                # # combine the 3 predictions at a new dimension at axis 1
                #pred_y = torch.cat((pred_y_lo, pred_y_m, pred_y_hi), dim=1)

                # clam pred_y to be between 0 and 255
                #pred_y = torch.clamp(pred_y, 0, 255)
                #pred_y = torch.cat((pred_y1[:, 0:1], pred_y2[:,0:1], pred_y3[:, 0:1],
                #pred_y1[:, 1:2], pred_y3[:, 2:3], pred_y2[:,2:3], pred_y1[:, 2:3]), dim=1)

                loss = self.criterion(pred_y[:,:,:,:,:,:], batch_y[:,:,:,:,:], batch_static[:,:,:], batch_quantiles[:,:,0,0,0,0])

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
                log_buffer = ' | pinball loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()
        return num_updates, pinball_m, eta
