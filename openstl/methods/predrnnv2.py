import time
import torch
import torch.nn as nn
from timm.utils import AverageMeter
from tqdm import tqdm

from openstl.models import PredRNNv2_Model
from openstl.utils import (reduce_tensor, reshape_patch, reshape_patch_back,
                           reserve_schedule_sampling_exp, schedule_sampling, IntervalScores)
from .predrnn import PredRNN
import pdb

class PredRNNv2(PredRNN):
    r"""PredRNNv2

    Implementation of `PredRNN: A Recurrent Neural Network for Spatiotemporal
    Predictive Learning <https://arxiv.org/abs/2103.09504v4>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        PredRNN.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.args)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        #self.criterion = nn.MSELoss()
        self.criterion = DifferentialDivergenceLoss()
        self.adapt_weights = torch.tensor([1,0,0,0,0])

    def _build_model(self, args):
        num_hidden = [int(x) for x in self.args.num_hidden.split(',')]
        num_layers = len(num_hidden)
        return PredRNNv2_Model(num_layers, num_hidden, args).to(self.device)


    def train_one_epoch(self, runner, train_loader, epoch, num_updates, eta=None, **kwargs):
        """Train the model with train_loader."""
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        losses_mse_m = AverageMeter()
        losses_reg_m = AverageMeter()
        losses_div_m = AverageMeter()
        losses_div_s = AverageMeter()
        losses_total = AverageMeter()
        losses_sum = AverageMeter()
        self.model.train()
        if self.by_epoch:
            self.scheduler.step(epoch)
        train_pbar = tqdm(train_loader) if self.rank == 0 else train_loader

        end = time.time()
        for batch_x, batch_y, batch_static in train_pbar:
            data_time_m.update(time.time() - end)
            self.model_optim.zero_grad()

            if not self.args.use_prefetcher:
                batch_x, batch_y, batch_static = batch_x.to(self.device), batch_y.to(self.device), batch_static.to(self.device)
            runner.call_hook('before_train_iter')

            # preprocess
            ims_cat = torch.cat([batch_x, batch_y], dim=1)
            ims = ims_cat.permute(0, 1, 3, 4, 2).contiguous()
            ims = reshape_patch(ims, self.args.patch_size)
            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(
                    num_updates, ims.shape[0], self.args)
            else:
                eta, real_input_flag = schedule_sampling(
                    eta, num_updates, ims.shape[0], self.args)

            with self.amp_autocast():
                img_gen, loss = self.model(ims, real_input_flag)
                img_gen = reshape_patch_back(img_gen, self.args.patch_size)
                img_gen = img_gen.permute(0,1,4,2,3)
                _, total_loss, mse_loss,mse_div,std_div,reg_loss, sum_loss = self.criterion(img_gen[:,:,4:5,:,:]*batch_static, ims_cat[:,1:,4:5,:,:]*batch_static)

                #loss = self.adapt_weights[0] * mse_loss + self.adapt_weights[1] * mse_div + self.adapt_weights[2] * std_div + self.adapt_weights[3] * reg_loss + self.adapt_weights[4] * sum_loss
            if not self.dist:
                losses_m.update(loss.item(), batch_x.size(0))
                losses_mse_m.update(mse_loss.item(), batch_x.size(0))
                losses_reg_m.update(reg_loss.item(), batch_x.size(0))
                losses_div_m.update(mse_div.item(), batch_x.size(0))
                losses_div_s.update(std_div.item(), batch_x.size(0))
                losses_total.update(total_loss.item(), batch_x.size(0))
                losses_sum.update(sum_loss.item(), batch_x.size(0))

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

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()

        return num_updates, losses_m, losses_mse_m,losses_reg_m,losses_div_m,losses_div_s, losses_total, losses_sum, eta
