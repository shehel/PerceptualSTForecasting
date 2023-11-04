import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from timm.utils import AverageMeter

from openstl.models import UNet_Model
from openstl.utils import reduce_tensor, DifferentialDivergenceLoss, DilateLoss
from .base_method import Base_method
import pdb

from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
class UNet(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        #self.criterion = nn.MSELoss()
        self.loss_wgt = torch.tensor(1)
        self.criterion = DifferentialDivergenceLoss()
        self.val_criterion = DilateLoss()
        self.adapt_object = LossWeightedSoftAdapt(beta=1.5)
        self.iters_to_make_updates = 100
        self.adapt_weights = torch.tensor([1,0,0,0,0])
        self.component_1 = []
        self.component_2 = []
        self.component_3 = []
        self.component_4 = []
        self.component_5 = []
        self.iter = 0

    def _build_model(self, config):
        return UNet_Model(**config).to(self.device)

    def _predict(self, batch_x, batch_y=None, **kwargs):
        """Forward the model"""

        # move axis from source=4 to destination=2 for the batch_x torch tensor
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
        losses_m = AverageMeter()
        losses_mse_m = AverageMeter()
        losses_reg_m = AverageMeter()
        losses_reg_s = AverageMeter()
        losses_std = AverageMeter()
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
                batch_x.requires_grad = True
            runner.call_hook('before_train_iter')

            with self.amp_autocast():
                pred_y, translated = self._predict(batch_x)
                #encoded = self.model.encode(batch_y)
                #recon = self.model.recon(batch_x)
                #mse_loss = self.criterion(pred_y[:,:,2:3,[52,83,63,42],[76,104,14,63]], batch_y[:,:,4:5,[52,83,63,42],[76,104,14,63]])
                #loss, total_loss, mse_loss,mse_div,std_div,reg_loss, sum_loss = self.criterion(pred_y[:,:,2:3,:,:]*batch_static, batch_y[:,:,4:5,:,:]*batch_static)
                _, total_loss, mse_loss,reg_mse,reg_std,std_loss, sum_loss = self.criterion(pred_y[:,:,0:1,:,:], batch_y[:,:,4:5,:,:], batch_static)
                #latent_loss = F.mse_loss(encoded, translated)
                #mse_loss = latent_loss
                loss = F.mse_loss(pred_y[:,:,0:1,64,64], batch_y[:,:,4:5,64,64])
                #self.adapt_weights[2] * std_div + (1-self.adapt_weights[2]) * mse_div

                #loss = self.adapt_weights[0] * mse_loss + self.adapt_weights[1] * mse_div + self.adapt_weights[2] * std_div + self.adapt_weights[3] * reg_loss + self.adapt_weights[4] * sum_loss
                #loss, mse_loss,mse_div,std_div,reg_loss = self.criterion(pred_y[:,:,2:3,[52,83,63,42],[76,104,14,63]],
                #                         j                                batch_y[:,:,4:5,[52,83,63,42],[76,104,14,63]])
                #encoded_norms = torch.mean(torch.norm(encoded.reshape(encoded.shape[0],-1), dim=(1)))
                #recon_loss = F.mse_loss(recon, batch_x[:,:,0::2])

                #loss = latent_loss + loss
                # take loss of pred_y[:,:,1:3,:,:] and batch_y[:,:,4:5,:,:] but multiply both by batch_static[np.newaxis, np.newaxis, :, :, :]
                #mse_loss = self.criterion(pred_y[:,:,2:3,:,:]*batch_static, batch_y[:,:,4:5,:,:]*batch_static)
                #mse_loss = self.criterion(pred_y[:,:,2:3,:,:]*batch_static, batch_y[:,:,4:5,:,:]*batch_static)
                # get tandard deviation of pred_y[:,:,2:3,:,:] and batch_y[:,:,4:5,:,:] over the first axis and get their MSE
                #reg_loss = self.criterion(torch.std(pred_y[:,:,2:3,:,:], dim=1), torch.std(batch_y[:,:,4:5,:,:], dim=1))
                #reg_loss = self.criterion(torch.std(pred_y[:,:,2:3,52,76], dim=1), torch.std(batch_y[:,:,4:5,52,76], dim=1))
                #mse_loss = self.criterion(pred_y[:,:,2:3,:,:], batch_y[:,:,4:5,:,:])
                #mse_loss = self.criterion(pred_y, batch_y[:,:,0::2])
                #reg_loss = torch.tensor(0)#self.criterion(translated, encoded)
                # gradients = torch.autograd.grad(pred_y, batch_x, grad_outputs=torch.ones_like(pred_y), create_graph=True, retain_graph=True)
                #loss, mse_loss, reg_loss = self.criterion(gradients, pred_y- batch_x)]
                
                # reg_loss = self.criterion(gradients[0][:,:,0::2], (pred_y-batch_x[:,:,0::2]))
                #loss = self.loss_wgt*(mse_loss) + (self.loss_wgt)*reg_loss

                # self.component_1.append(mse_loss.item())
                # self.component_2.append(mse_div.item())
                # self.component_3.append(std_div.item())
                # self.component_4.append(reg_loss.item())
                # self.component_5.append(sum_loss.item())
                # if self.iter % self.iters_to_make_updates == 0 and self.iter != 0:
                #     try:
                #         self.adapt_weights = self.adapt_object.get_component_weights(torch.tensor(self.component_1[-70:]),torch.tensor(self.component_2[-70:]),torch.tensor(self.component_3[-70:]),torch.tensor(self.component_4[-70:]),torch.tensor(self.component_5[-70:]),verbose=False)
                #     except:
                #         print ("FAILURE in softadapt")
                #         pdb.set_trace()
                #     self.component_1 = []
                #     self.component_2 = []
                #     self.component_3 = []
                #     self.component_4 = []
                #     self.component_5 = []
                #     self.component_1.append(mse_loss.item())
                #     self.component_2.append(mse_div.item())
                #     self.component_3.append(std_div.item())
                #     self.component_4.append(reg_loss.item())
                #     self.component_5.append(sum_loss.item())

                # self.iter += 1

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
                losses_m.update(loss.item(), batch_x.size(0))
                losses_total.update(total_loss.item(), batch_x.size(0))
                losses_mse_m.update(mse_loss.item(), batch_x.size(0))
                losses_reg_m.update(reg_mse.item(), batch_x.size(0))
                losses_reg_s.update(reg_std.item(), batch_x.size(0))
                losses_std.update(std_loss.item(), batch_x.size(0))
                losses_sum.update(sum_loss.item(), batch_x.size(0))

            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | train mse loss: {:.4f}'.format(mse_loss.item())
                log_buffer += ' | train reg loss: {:.4f}'.format(reg_mse.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()
        return num_updates, losses_m, losses_total, losses_mse_m,losses_reg_m,losses_reg_s,losses_std, losses_sum, eta
