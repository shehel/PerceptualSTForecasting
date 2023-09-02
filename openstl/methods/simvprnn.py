import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from timm.utils import AverageMeter

from openstl.models import SimVPRnn_Model
from openstl.utils import (reduce_tensor, DifferentialDivergenceLoss,
                           reserve_schedule_sampling_exp, schedule_sampling)
from .base_method import Base_method

from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
import pdb
class SimVPRnn(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        #self.criterion = nn.MSELoss()
        self.criterion = DifferentialDivergenceLoss()
        self.adapt_object = LossWeightedSoftAdapt(beta=-0.3)
        self.iters_to_make_updates = 50
        self.adapt_weights = torch.tensor([1,0,0,0,0])
        self.component_1 = []
        self.component_2 = []
        self.component_3 = []
        self.component_4 = []
        self.component_5 = []
        self.iter = 0
        #fun = pysdtw.distance.pairwise_l2_squared

# create the SoftDTW distance function
        #self.criterion = pysdtw.SoftDTW(gamma=1.0, dist_func=fun, use_cuda=True)
        #self.criterion_cpu =  pysdtw.SoftDTW(gamma=1.0, dist_func=fun, use_cuda=False)

    def _build_model(self, args):
        return SimVPRnn_Model(**args).to(self.device)

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model"""
        if self.args.reverse_scheduled_sampling == 1:
            mask_input = 1
        else:
            mask_input = self.args.pre_seq_length
        
        #TODO
        _, img_channel, img_height, img_width = 1, 32,32,32

        # preprocess
        test_dat = torch.cat([batch_x, batch_y], dim=1)

        real_input_flag = torch.zeros(
            (batch_x.shape[0],
            self.args.total_length - mask_input - 1,
            img_channel,
            img_height,
            img_width,
            )).to(self.device)
            
        if self.args.reverse_scheduled_sampling == 1:
            real_input_flag[:, :self.args.pre_seq_length - 1, :, :] = 1.0
        
        pred_y,_ = self.model(test_dat, real_input_flag)
        #pred_y = img_gen[:, -self.args.aft_seq_length:]

        return pred_y, None

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

            
            # TODO fix ims shape to sampling function
            ims = torch.cat([batch_x, batch_y], dim=1)
            if self.args.reverse_scheduled_sampling == 1:
                real_input_flag = reserve_schedule_sampling_exp(
                    num_updates, ims.shape[0], self.args)
            else:
                eta, real_input_flag = schedule_sampling(
                    eta, num_updates, ims.shape[0], self.args)
            with self.amp_autocast():
                pred_y, _ = self.model(ims, real_input_flag)
                #encoded = self.model.encode(batch_y)
                #recon = self.model.recon(batch_x)
                #reg_loss = self.criterion(torch.std(pred_y[:,:,2:3,:,:], dim=1)*batch_static[:,0], torch.std(batch_y[:,:,4:5,:,:], dim=1)*batch_static[:,0])
                #reg_loss = torch.tensor(0)#self.criterion(translated, encoded)
                #mse_loss = self.criterion(pred_y[:,:,2:3,:,:]*batch_static, batch_y[:,:,4:5,:,:]*batch_static)
                #mse_loss = self.criterion(pred_y[:,:,2:3,[52,83,63,42],[76,104,14,63]], batch_y[:,:,4:5,[52,83,63,42],[76,104,14,63]])
                #mse_loss = F.mse_loss(pred_y[:,:,2:3,52,76], batch_y[:,:,4:5,52,76])
                #loss = self.loss_wgt*(mse_loss) + (self.loss_wgt)*reg_loss
                #recon_loss = loss
                #encoded_norms = loss
                
                _, total_loss, mse_loss,mse_div,std_div,reg_loss, sum_loss = self.criterion(pred_y[:,:,4:5,:,:]*batch_static, batch_y[:,:,4:5,:,:]*batch_static)

                loss = self.adapt_weights[0] * mse_loss + self.adapt_weights[1] * mse_div + self.adapt_weights[2] * std_div + self.adapt_weights[3] * reg_loss + self.adapt_weights[4] * sum_loss
                #encoded_norms = torch.mean(torch.norm(encoded.reshape(encoded.shape[0],-1), dim=(1)))
                #recon_loss = F.mse_loss(recon[:,:,0::2], batch_y[:,:,0::2])
                #latent_loss = F.mse_loss(encoded, translated)

                 
                
                #loss = latent_loss + recon_loss + encoded_norms

                #loss, mse_loss,mse_div,std_div,reg_loss = self.criterion(pred_y[:,:,2:3,[52,83,63,42],[76,104,14,63]],
                #                                                         batch_y[:,:,4:5,[52,83,63,42],[76,104,14,63]])
                #mse_loss = self.criterion(pred_y[:,:,2:3,:,:], batch_y[:,:,4:5,:,:])
                #mse_loss = mse_loss.mean()
                #mse_loss = self.criterion(pred_y, batch_y[:,:,0::2])
                #reg_loss = self.criterion(translated, encoded)

                # self.component_1.append(mse_loss.item())
                # self.component_2.append(mse_div.item())
                # self.component_3.append(std_div.item())
                # self.component_4.append(reg_loss.item())
                # self.component_5.append(sum_loss.item())


                # if self.iter % self.iters_to_make_updates == 0 and self.iter != 0:
                #     try:
                #         self.adapt_weights = self.adapt_object.get_component_weights(torch.tensor(self.component_1),torch.tensor(self.component_2),torch.tensor(self.component_3),torch.tensor(self.component_4),torch.tensor(self.component_5),verbose=True)
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
                losses_mse_m.update(mse_loss.item(), batch_x.size(0))
                losses_reg_m.update(reg_loss.item(), batch_x.size(0))
                losses_div_m.update(mse_div.item(), batch_x.size(0))
                losses_div_s.update(std_div.item(), batch_x.size(0))
                losses_total.update(total_loss.item(), batch_x.size(0))
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
                log_buffer += ' | train reg loss: {:.4f}'.format(reg_loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()
        return num_updates, losses_m, losses_mse_m,losses_reg_m,losses_div_m,losses_div_s, losses_total, losses_sum, eta
