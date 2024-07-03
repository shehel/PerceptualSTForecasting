import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from timm.utils import AverageMeter

from openstl.models import SimVP_Model, SimVPGAN_Model
from openstl.utils import reduce_tensor, IntervalScores
from .base_method import Base_method
from openstl.core.optim_scheduler import get_optim_scheduler

from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
import pdb
import math


class SimVPGAN(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        Base_method.__init__(self, args, device, steps_per_epoch)
        self.model, self.d_model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch, self.dmodel_optim, self.d_scheduler, self.d_epoch = self._init_optimizer(steps_per_epoch)
        #self.criterion = nn.MSELoss()
        self.criterion = IntervalScores()
        self.BCE_loss = nn.BCEWithLogitsLoss()
        # set 1 to be a torch.tensor and move it to gpu
        self.real_label = torch.tensor(1.).to(self.device)
        self.fake_label = torch.tensor(0.).to(self.device)
        self.adapt_object = LossWeightedSoftAdapt(beta=-0.2)
        self.iters_to_make_updates = 70
        self.adapt_weights = torch.tensor([100,0,0,100,0])
        n_steps = 100
        y_50 = 0.01
        decay_constant = -math.log(y_50) / 50

        time_steps = torch.arange(0, n_steps, dtype=torch.float32)
        self.mse_adapt = torch.exp(-decay_constant * time_steps)

        self.component_1 = []
        self.component_2 = []
        self.component_3 = []
        self.component_4 = []
        self.component_5 = []
        self.iter = 0
        
        self.clip_value = 0.01
        # create a list of numbers linearly exponentially decreasing from 1 to 0 in 100 steps
        #fun = pysdtw.distance.pairwise_l2_squared

# create the SoftDTW distance function
        #self.criterion = pysdtw.SoftDTW(gamma=1.0, dist_func=fun, use_cuda=True)
        #self.criterion_cpu =  pysdtw.SoftDTW(gamma=1.0, dist_func=fun, use_cuda=False)
    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def _init_optimizer(self, steps_per_epoch):
        opt_gen, sched_gen, epoch_gen = get_optim_scheduler(
            self.args, self.args.epoch, self.model, steps_per_epoch)
        self.args.lr = 1e-7
        opt_dis, sched_dis, epoch_dis = get_optim_scheduler(self.args, self.args.epoch, self.d_model, steps_per_epoch)
        return opt_gen, sched_gen, epoch_gen, opt_dis, sched_dis, epoch_dis
    def _build_model(self, args):
        gen_model = SimVP_Model(**args)
        gen_model.load_state_dict(torch.load("work_dirs/e1_q28_m16_simconvsc/checkpoints/latest.pth")['state_dict'], strict=False)
        gen_model.to(self.device),
        disc = SimVPGAN_Model().to(self.device)
        return gen_model, disc

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
            
            if not self.args.use_prefetcher:
                batch_x, batch_y, batch_static = batch_x.to(self.device), batch_y.to(self.device), batch_static.to(self.device)
            runner.call_hook('before_train_iter')

            b_size = batch_y.size(0)
            with self.amp_autocast():
                for _ in range(5):
                    self.d_model.zero_grad()
                    real_output = self.d_model(batch_y).view(-1)
                    #label = self.get_target_tensor(output, True)   #torch.full((b_size,), self.real_label, dtype=torch.float, device=self.device)
                    #errD_real = self.BCE_loss(output, label)
                    errD_real = -torch.mean(real_output)
                    errD_real.backward()
                    # clam pred_y to be between 0 and 255
                    #pred_y = torch.clamp(pred_y, 0, 255)
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
                    pred_y, _ = self._predict(batch_x)
                    
                    fake_output = self.d_model(pred_y.detach()).view(-1)
                    #label = self.get_target_tensor(output, False)
                    #label.fill_(self.fake_label)
                    #errD_fake = self.BCE_loss(output, label)
                    errD_fake = torch.mean(fake_output)
                    errD_fake.backward()

                    errD = errD_real + errD_fake
                    # Update the discriminator
                    self.dmodel_optim.step()
                    #_, total_loss, mse_loss,mse_div,std_div,reg_loss, sum_loss = self.criterion(pred_y[:,:,4:5,:,:], batch_y[:,:,4:5,:,:], batch_static)
                    for p in self.d_model.parameters():
                        p.data.clamp_(-self.clip_value, self.clip_value)
 
                self.model.zero_grad()

                gen_output = self.d_model(pred_y).view(-1)
                #gen_label = self.get_target_tensor(gen_output, True)
                #gen_loss = self.BCE_loss(gen_output, gen_label)
                gen_loss = -torch.mean(gen_output)

                # Calculate the reconstruction loss
                _, total_loss, mse_loss, reg_mse, reg_std, std_loss, sum_loss = self.criterion(pred_y[:,:,4:5,:,:], batch_y[:,:,4:5,:,:], batch_static)
                recon_loss = sum(self.adapt_weights[i] * loss for i, loss in enumerate([mse_loss, reg_mse, reg_std, std_loss, sum_loss]))

                # Combine losses for the generator update
                loss = gen_loss + recon_loss
                #loss.backward()
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


            #     self.component_1.append(mse_loss.item())
            #     self.component_2.append(mse_div.item())
            #     self.component_3.append(std_div.item())
            #     self.component_4.append(reg_loss.item())
            #     self.component_5.append(sum_loss.item())
            # if self.iter % self.iters_to_make_updates == 0 and self.iter != 0:
            #         try:
            #             #self.adapt_weights = self.adapt_object.get_component_weights(torch.tensor(self.component_1[-71:]),torch.tensor(self.component_2[-71:]),torch.tensor(self.component_3[-71:]),torch.tensor(self.component_4[-71:]),torch.tensor(self.component_5[-71:]),verbose=True)
            #             self.adapt_weights = self.adapt_object.get_component_weights(torch.tensor(self.component_1),torch.tensor(self.component_2),torch.tensor(self.component_3),torch.tensor(self.component_4),torch.tensor(self.component_5),verbose=True)
            #             # print elements in self.adapt weights after rounding it to nearest 2 decimal places
            #             print ("adapt weights: ", torch.round(self.adapt_weights*100)/100)

            #         except:
            #             print ("FAILURE in softadapt")
            #             pdb.set_trace()
            #         self.component_1 = []
            #         self.component_2 = []
            #         self.component_3 = []
            #         self.component_4 = []
            #         self.component_5 = []
            #         self.component_1.append(mse_loss.item())
            #         self.component_2.append(mse_div.item())
            #         self.component_3.append(std_div.item())
            #         self.component_4.append(reg_loss.item())
            #         self.component_5.append(sum_loss.item())
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
                losses_total.update(errD_real.item(), batch_x.size(0))
                losses_mse_m.update(mse_loss.item(), batch_x.size(0))
                losses_reg_m.update(reg_mse.item(), batch_x.size(0))
                losses_reg_s.update(reg_std.item(), batch_x.size(0))
                losses_std.update(std_loss.item(), batch_x.size(0))
                losses_sum.update(errD_fake.item(), batch_x.size(0))


            if self.dist:
                losses_m.update(reduce_tensor(loss), batch_x.size(0))

            if not self.by_epoch:
                self.d_scheduler.step()
                self.scheduler.step()
            runner.call_hook('after_train_iter')
            runner._iter += 1

            if self.rank == 0:
                log_buffer = 'train loss: {:.4f}'.format(loss.item())
                log_buffer += ' | train mse loss: {:.4f}'.format(mse_loss.item())
                log_buffer += ' | train reg loss: {:.4f}'.format(std_loss.item())
                log_buffer += ' | data time: {:.4f}'.format(data_time_m.avg)
                train_pbar.set_description(log_buffer)

            end = time.time()  # end for

        if hasattr(self.model_optim, 'sync_lookahead'):
            self.model_optim.sync_lookahead()
        return num_updates, losses_m, losses_total, losses_mse_m,losses_reg_m,losses_reg_s,losses_std, losses_sum, eta