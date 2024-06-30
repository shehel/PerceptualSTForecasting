import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from timm.utils import AverageMeter

from openstl.models import SimVP_Model
from openstl.utils import reduce_tensor, DifferentialDivergenceLoss
from .simvp import SimVP
import pdb
from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt



class TAU(SimVP):
    r"""TAU

    Implementation of `Temporal Attention Unit: Towards Efficient Spatiotemporal 
    Predictive Learning <https://arxiv.org/abs/2206.12126>`_.

    """

    def __init__(self, args, device, steps_per_epoch):
        SimVP.__init__(self, args, device, steps_per_epoch)
        self.model = self._build_model(self.config)
        self.model_optim, self.scheduler, self.by_epoch = self._init_optimizer(steps_per_epoch)
        self.criterion1 = nn.MSELoss()
        self.criterion = DifferentialDivergenceLoss()
        self.adapt_weights = torch.tensor([1,0,0,0,0])
        self.component_1 = []
        self.component_2 = []
        self.component_3 = []
        self.component_4 = []
        self.component_5 = []
        self.iters_to_make_updates = 50
        self.iter = 0
    def _build_model(self, args):
        return SimVP_Model(**args).to(self.device)
    
    def diff_div_reg(self, pred_y, batch_y, tau=0.1, eps=1e-12):
        B, T, C = pred_y.shape[:3]
        if T <= 2:  return 0
        gap_pred_y = (pred_y[:, 1:] - pred_y[:, :-1]).reshape(B, T-1, -1)
        gap_batch_y = (batch_y[:, 1:] - batch_y[:, :-1]).reshape(B, T-1, -1)
        softmax_gap_p = F.softmax(gap_pred_y / tau, -1)
        softmax_gap_b = F.softmax(gap_batch_y / tau, -1)
        loss_gap = softmax_gap_p * \
            torch.log(softmax_gap_p / (softmax_gap_b + eps) + eps)
        return loss_gap.mean()

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

            with self.amp_autocast():
                pred_y, _ = self._predict(batch_x)

                _, total_loss, mse_loss,mse_div,std_div,reg_loss, sum_loss = self.criterion(pred_y[:,:,4:5,:,:]*batch_static, batch_y[:,:,4:5,:,:]*batch_static)
                #mse_div = std_div*0
                mse_loss = self.criterion1(pred_y[:,:,4:5,:,:]*batch_static, batch_y[:,:,4:5,:,:]*batch_static) + self.args.alpha * self.diff_div_reg(pred_y[:,:,4:5,:,:]*batch_static, batch_y[:,:,4:5,:,:]*batch_static)
                loss = self.adapt_weights[0] * mse_loss + self.adapt_weights[1] * mse_div + self.adapt_weights[2] * std_div + self.adapt_weights[3] * reg_loss + self.adapt_weights[4] * sum_loss

                # self.component_1.append(mse_loss.item())
                # self.component_2.append(mse_div.item())
                # self.component_3.append(std_div.item())
                # self.component_4.append(reg_loss.item())
                # self.component_5.append(sum_loss.item())


                # if self.iter % self.iters_to_make_updates == 0 and self.iter != 0:
                #     try:
                #         self.adapt_weights = self.adapt_object.get_component_weights(torch.tensor(self.component_1),torch.tensor(self.component_2),torch.tensor(self.component_3),torch.tensor(self.component_4),torch.tensor(self.component_5),verbose=False)
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
