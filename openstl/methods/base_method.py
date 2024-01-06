from typing import Dict, List, Union
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from contextlib import suppress
from timm.utils import NativeScaler
from timm.utils.agc import adaptive_clip_grad

from openstl.core import metric
from openstl.core.optim_scheduler import get_optim_scheduler
from openstl.utils import gather_tensors_batch, get_dist_info, ProgressBar

has_native_amp = False
import pdb
try:

    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

from torch.autograd import grad
class Base_method(object):
    """Base Method.

    This class defines the basic functions of a video prediction (VP)
    method training and testing. Any VP method that inherits this class
    should at least define its own `train_one_epoch`, `vali_one_epoch`,
    and `test_one_epoch` function.

    """

    def __init__(self, args, device, steps_per_epoch):
        super(Base_method, self).__init__()
        self.args = args
        self.dist = args.dist
        self.device = device
        self.config = args.__dict__
        self.criterion = None
        self.model_optim = None
        self.scheduler = None
        if self.dist:
            self.rank, self.world_size = get_dist_info()
            assert self.rank == int(device.split(':')[-1])
        else:
            self.rank, self.world_size = 0, 1
        self.clip_value = self.args.clip_grad
        self.clip_mode = self.args.clip_mode if self.clip_value is not None else None
        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self.amp_autocast = suppress  # do nothing
        self.loss_scaler = None
        # setup metrics
        if 'weather' in self.args.dataname:
            self.metric_list, self.spatial_norm = ['mse', 'rmse', 'mae'], True
        else:
            self.metric_list, self.spatial_norm = ['mse', 'mae'], False

    def _build_model(self, **kwargs):
        raise NotImplementedError

    def _init_optimizer(self, steps_per_epoch):
        return get_optim_scheduler(
            self.args, self.args.epoch, self.model, steps_per_epoch)

    def _init_distributed(self):
        """Initialize DDP training"""
        if self.args.fp16 and has_native_amp:
            self.amp_autocast = torch.cuda.amp.autocast
            self.loss_scaler = NativeScaler()
            if self.rank == 0:
               print('Using native PyTorch AMP. Training in mixed precision (fp16).')
        else:
            print('AMP not enabled. Training in float32.')
        self.model = NativeDDP(self.model, device_ids=[self.rank],
                               broadcast_buffers=self.args.broadcast_buffers,
                               find_unused_parameters=self.args.find_unused_parameters)

    def train_one_epoch(self, runner, train_loader, **kwargs): 
        """Train the model with train_loader.

        Args:
            runner: the trainer of methods.
            train_loader: dataloader of train.
        """
        raise NotImplementedError

    def _predict(self, batch_x, batch_y, **kwargs):
        """Forward the model.

        Args:
            batch_x, batch_y: testing samples and groung truth.
        """
        raise NotImplementedError

    def _dist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios in a distributed manner.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        length = len(data_loader.dataset) if length is None else length
        if self.rank == 0:
            prog_bar = ProgressBar(len(data_loader))

        # loop
        for idx, (batch_x, batch_y) in enumerate(data_loader):
            if idx == 0:
                part_size = batch_x.shape[0]
            with torch.no_grad():
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y,_ = self._predict(batch_x, batch_y)

            if gather_data:  # return raw datas
                results.append(dict(zip(['inputs', 'preds', 'trues'],
                                        [batch_x.cpu().numpy(), pred_y.cpu().numpy(), batch_y.cpu().numpy()])))
            else:  # return metrics
                eval_res, _ = metric(pred_y.cpu().numpy(), batch_y.cpu().numpy(),
                                     data_loader.dataset.mean, data_loader.dataset.std,
                                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)
                eval_res['loss'] = self.criterion(pred_y, batch_y).cpu().numpy()
                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].reshape(1)
                results.append(eval_res)

            if self.args.empty_cache:
                torch.cuda.empty_cache()
            if self.rank == 0:
                prog_bar.update()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_cat = np.concatenate([batch[k] for batch in results], axis=0)
            # gether tensors by GPU (it's no need to empty cache)
            results_gathered = gather_tensors_batch(results_cat, part_size=min(part_size*8, 16))
            results_strip = np.concatenate(results_gathered, axis=0)[:length]
            results_all[k] = results_strip
        return results_all

    def _nondist_forward_grad(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        results = []
        prog_bar = ProgressBar(len(data_loader))
        length = len(data_loader.dataset) if length is None else length
        for i, (batch_x, batch_y, batch_static) in enumerate(data_loader):
            batch_x, batch_y, batch_static = batch_x.to(self.device), batch_y.to(self.device), batch_static.to(self.device)
            pred_y,_ = self._predict(batch_x, batch_y)
            # make it so that pred_y requires grad
            self.model_optim.zero_grad()

            _, total_loss, mse_loss,mse_div,std_div,reg_loss, sum_loss = self.criterion(pred_y[:,:,4:5,:,:]*batch_static, batch_y[:,:,4:5,:,:]*batch_static)

            loss = self.adapt_weights[0] * mse_loss + self.adapt_weights[1] * mse_div + self.adapt_weights[2] * std_div + self.adapt_weights[3] * reg_loss + self.adapt_weights[4] * sum_loss
            output_gradients = grad(loss, pred_y, retain_graph=True)[0]
            if gather_data:  # return raw datas
                results.append(dict(zip(['inputs', 'preds', 'trues'],
                                        [batch_x[:,:,4:5,:,:].cpu().numpy(),
                                        output_gradients[:,:,4:5,:,:].cpu().numpy()*batch_static.cpu().numpy(),
                                     batch_y[:,:,4:5,:,:].cpu().numpy()*batch_static.cpu().numpy()])))
            else:  # return metrics
                #eval_res, _ = metric(pred_y.cpu().numpy()*batch_static.numpy(), batch_y.cpu().numpy()*batch_static.numpy(),
                #                     data_loader.dataset.mean, data_loader.dataset.std,
                #                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)
                eval_res = {}
                eval_res['train_loss'],eval_res['total_loss'],eval_res['mse'],eval_res['div'],eval_res['div_std'],eval_res['std'], eval_res['sum'] = self.criterion(pred_y, batch_y)
                for k in eval_res.keys():
                    eval_res[k] = eval_res[k].cpu().numpy().reshape(1)
                results.append(eval_res)

            prog_bar.update()
            if self.args.empty_cache:
                torch.cuda.empty_cache()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in results], axis=0)
        preds = torch.tensor(results_all['preds'])
        #results['trues'] = results['trues'][:,0:1,4:5,70,65]
        trues = torch.tensor(results_all['trues'])
        #losses_m = self.criterion_cpu(preds, trues)
        losses_m= self.criterion(preds, trues)
        results_all["loss"] = losses_m
        return results_all

    def _nondist_forward_collect(self, data_loader, length=None, gather_data=False):
        """Forward and collect predictios.

        Args:
            data_loader: dataloader of evaluation.
            length (int): Expected length of output arrays.
            gather_data (bool): Whether to gather raw predictions and inputs.

        Returns:
            results_all (dict(np.ndarray)): The concatenated outputs.
        """
        # preparation
        perm = [[0,1,2,3,4,5,6,7],
        [2,3,4,5,6,7,0,1],
        [4,5,6,7,0,1,2,3],
        [6,7,0,1,2,3,4,5]
        ]
        results = []
        prog_bar = ProgressBar(len(data_loader))
        length = len(data_loader.dataset) if length is None else length
        eval_res = []
        
        if data_loader.dataset.perm:
            data_loader.dataset.perm = False
            for i, (batch_x, batch_y, batch_static) in enumerate(data_loader):
                    #pred_y = torch.zeros_like(batch_y)
                    
                    for x in range(2,3):
                        with torch.no_grad():
                            
                            batch_x, batch_y= batch_x.to(self.device), batch_y.to(self.device)
                            batch_x_p = batch_x[:,:,perm[x],:,:]
                            batch_y_p = batch_y[:,:,4:5,:,:]
                            pred_y, trend = self._predict(batch_x_p, batch_y_p)
                    #assert pred_y.shape == batch_y.shape
                    #assert trend.shape == batch_y.shape
                    #pred_y = pred_y + trend
                    
                    


                    if gather_data:  # return raw datas
                        results.append(dict(zip(['inputs', 'preds', 'trues', 'static'],
                                                [batch_x_p[:,:,0:1,:,:].cpu().numpy(),
                                                pred_y[:,:,0:1,:,:].cpu().numpy(),
                                                batch_y_p[:,:,0:1,:,:].cpu().numpy(),
                                                batch_static.cpu().numpy()])))
                    else:  # return metrics
                        #eval_res, _ = metric(pred_y.cpu().numpy()*batch_static.numpy(), batch_y.cpu().numpy()*batch_static.numpy(),
                        #                     data_loader.dataset.mean, data_loader.dataset.std,
                        #                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)
                        eval_res = {}
                        eval_res['train_loss'],eval_res['total_loss'],eval_res['mse'],eval_res['div'],eval_res['div_std'],eval_res['std'], eval_res['sum'] = self.criterion(pred_y, batch_y)
                        for k in eval_res.keys():
                            eval_res[k] = eval_res[k].cpu().numpy().reshape(1)
                        results.append(eval_res)

                    prog_bar.update()
                    if self.args.empty_cache:
                        torch.cuda.empty_cache()
            data_loader.dataset.perm = True
        else:
            for i, (batch_x, batch_y, batch_static) in enumerate(data_loader):
                with torch.no_grad():
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    pred_y, trend = self._predict(batch_x, batch_y)
                    #assert pred_y.shape == batch_y.shape
                    #assert trend.shape == batch_y.shape
                    #pred_y = pred_y + trend
                    
                    


                if gather_data:  # return raw datas
                    results.append(dict(zip(['inputs', 'preds', 'trues', 'static'],
                                            [batch_x[:,:,4:5,:,:].cpu().numpy(),
                                        pred_y[:,:,4:5,:,:].cpu().numpy(),
                                        batch_y[:,:,4:5,:,:].cpu().numpy(),
                                        batch_static.cpu().numpy()])))
                else:  # return metrics
                    #eval_res, _ = metric(pred_y.cpu().numpy()*batch_static.numpy(), batch_y.cpu().numpy()*batch_static.numpy(),
                    #                     data_loader.dataset.mean, data_loader.dataset.std,
                    #                     metrics=self.metric_list, spatial_norm=self.spatial_norm, return_log=False)
                    eval_res = {}
                    eval_res['train_loss'],eval_res['total_loss'],eval_res['mse'],eval_res['div'],eval_res['div_std'],eval_res['std'], eval_res['sum'] = self.criterion(pred_y, batch_y)
                    for k in eval_res.keys():
                        eval_res[k] = eval_res[k].cpu().numpy().reshape(1)
                    results.append(eval_res)

                prog_bar.update()
                if self.args.empty_cache:
                    torch.cuda.empty_cache()

        # post gather tensors
        results_all = {}
        for k in results[0].keys():
            results_all[k] = np.concatenate([batch[k] for batch in results], axis=0)
        preds = torch.tensor(results_all['preds'])
        #results['trues'] = results['trues'][:,0:1,4:5,70,65]
        trues = torch.tensor(results_all['trues'])
        #losses_m = self.criterion_cpu(preds, trues)
        static_ch = torch.tensor(results_all['static'])
        # set static_ch to be a zeros tensor with shape static_ch.shape
        #static_ch = torch.zeros_like(static_ch)
        #static_ch[:,:,0:1,64,64] = 1
        #static_ch = torch.where(static_ch > 0, torch.ones_like(static_ch), torch.zeros_like(static_ch))
        losses_m= self.criterion(preds[:,:,:], trues[:,:,:], static_ch[:,:,:], train_run=False)
        #dilate = self.val_criterion(preds, trues, static_ch)
        results_all["loss"] = losses_m
        _, total_loss, mse_loss,reg_mse,reg_std,std_loss, sum_loss = losses_m
        results_all["loss"][0] = (reg_mse)*0.001 + reg_std
        return results_all

    def vali_one_epoch(self, runner, vali_loader, **kwargs):
        """Evaluate the model with val_loader.

        Args:
            runner: the trainer of methods.
            val_loader: dataloader of validation.

        Returns:
            list(tensor, ...): The list of predictions and losses.
            eval_log(str): The string of metrics.
        """
        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(vali_loader, len(vali_loader.dataset), gather_data=False)
        else:
            results = self._nondist_forward_collect(vali_loader, len(vali_loader.dataset), gather_data=True)

        # eval_log = ""
        # for k, v in results.items():
        #     v = v.mean()
        #     if k != "loss":
        #         eval_str = f"{k}:{v.mean()}" if len(eval_log) == 0 else f", {k}:{v.mean()}"
        #         eval_log += eval_str

        return results

    def test_one_epoch(self, runner, test_loader, **kwargs):
        """Evaluate the model with test_loader.

        Args:
            runner: the trainer of methods.
            test_loader: dataloader of testing.

        Returns:
            list(tensor, ...): The list of inputs and predictions.
        """
        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(test_loader, gather_data=True)
        else:
            results = self._nondist_forward_collect(test_loader, gather_data=True)

        return results
    def grads_one_epoch(self, runner, test_loader, **kwargs):
        """Evaluate the model with test_loader.

        Args:
            runner: the trainer of methods.
            test_loader: dataloader of testing.

        Returns:
            list(tensor, ...): The list of inputs and predictions.
        """
        self.model.eval()
        if self.dist and self.world_size > 1:
            results = self._dist_forward_collect(test_loader, gather_data=True)
        else:
            results = self._nondist_forward_grad(test_loader, gather_data=True)

        return results

    def current_lr(self) -> Union[List[float], Dict[str, List[float]]]:
        """Get current learning rates.

        Returns:
            list[float] | dict[str, list[float]]: Current learning rates of all
            param groups. If the runner has a dict of optimizers, this method
            will return a dict.
        """
        lr: Union[List[float], Dict[str, List[float]]]
        if isinstance(self.model_optim, torch.optim.Optimizer):
            lr = [group['lr'] for group in self.model_optim.param_groups]
        elif isinstance(self.model_optim, dict):
            lr = dict()
            for name, optim in self.model_optim.items():
                lr[name] = [group['lr'] for group in optim.param_groups]
        else:
            raise RuntimeError(
                'lr is not applicable because optimizer does not exist.')
        return lr

    def clip_grads(self, params, norm_type: float = 2.0):
        """ Dispatch to gradient clipping method

        Args:
            parameters (Iterable): model parameters to clip
            value (float): clipping value/factor/norm, mode dependant
            mode (str): clipping mode, one of 'norm', 'value', 'agc'
            norm_type (float): p-norm, default 2.0
        """
        if self.clip_mode is None:
            return
        if self.clip_mode == 'norm':
            torch.nn.utils.clip_grad_norm_(params, self.clip_value, norm_type=norm_type)
        elif self.clip_mode == 'value':
            torch.nn.utils.clip_grad_value_(params, self.clip_value)
        elif self.clip_mode == 'agc':
            adaptive_clip_grad(params, self.clip_value, norm_type=norm_type)
        else:
            assert False, f"Unknown clip mode ({self.clip_mode})."
