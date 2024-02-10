# Copyright (c) CAIRI AI Lab. All rights reserved

import os
import io
import os.path as osp
import time
import logging
import json
import numpy as np
from typing import Dict, List
from fvcore.nn import FlopCountAnalysis, flop_count_table

import torch
import torch.distributed as dist

from openstl.core import Hook, metric, Recorder, get_priority, hook_maps
from openstl.methods import method_maps
from openstl.utils import (set_seed, print_log, output_namespace, check_dir, collect_env,
                           init_dist, init_random_seed,
                           get_dataset, get_dist_info, measure_throughput, weights_to_cpu)
from PIL import Image
import matplotlib.animation as animation
from clearml import Task, OutputModel
try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False

import matplotlib.pyplot as plt
import pdb

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img.convert('RGB')

def get_ani(mat):
    fig, ax = plt.subplots(figsize=(8, 8))
    imgs = []
    for img in mat:
        img = ax.imshow(img, animated=True, vmax=30, vmin=0)
        imgs.append([img])
    ani = animation.ArtistAnimation(fig, imgs, interval=1000, blit=True, repeat_delay=3000)
    plt.close()
    return ani.to_html5_video()

def plot_tmaps(true, pred, inputs, epoch, logger):
    logger.current_logger().report_media(
            "viz", "true frames", iteration=epoch, stream=get_ani(true), file_extension='html')

    logger.current_logger().report_media(
                "viz", "pred frames", iteration=epoch, stream=get_ani(pred), file_extension='html')
    logger.current_logger().report_media(
                "viz", "input frames", iteration=epoch, stream=get_ani(inputs), file_extension='html')

class BaseExperiment(object):
    """The basic class of PyTorch training and evaluation."""

    def __init__(self, args, task, dataloaders=None):
        """Initialize experiments (non-dist as an example)"""
        self.task = task
        self.args = args
        self.config = self.args.__dict__
        self.device = self.args.device
        self.method = None
        self.args.method = self.args.method.lower()
        self._epoch = 0
        self._iter = 0
        self._inner_iter = 0
        self._max_epochs = self.config['epoch']
        self._max_iters = None
        self._hooks: List[Hook] = []
        self._rank = 0
        self._world_size = 1
        self._dist = self.args.dist
        self._early_stop = self.args.early_stop_epoch
        
        self.losses = ['val_train_loss', 'val_total_loss', 'val_mse_loss', 'reg_mse', "reg_std", 'std_loss', 'sum_loss']

        self._preparation(dataloaders)
        if self._rank == 0:
            print_log(output_namespace(self.args))
            if not self.args.no_display_method_info:
                self.display_method_info()

    def _acquire_device(self):
        """Setup devices"""
        if self.args.use_gpu:
            self._use_gpu = True
            if self.args.dist:
                device = f'cuda:{self._rank}'
                torch.cuda.set_device(self._rank)
                print_log(f'Use distributed mode with GPUs: local rank={self._rank}')
            else:
                device = torch.device('cuda:0')
                print_log(f'Use non-distributed mode with GPU: {device}')
        else:
            self._use_gpu = False
            device = torch.device('cpu')
            print_log('Use CPU')
            if self.args.dist:
                assert False, "Distributed training requires GPUs"
        return device

    def _preparation(self, dataloaders=None):
        """Preparation of environment and basic experiment setups"""
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(self.args.local_rank)

        # init distributed env first, since logger depends on the dist info.
        if self.args.launcher != 'none' or self.args.dist:
            self._dist = True
        if self._dist:
            assert self.args.launcher != 'none'
            dist_params = dict(backend='nccl', init_method='env://')
            if self.args.launcher == 'slurm':
                dist_params['port'] = self.args.port
            init_dist(self.args.launcher, **dist_params)
            self._rank, self._world_size = get_dist_info()
            # re-set gpu_ids with distributed training mode
            self._gpu_ids = range(self._world_size)
        self.device = self._acquire_device()
        if self._early_stop <= self._max_epochs // 5:
            self._early_stop = self._max_epochs * 2

        # log and checkpoint
        base_dir = self.args.res_dir if self.args.res_dir is not None else 'work_dirs'
        try:
            task = Task.get_task(task_id=self.args.ex_name)
            #model_path = task.artifacts['best_model_weights'].get_local_copy()
            model_path = task.artifacts['latest_model_weights'].get_local_copy()
            # copy the model at location self.path to ./work_dirs/task.name
            # but make the dir before if it doesnt exist
            if not os.path.exists(f"{base_dir}/{task.name}"):
                os.makedirs(f"{base_dir}/{task.name}/checkpoints")
            #os.system(f"cp {model_path} {base_dir}/{task.name}/checkpoint.pth")
            os.system(f"cp {model_path} {base_dir}/{task.name}/checkpoints/latest.pth")
            self.args.ex_name = task.name
        except:
            print ("Not a clearml task. Using local directory")

        self.path = osp.join(base_dir, self.args.ex_name if not self.args.ex_name.startswith(self.args.res_dir) \
        else self.args.ex_name.split(self.args.res_dir+'/')[-1])
        self.checkpoints_path = osp.join(self.path, 'checkpoints')
        if self._rank == 0:
            check_dir(self.path)
            check_dir(self.checkpoints_path)

        sv_param = osp.join(self.path, 'model_param.json')
        if self._rank == 0:
            with open(sv_param, 'w') as file_obj:
                json.dump(self.args.__dict__, file_obj)

            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            prefix = 'train' if (not self.args.test and not self.args.inference) else 'test'
            logging.basicConfig(level=logging.INFO,
                                filename=osp.join(self.path, '{}_{}.log'.format(prefix, timestamp)),
                                filemode='a', format='%(asctime)s - %(message)s')

        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        if self._rank == 0:
            print_log('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

        # set random seeds
        if self._dist:
            seed = init_random_seed(self.args.seed)
            seed = seed + dist.get_rank() if self.args.diff_seed else seed
        else:
            seed = self.args.seed
        set_seed(seed)

        # prepare data
        self._get_data(dataloaders)
        # build the method
        self._build_method()
        # build hooks
        self._build_hook()
        # resume traing
        if self.args.auto_resume:
            self.args.resume_from = osp.join(self.checkpoints_path, 'latest.pth')
        if self.args.resume_from is not None:

            self._load(name=self.args.resume_from)
        self.call_hook('before_run')

    def _build_method(self):
        self.steps_per_epoch = len(self.train_loader)
        self.method = method_maps[self.args.method](self.args, self.device, self.steps_per_epoch)
        self.method.model.eval()
        # setup ddp training
        if self._dist:
            self.method.model.cuda()
            if self.args.torchscript:
                self.method.model = torch.jit.script(self.method.model)
            self.method._init_distributed()

    def _build_hook(self):
        for k in self.args.__dict__:
            if k.lower().endswith('hook'):
                hook_cfg = self.args.__dict__[k].copy()
                priority = get_priority(hook_cfg.pop('priority', 'NORMAL'))
                hook = hook_maps[k.lower()](**hook_cfg)
                if hasattr(hook, 'priority'):
                    raise ValueError('"priority" is a reserved attribute for hooks')
                hook.priority = priority  # type: ignore
                # insert the hook to a sorted list
                inserted = False
                for i in range(len(self._hooks) - 1, -1, -1):
                    if priority >= self._hooks[i].priority:  # type: ignore
                        self._hooks.insert(i + 1, hook)
                        inserted = True
                        break
                if not inserted:
                    self._hooks.insert(0, hook)

    def call_hook(self, fn_name: str) -> None:
        """Run hooks by the registered names"""
        for hook in self._hooks:
            getattr(hook, fn_name)(self)

    def _get_hook_info(self):
        """Get hook information in each stage"""
        stage_hook_map: Dict[str, list] = {stage: [] for stage in Hook.stages}
        for hook in self._hooks:
            priority = hook.priority  # type: ignore
            classname = hook.__class__.__name__
            hook_info = f'({priority:<12}) {classname:<35}'
            for trigger_stage in hook.get_triggered_stages():
                stage_hook_map[trigger_stage].append(hook_info)

        stage_hook_infos = []
        for stage in Hook.stages:
            hook_infos = stage_hook_map[stage]
            if len(hook_infos) > 0:
                info = f'{stage}:\n'
                info += '\n'.join(hook_infos)
                info += '\n -------------------- '
                stage_hook_infos.append(info)
        return '\n'.join(stage_hook_infos)

    def _get_data(self, dataloaders=None):
        """Prepare datasets and dataloaders"""
        if dataloaders is None:
            self.train_loader, self.vali_loader, self.test_loader = \
                get_dataset(self.args.dataname, self.config)
        else:
            self.train_loader, self.vali_loader, self.test_loader = dataloaders

        if self.vali_loader is None:
            self.vali_loader = self.test_loader
        self._max_iters = self._max_epochs * len(self.train_loader)

    def _save(self, name=''):
        """Saving models and meta data to checkpoints"""
        checkpoint = {
            'epoch': self._epoch + 1,
            'optimizer': self.method.model_optim.state_dict(),
            'state_dict': weights_to_cpu(self.method.model.state_dict()) \
                if not self._dist else weights_to_cpu(self.method.model.module.state_dict()),
            'scheduler': self.method.scheduler.state_dict()}
        torch.save(checkpoint, osp.join(self.checkpoints_path, name + '.pth'))

    def _load(self, name=''):
        """Loading models from the checkpoint"""
        filename = name if osp.isfile(name) else osp.join(self.checkpoints_path, name + '.pth')
        try:
            checkpoint = torch.load(filename)
        except:
            return
        # OrderedDict is a subclass of dict
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f'No state_dict found in checkpoint file {filename}')
        self._load_from_state_dict(checkpoint['state_dict'])
        # if checkpoint.get('epoch', None) is not None:
        #     self._epoch = checkpoint['epoch']
        #     self.method.model_optim.load_state_dict(checkpoint['optimizer'])
        #     self.method.scheduler.load_state_dict(checkpoint['scheduler'])

    def _load_from_state_dict(self, state_dict):
        if self._dist:
            try:
                self.method.model.module.load_state_dict(state_dict)
            except:
                self.method.model.load_state_dict(state_dict)
        else:
            self.method.model.load_state_dict(state_dict)

    def display_method_info(self):
        """Plot the basic infomation of supported methods"""
        T, C, H, W = self.args.in_shape
        if self.args.method in ['simvp', 'unet', 'tau', 'simvpresid', 'unetresid', 'simvpgan']:
            input_dummy = torch.ones(1, self.args.pre_seq_length, C, H, W).to(self.device)
        elif self.args.method == 'simvprnn':
            Hp, Wp = 32, 32
            Cp = 32
            _tmp_input = torch.ones(1, self.args.total_length, C, H, W).to(self.device)
            _tmp_flag = torch.ones(1, self.args.aft_seq_length - 1, Cp, Hp, Wp).to(self.device)
            input_dummy = (_tmp_input, _tmp_flag)

        elif self.args.method == 'crevnet':
            # crevnet must use the batchsize rather than 1
            input_dummy = torch.ones(self.args.batch_size, 20, C, H, W).to(self.device)
        elif self.args.method == 'phydnet':
            _tmp_input1 = torch.ones(1, self.args.pre_seq_length, C, H, W).to(self.device)
            _tmp_input2 = torch.ones(1, self.args.aft_seq_length, C, H, W).to(self.device)
            _tmp_constraints = torch.zeros((49, 7, 7)).to(self.device)
            input_dummy = (_tmp_input1, _tmp_input2, _tmp_constraints)
        elif self.args.method in ['convlstm', 'predrnnpp', 'predrnn', 'mim', 'e3dlstm', 'mau']:
            Hp, Wp = H // self.args.patch_size, W // self.args.patch_size
            Cp = self.args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, self.args.total_length, Hp, Wp, Cp).to(self.device)
            _tmp_flag = torch.ones(1, self.args.aft_seq_length - 1, Hp, Wp, Cp).to(self.device)
            input_dummy = (_tmp_input, _tmp_flag)
        elif self.args.method == 'predrnnv2':
            Hp, Wp = H // self.args.patch_size, W // self.args.patch_size
            Cp = self.args.patch_size ** 2 * C
            _tmp_input = torch.ones(1, self.args.total_length, Hp, Wp, Cp).to(self.device)
            _tmp_flag = torch.ones(1, self.args.total_length - 2, Hp, Wp, Cp).to(self.device)
            input_dummy = (_tmp_input, _tmp_flag)
        elif self.args.method == 'dmvfn':
            input_dummy = torch.ones(1, 3, C, H, W, requires_grad=True).to(self.device)
        elif self.args.method == 'prednet':
           input_dummy = torch.ones(1, 1, C, H, W, requires_grad=True).to(self.device)
        else:
            raise ValueError(f'Invalid method name {self.args.method}')

        dash_line = '-' * 80 + '\n'
        info = self.method.model.__repr__()
        flops = FlopCountAnalysis(self.method.model, input_dummy)
        flops = flop_count_table(flops)
        if self.args.fps:
            fps = measure_throughput(self.method.model, input_dummy)
            fps = 'Throughputs of {}: {:.3f}\n'.format(self.args.method, fps)
        else:
            fps = ''
        print_log('Model info:\n' + info+'\n' + flops+'\n' + fps + dash_line)

    def train(self):
        """Training loops of STL methods"""
        recorder = Recorder(verbose=True, early_stop_time=min(self._max_epochs // 10, 10))
        num_updates = self._epoch * self.steps_per_epoch
        early_stop = False
        self.call_hook('before_train_epoch')

        logger = self.task.get_logger()

        eta = 1.0  # PredRNN variants
        for epoch in range(self._epoch, self._max_epochs):
            if self._dist and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            num_updates, loss_mean, loss_total, loss_mse, loss_reg, loss_div, loss_divs, loss_sum, eta = self.method.train_one_epoch(self, self.train_loader,
                                                                      epoch, num_updates, eta)

            self._epoch = epoch
            if epoch % self.args.log_step == 0:
                cur_lr = self.method.current_lr()
                cur_lr = sum(cur_lr) / len(cur_lr)
                with torch.no_grad():
                    vali_loss = self.vali(logger, epoch)

                if self._rank == 0:

                    print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}\n'.format(
                        epoch + 1, len(self.train_loader), cur_lr, loss_mean.avg, vali_loss))
                    logger.report_scalar(title='Training Report', 
                        series='Train Loss', value=loss_mean.avg, iteration=epoch)
                    logger.report_scalar(title='Training Report', 
                        series='Train MSE Loss', value=loss_mse.avg, iteration=epoch)
                    logger.report_scalar(title='Training Report', 
                        series='Train Reg MSE', value=loss_reg.avg, iteration=epoch)
                    logger.report_scalar(title='Training Report', 
                        series='Train Reg Std', value=loss_div.avg, iteration=epoch)
                    logger.report_scalar(title='Training Report',
                        series='Train Std', value=loss_divs.avg, iteration=epoch)
                    logger.report_scalar(title='Training Report',
                        series='Train total loss', value=loss_total.avg, iteration=epoch)
                    logger.report_scalar(title='Training Report',
                        series='Train sum loss', value=loss_sum.avg, iteration=epoch) 
                    early_stop =recorder(vali_loss, self.method.model, self.path, epoch)
                    self._save(name='latest')
            if self._use_gpu and self.args.empty_cache:
                torch.cuda.empty_cache()
            # if epoch > self._early_stop and early_stop:  # early stop training
            #     print_log('Early stop training at f{} epoch'.format(epoch))

        if not check_dir(self.path):  # exit training when work_dir is removed
            assert False and "Exit training because work_dir is removed"
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        latest_model_path = osp.join(self.path, 'checkpoints/latest.pth')
        self._load_from_state_dict(torch.load(best_model_path))
        self.task.upload_artifact(artifact_object=best_model_path, name='best_model_weights')
        self.task.upload_artifact(artifact_object=latest_model_path, name='latest_model_weights')
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')

    def vali(self, logger, epoch):
        """A validation loop during training"""
        self.call_hook('before_val_epoch')
        results = self.method.vali_one_epoch(self, self.vali_loader)
        for loss, name in zip(results['loss'], self.losses):
            logger.report_scalar(title='Training Report',
                        series=name, value=loss.cpu().numpy(), iteration=epoch)
        # subtract results['inputs'] by its temporal mean along first dimension using numpy
        #results['inputs'] = results['inputs'] - np.mean(results['inputs'], axis=1, keepdims=True)

        plot_tmaps(results['trues'][200,:,0,:,:,np.newaxis], results['preds'][200,:,0,:,:,np.newaxis],
                    results['inputs'][200,:,0,:,:,np.newaxis], epoch, logger)

        shift_amount = 12  # Define the amount by which you want to shift the 'inputs' on the x-axis

        x_values = range(len(results['trues'][19, :, 0, 64, 64]))
        shifted_x_values = [x - shift_amount for x in x_values]  # Shift x-values for 'inputs'

        # Define the list of pixel coordinates
        pixel_list = [(64, 64), (51, 56), (51, 40)]

        # Iterate over each time step for which we want to visualize the data
        for i in [10, 50, 100, 150]:
            # Iterate over each pixel coordinate
            for pixel in pixel_list:
                y, x = pixel  # Unpack the tuple into x and y coordinates

                # Plot the inputs, true values, and predictions for each pixel
                plt.plot(shifted_x_values, results['inputs'][i, :, 0, y, x], label=f"Inputs at {pixel}")
                plt.plot(x_values, results['trues'][i, :, 0, y, x], label=f"True at {pixel}")
                plt.plot(x_values, results['preds'][i, :, 0, y, x], label=f"Preds_m at {pixel}")

                # Show the legend and get the current figure
                plt.legend()
                fig = plt.gcf()  # Get the current figure

                # Log the figure using the logger for the specific pixel and time step
                logger.report_matplotlib_figure(
                    f"px_{pixel}_{i}",
                    "true and pred",
                    iteration=epoch,
                    figure=fig,
                    report_image=True,
                    report_interactive=True
                )
            # After plotting for all pixels at the current time step, close the plot to avoid overlap
            plt.close()

        # The second part of the visualization seems to be a time series plot for the first pixel.
        # If similar time series plots are required for all pixels, iterate over the pixel list:
        for pixel in pixel_list:
            y, x = pixel  # Unpack the tuple into x and y coordinates

            # Plot the true values and predictions over the specified range for each pixel
            plt.plot(results['trues'][:240, 0, 0, y, x], label=f"True at {pixel}")
            plt.plot(results['preds'][:240, 0, 0, y, x], label=f"Preds_m at {pixel}")

            # Show the legend and get the current figure
            plt.legend()
            fig = plt.gcf()  # Get the current figure

            # Log the figure using the logger for the specific pixel
            logger.report_matplotlib_figure(
                f"px_{pixel}_2",
                "true and pred",
                iteration=epoch,
                figure=fig,
                report_image=True,
                report_interactive=True
            )
            # After plotting for the current pixel, close the plot to avoid overlap
            plt.close()

        self.call_hook('after_val_epoch')

        if self._rank == 0:
            if 'weather' in self.args.dataname:
                metric_list, spatial_norm = ['mse', 'rmse', 'mae'], True
            else:
                metric_list, spatial_norm = ['mse', 'mae'], False
            eval_res, eval_log = metric(results["preds"][:], results["trues"], self.vali_loader.dataset.mean, self.vali_loader.dataset.std,
                                        metrics=metric_list, spatial_norm=spatial_norm)

            print_log('val\t '+eval_log)
            if has_nni:
                nni.report_intermediate_result(eval_res['mse'].mean())

        return results['loss'][0]

    def test(self):
        """A testing loop of STL methods"""
        if self.args.test:
            best_model_path = osp.join(self.path, 'checkpoint.pth')
            #best_model_path = osp.join(self.path, 'checkpoints/latest.pth')
            self._load_from_state_dict(torch.load(best_model_path))
            #self._load(best_model_path)


        self.call_hook('before_val_epoch')
        results = self.method.test_one_epoch(self, self.test_loader)
        self.call_hook('after_val_epoch')

        # inputs is of shape (240,12,8,128,128), sum the first axis and get non-zero indices as a binary mask of shape (240, 1, 8, 128, 128)


        # TODO Fix inp_mean calculation since adding by results will make it expand dims

        inp_mean = np.mean(results["inputs"], axis=1, keepdims=True)
        results["preds"] = ((results["preds"]+inp_mean)*self.train_loader.dataset.s[0,4,0,0])+self.train_loader.dataset.m[0,4,0,0]
        results["trues"] = ((results["trues"]+inp_mean)*self.train_loader.dataset.s[0,4,0,0])+self.train_loader.dataset.m[0,4,0,0]
        results["inputs"] = ((results["trues"]+inp_mean)*self.train_loader.dataset.s[0,4,0,0])+self.train_loader.dataset.m[0,4,0,0]
        # Add a dimension to self.train_loader.dataset.s and self.train_loader.dataset.m
        # norm_mean = np.expand_dims(self.train_loader.dataset.m, axis=0)
        # norm_std = np.expand_dims(self.train_loader.dataset.s, axis=0
        #results["preds"] = ((results["preds"] + inp_mean)*norm_std[:,:,4:5])+norm_mean[:,:,4:5]
        #trues = trues[:,:,0::2]
        #preds = preds[:,:,0::2]
        #trues = trues[:,:,2:3]#, 62-10,92-40]

        if 'weather' in self.args.dataname:
            metric_list, spatial_norm = self.args.metrics, True
            channel_names = self.test_loader.dataset.data_name if 'mv' in self.args.dataname else None
        else:
            metric_list, spatial_norm, channel_names = self.args.metrics, False, None
        eval_res, eval_log = metric(results['preds'], results['trues'],
                                    self.test_loader.dataset.mean, self.test_loader.dataset.std,
                                    metrics=metric_list, channel_names=channel_names, spatial_norm=spatial_norm)
        results['metrics'] = np.array([eval_res['mae'], eval_res['mse']])

        if self._rank == 0:
            print_log(eval_log)
            folder_path = osp.join(self.path, 'saved_comb')
            check_dir(folder_path)

            if self.args.ex_name.endswith('unet'):
                for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                    np.save(osp.join(folder_path, np_data + '.npy'), results[np_data])
            else:
                for np_data in ['metrics', 'trues', 'preds']:
                    np.save(osp.join(folder_path, np_data + '.npy'), results[np_data])

        return eval_res['mse']

    def inference(self, best_model=True):
        """A inference loop of STL methods"""
        if best_model:
            best_model_path = osp.join(self.path, 'checkpoint.pth')
            self._load_from_state_dict(torch.load(best_model_path))
        else:
            best_model_path = osp.join(self.path, 'checkpoints/latest.pth')
            self._load(best_model_path)
        print ("loaded from ", best_model_path)

        self.call_hook('before_val_epoch')
        results = self.method.test_one_epoch(self, self.test_loader)
        
        self.call_hook('after_val_epoch')
        inp_mean = np.mean(results["inputs"], axis=1, keepdims=True)
        results["trues"] = (results["trues"]+inp_mean)
        results["preds"] = (results["preds"]+np.expand_dims(inp_mean, axis=1))
        results["inputs"] = (results["inputs"]).astype(np.uint8)

        # clamp trues and preds to be between 0 and 255 and convert to uint8
        results["trues"] = np.clip(results["trues"], 0, 255).astype(np.uint8)
        results["preds"] = np.clip(results["preds"], 0, 255).astype(np.uint8)

        # if self._rank == 0:
        #     folder_path = osp.join(self.path, 'saved1')
        #     check_dir(folder_path)
        #     for np_data in ['inputs', 'trues', 'preds']:
        #         np.save(osp.join(folder_path, np_data + '.npy'), results[np_data])

        return None

