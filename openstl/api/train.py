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
        img = ax.imshow(img, animated=True, vmax=30,vmin=0)
        imgs.append([img])
    ani = animation.ArtistAnimation(fig, imgs, interval=1000, blit=True, repeat_delay=3000)
    plt.close()
    return ani.to_html5_video()

def plot_tmaps(true, pred, epoch, logger):
    logger.current_logger().report_media(
            "viz", "true frames", iteration=epoch, stream=get_ani(true), file_extension='html')

    logger.current_logger().report_media(
                "viz", "pred frames", iteration=epoch, stream=get_ani(pred), file_extension='html')
class BaseExperiment(object):
    """The basic class of PyTorch training and evaluation."""

    def __init__(self, args, task):
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
        self.losses = ['val_train_loss', 'val_total_loss', 'val_main_loss', 'val_reg_loss', "val_div_loss", 'val_std_div', 'val_sum']

        self._preparation()
        if self._rank == 0:
            print_log(output_namespace(self.args))
            self.display_method_info()

    def _acquire_device(self):
        """Setup devices"""
        if self.args.use_gpu:
            self._use_gpu = True
            if self.args.dist:
                device = f'cuda:{self._rank}'
                torch.cuda.set_device(self._rank)
                print(f'Use distributed mode with GPUs: local rank={self._rank}')
            else:
                device = torch.device('cuda:0')
                print('Use non-distributed mode with GPU:', device)
        else:
            self._use_gpu = False
            device = torch.device('cpu')
            print('Use CPU')
            if self.args.dist:
                assert False, "Distributed training requires GPUs"
        return device

    def _preparation(self):
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

        # log and checkpoint
        base_dir = self.args.res_dir if self.args.res_dir is not None else 'work_dirs'
        try:
            task = Task.get_task(task_id=self.args.ex_name)
            model_path = task.artifacts['best_model_weights'].get_local_copy()
            #model_path = task.artifacts['latest_model_weights'].get_local_copy()
            # copy the model at location self.path to ./work_dirs/task.name
            # but make the dir before if it doesnt exist
            if not os.path.exists(f"{base_dir}/{task.name}"):
                os.makedirs(f"{base_dir}/{task.name}/checkpoints")
            os.system(f"cp {model_path} {base_dir}/{task.name}/checkpoint.pth")
            #os.system(f"cp {model_path} {base_dir}/{task.name}/checkpoints/latest.pth")
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
        prefix = 'train' if not self.args.test else 'test'
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
        self._get_data()
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

    def _get_data(self):
        """Prepare datasets and dataloaders"""
        self.train_loader, self.vali_loader, self.test_loader = \
            get_dataset(self.args.dataname, self.config)
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
        if self.args.method in ['simvp','unet'] :
            input_dummy = torch.ones(1, self.args.pre_seq_length, C, H, W).to(self.device)
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
        recorder = Recorder(verbose=True)
        num_updates = self._epoch * self.steps_per_epoch
        self.call_hook('before_train_epoch')

        logger = self.task.get_logger()

        eta = 1.0  # PredRNN variants
        for epoch in range(self._epoch, self._max_epochs):
            if self._dist and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            num_updates, loss_mean, loss_mse, loss_reg, loss_div, loss_divs, loss_total, loss_sum, eta = self.method.train_one_epoch(self, self.train_loader,
                                                                      epoch, num_updates, eta)

            self._epoch = epoch
            if epoch % self.args.log_step == 0:
                cur_lr = self.method.current_lr()
                cur_lr = sum(cur_lr) / len(cur_lr)
                with torch.no_grad():
                    vali_loss, eval_res = self.vali(self.vali_loader, logger, epoch)

                if self._rank == 0:

                    print_log('Epoch: {0}, Steps: {1} | Lr: {2:.7f} | Train Loss: {3:.7f} | Vali Loss: {4:.7f}\n'.format(
                        epoch + 1, len(self.train_loader), cur_lr, loss_mean.avg, vali_loss))
                    logger.report_scalar(title='Training Report', 
                        series='Train Loss', value=loss_mean.avg, iteration=epoch)
                    logger.report_scalar(title='Training Report', 
                        series='Train MSE Loss', value=loss_mse.avg, iteration=epoch)
                    logger.report_scalar(title='Training Report', 
                        series='Train Reg Loss', value=loss_reg.avg, iteration=epoch)
                    logger.report_scalar(title='Training Report', 
                        series='Train Div Loss', value=loss_div.avg, iteration=epoch)
                    logger.report_scalar(title='Training Report',
                        series='Train Div Std', value=loss_divs.avg, iteration=epoch)
                    logger.report_scalar(title='Training Report',
                        series='Train total loss', value=loss_total.avg, iteration=epoch)
                    logger.report_scalar(title='Training Report',
                        series='Train sum loss', value=loss_sum.avg, iteration=epoch)

                    #logger.report_scalar(title='Training Report',
                    #    series='Val Loss', value=vali_loss, iteration=epoch)

                    recorder(vali_loss, self.method.model, self.path, epoch)
                    self._save(name='latest')
            if self._use_gpu and self.args.empty_cache:
                torch.cuda.empty_cache()

        if not check_dir(self.path):  # exit training when work_dir is removed
            assert False and "Exit training because work_dir is removed"
        best_model_path = osp.join(self.path, 'checkpoint.pth')
        latest_model_path = osp.join(self.path, 'checkpoints/latest.pth')
        self._load_from_state_dict(torch.load(best_model_path))
        self.task.upload_artifact(artifact_object=best_model_path, name='best_model_weights')
        self.task.upload_artifact(artifact_object=latest_model_path, name='latest_model_weights')
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_run')


    def vali(self, vali_loader, logger, epoch):
        """A validation loop during training"""
        self.call_hook('before_val_epoch')
        preds, trues, val_loss = self.method.vali_one_epoch(self, self.vali_loader)
        for loss, name in zip(val_loss, self.losses):
            logger.report_scalar(title='Training Report',
                        series=name, value=loss.cpu().numpy(), iteration=epoch)

        plot_tmaps(trues[200,:,0,:,:,np.newaxis], preds[200,:,0,:,:,np.newaxis], epoch, logger)

        for i in [[52,76],[73,54],[76,101],[100,105]]:
            plt.plot(trues[200,:,0,i[0],i[1]], label="True")
            plt.plot(preds[200,:,0,i[0],i[1]], label="Preds")
            plt.legend()
            fig = plt.gcf()  # Get the current figure

            logger.report_matplotlib_figure(
                "px_"+str(i),
                "true and pred",
                iteration=epoch,
                figure = fig,
                report_image = True,
               report_interactive = True
                )
            plt.close()
        self.call_hook('after_val_epoch')

        if self._rank == 0:
            if 'weather' in self.args.dataname:
                metric_list, spatial_norm = ['mse', 'rmse', 'mae'], True
            else:
                metric_list, spatial_norm = ['mse', 'mae'], False
             
            eval_res, eval_log = metric(preds, trues, vali_loader.dataset.mean, vali_loader.dataset.std,
                                        metrics=metric_list, spatial_norm=spatial_norm)

            print_log('val\t '+eval_log)
            if has_nni:
                nni.report_intermediate_result(eval_res['mse'])

        return val_loss[0].cpu().numpy(), eval_res

    def test(self):
        """A testing loop of STL methods"""
        if self.args.test:
            best_model_path = osp.join(self.path, 'checkpoint.pth')
            #best_model_path = osp.join(self.path, 'checkpoints/latest.pth')
            self._load_from_state_dict(torch.load(best_model_path))
            #self._load(best_model_path)


        self.call_hook('before_val_epoch')
        inputs, preds, trues = self.method.test_one_epoch(self, self.test_loader)
        trues = trues[:,:,0::2]
        self.call_hook('after_val_epoch')

        # inputs is of shape (240,12,8,128,128), sum the first axis and get non-zero indices as a binary mask of shape (240, 1, 8, 128, 128)
        
        inp_sum = inputs[:,:,0::2].sum(axis=1, keepdims=True)
        mask = (inp_sum > 0).astype(np.float32)
        
        
        #trues = trues[:,:,0::2]
        #preds = preds[:,:,0::2]
        #trues = trues[:,:,2:3]#, 62-10,92-40]
        trues = trues[:,:,:]# 62-10,92-40]
        # multiply mask by preds
        preds = preds #* mask
        #preds=  preds[:,:,2:3]#, 62-10,92-40]
        preds=  preds[:,:,:]#, 62-10,92-40]

        if 'weather' in self.args.dataname:
            metric_list, spatial_norm = ['mse', 'rmse', 'mae'], True
        else:
            metric_list, spatial_norm = ['mse', 'mae'], False
        eval_res, eval_log = metric(preds, trues, self.test_loader.dataset.mean, self.test_loader.dataset.std,
                                    metrics=metric_list, spatial_norm=spatial_norm)
        metrics = np.array([eval_res['mae'], eval_res['mse']])

        if self._rank == 0:
            print_log(eval_log)
            folder_path = osp.join(self.path, 'saved_comb_training')
            check_dir(folder_path)

            # check if self.args.exp_name ends with unet
            if self.args.ex_name.endswith('unet'):
                for np_data in ['metrics', 'inputs', 'trues', 'preds']:
                    np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
            else:
                for np_data in ['metrics', 'trues', 'preds']:
                    np.save(osp.join(folder_path, np_data + '.npy'), vars()[np_data])
        return eval_res['mse']