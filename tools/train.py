# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
from clearml import Task
warnings.filterwarnings('ignore')

from openstl.api import BaseExperiment
from openstl.utils import (create_parser, default_parser, get_dist_info, load_config,
                           setup_multi_processes, update_config)
import pdb
try:
    import nni
    has_nni = True
except ImportError: 
    has_nni = False
import torch

if __name__ == '__main__':

    parser = create_parser()
    parser.add_argument("--local-rank", type=int)
    args = parser.parse_args()
    config = args.__dict__

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    cfg_path = osp.join('./configs', args.dataname, f'{args.method}.py') \
        if args.config_file is None else args.config_file
    if args.overwrite:
        config = update_config(config, load_config(cfg_path),
                               exclude_keys=['method', 'data_root', 'perm'])
    else:
        loaded_cfg = load_config(cfg_path)
        config = update_config(config, loaded_cfg,
                               exclude_keys=['method', 'batch_size', 'val_batch_size',
                                             'drop_path', 'warmup_epoch'])
        default_values = default_parser()
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]

    task = Task.init(project_name='simvp/e5/q3', task_name=config['ex_name'])
    task.connect_configuration(config)
    # set multi-process settings

    torch.cuda.set_device(args.local_rank)
    setup_multi_processes(config)

    print('>'*35 + ' training ' + '<'*35)
    exp = BaseExperiment(args, task)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print('>'*35 + ' testing  ' + '<'*35)
    mse = exp.test()

    if rank == 0 and has_nni:
        nni.report_final_result(mse)
