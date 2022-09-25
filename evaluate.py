"""
References:
PointPWC: https://github.com/DylanWusee/PointPWC
HPLFlowNet: https://github.com/laoreja/HPLFlowNet
"""

import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, multiprocessing as mp
import torch.nn.functional as F
import time
import torch.nn as nn
import pickle 
import datetime
import logging

from tqdm import tqdm 
from FLOT_utils.model import FLOT
from pathlib import Path
from collections import defaultdict

import datasets
import cmd_args 
from main_utils import *
from utils import geometry
from evaluation_utils import evaluate_2d, evaluate_3d

def main():

    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args 
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1'

    # Creating Dir
    experiment_dir = Path('./experiment_evaluation/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s-'%args.model_name + '%s-'%args.dataset + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')


    # Log
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '_val_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)


    # Model and datasets
    model = FLOT(nb_iter=1)

    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        num_points=args.num_points,
        data_root=args.data_root
    )
    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )


    # Loading test model
    pretrain = args.pretrain
    model.load_state_dict(torch.load(pretrain))
    print('load model %s'%pretrain)
    logger.info('load model %s'%pretrain)

    model.cuda()

    EPE = AverageMeter()
    AS = AverageMeter()
    AR = AverageMeter()
    Out = AverageMeter()

    model = model.eval()
    test_times = args.num_test
    for index_test in range(test_times):
        for i, data in tqdm(enumerate(val_loader, 0), total=len(val_loader), smoothing=0.9):
            pos1, pos2, flow, _ = data
            pos1 = pos1.cuda()
            pos2 = pos2.cuda()
            flow = flow.cuda()

            with torch.no_grad():
                pred_flows = model(pos1, pos2)

            sf_np = flow.cpu().numpy()
            pred_sf = pred_flows.cpu().numpy()

            EPE3D, acc3d_strict, acc3d_relax, outlier = evaluate_3d(pred_sf, sf_np)

            EPE.update(EPE3D)
            AS.update(acc3d_strict)
            AR.update(acc3d_relax)
            Out.update(outlier)

    res_str = (' * EPE3D {epe3d_.avg:.4f}\t'
               'ACC3DS {acc3d_s.avg:.4f}\t'
               'ACC3DR {acc3d_r.avg:.4f}\t'
               'Outliers3D {outlier_.avg:.4f}'
               .format(
                       epe3d_=EPE,
                       acc3d_s=AS,
                       acc3d_r=AR,
                       outlier_=Out
                       ))

    print(res_str)
    logger.info(res_str)


if __name__ == '__main__':
    main()




