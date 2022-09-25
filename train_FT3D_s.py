"""
References:
PointPWC-Net: https://github.com/DylanWusee/PointPWC
HPLFlowNet: https://github.com/laoreja/HPLFlowNet
"""

import datetime
import logging

from tqdm import tqdm 
from pathlib import Path
from collections import defaultdict

import datasets
import cmd_args
from main_utils import *
from evaluation_utils import evaluate_2d, evaluate_3d

from FLOT_utils.model import FLOT
from pseudo_labels_utils.Rigid_iter_utils import Pseudo_label_gen_module

def main():

    if 'NUMBA_DISABLE_JIT' in os.environ:
        del os.environ['NUMBA_DISABLE_JIT']

    global args
    args = cmd_args.parse_args_from_yaml(sys.argv[1])

    torch.backends.cudnn.deterministic = False
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.multi_gpu is None else '0,1'

    # Creating Dir
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s-FT3D_s-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)


    # Log
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_%s_sceneflow.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('----------------------------------------TRAINING----------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    blue = lambda x: '\033[94m' + x + '\033[0m'


    # Model and datasets
    model = FLOT(nb_iter=1)

    train_dataset = datasets.__dict__[args.dataset](
        train=True,
        num_points=args.num_points,
        data_root=args.data_root,
    )
    logger.info('train_dataset: ' + str(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )

    val_dataset = datasets.__dict__[args.dataset](
        train=False,
        num_points=args.num_points,
        data_root=args.data_root
    )
    logger.info('val_dataset: ' + str(val_dataset))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=1,
        worker_init_fn=lambda x: np.random.seed((torch.initial_seed()) % (2 ** 32))
    )


    # GPU selection and multi-GPU
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids = device_ids)
    else:
        model.cuda()


    # Optimizer and Scheduler
    init_epoch = 0
    lr_lambda = lambda epoch: 1.0 if epoch < 30 else 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    Learning_rate_clip = 1e-5

    # Loading pretrained model
    if args.pretrain is not None:
        file = torch.load(args.pretrain)
        model.load_state_dict(file["model"])
        optimizer.load_state_dict(file['optimizer'])
        scheduler.load_state_dict(file['scheduler'])
        init_epoch = file['init_epoch']

        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)

    else:
        print('Training from scratch')
        logger.info('Training from scratch')


    best_epe = 1000.0
    best_acc_3d = -1
    for epoch in range(init_epoch, args.epochs):
        lr = max(optimizer.param_groups[0]['lr'], Learning_rate_clip)
        print('Learning rate:%f'%lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


        # Training
        train_loss = train_sceneflow(model.train(), train_loader, optimizer, epoch)
        scheduler.step()
        str_out = 'EPOCH %d %s mean loss: %f'%(epoch, blue('train'), train_loss)
        print(str_out)
        logger.info(str_out)


        # Validation
        eval_epe3d, eval_acc_3d = eval_sceneflow(model.eval(), val_loader)
        str_out = 'EPOCH %d %s mean epe3d: %f mean acc_3d: %f'%(epoch, blue('eval'), eval_epe3d, eval_acc_3d)
        print(str_out)
        logger.info(str_out)


        # Saving model
        state = {
            "init_epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }
        torch.save(state, os.path.join(checkpoints_dir, "model.tar"))

        if (eval_epe3d < best_epe) or (eval_acc_3d > best_acc_3d):
            best_epe = eval_epe3d if eval_epe3d < best_epe else best_epe
            best_acc_3d = eval_acc_3d if eval_acc_3d > best_acc_3d else best_acc_3d

            if args.multi_gpu is not None:
                torch.save(model.module.state_dict(), '%s/%s_%.3d_%.4f_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, eval_epe3d, eval_acc_3d))
            else:
                torch.save(model.state_dict(), '%s/%s_%.3d_%.4f_%.4f.pth'%(checkpoints_dir, args.model_name, epoch, eval_epe3d, eval_acc_3d))
            logger.info('Save model ...')
            print('Save model ...')

        print('Best epe is: %.5f'%(best_epe))
        logger.info('Best loss is: %.5f'%(best_epe))
        print('Best acc_3d is: %.5f'%(best_acc_3d))
        logger.info('Best acc_3d is: %.5f'%(best_acc_3d))





def train_sceneflow(model, train_loader, optimizer, epoch):

    total_loss = 0
    total_seen = 0
    optimizer.zero_grad()

    for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader), smoothing=0.9):
        pos1, pos2, _, _, voxel_list = data
        pos1 = pos1.cuda()
        pos2 = pos2.cuda()
        voxel_list = voxel_list.cuda().int()

        model = model.train()
        pred_flows = model(pos1, pos2)

        # Generating pseudo labels
        with torch.no_grad():
            pseudo_gt = Pseudo_label_gen_module(pos1, pred_flows.detach().clone(), voxel_list, pos2,
                                     iter=4)
            pseudo_gt = pseudo_gt.detach()

        # Computing loss
        diff_flow = pred_flows - pseudo_gt
        loss = torch.mean(torch.abs(diff_flow))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.cpu().data * args.batch_size
        total_seen += args.batch_size

    train_loss = total_loss / total_seen
    return train_loss



def eval_sceneflow(model, loader):

    metrics = defaultdict(lambda:list())
    for batch_id, data in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        pos1, pos2, flow, _, _ = data

        pos1 = pos1.cuda()
        pos2 = pos2.cuda()
        flow = flow.cuda()

        with torch.no_grad():
            pred_flows = model(pos1, pos2)
            epe_3d, acc_3d, acc_3d_2, _ = evaluate_3d(pred_flows.detach().cpu().numpy(),
                                                     flow.detach().cpu().numpy())

        metrics['epe3d'].append(epe_3d)
        metrics['acc_3d'].append(acc_3d)

    epe3d = np.mean(metrics['epe3d'])
    acc_3d = np.mean(metrics['acc_3d'])

    return epe3d, acc_3d

if __name__ == '__main__':
    main()

