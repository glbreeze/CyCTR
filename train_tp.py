import os
import random
import time
import cv2
import numpy as np
import logging
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboardX import SummaryWriter

from util import dataset
from util import transform, config
from util.util import AverageMeter, poly_learning_rate, step_learning_rate, intersectionAndUnionGPU



parser = argparse.ArgumentParser(description='PyTorch Semantic Segmentation')
parser.add_argument('--config', type=str, default='config/pascal/pascal_split0_resnet50.yaml', help='config file')
parser.add_argument('opts', help='see config/ade20k/ade20k_pspnet50.yaml for all options', default=None, nargs=argparse.REMAINDER)
args = parser.parse_args([])
cfg = config.load_cfg_from_cfg_file(args.config)
if args.opts is not None:
    cfg = config.merge_cfg_from_list(cfg, args.opts)

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

# ========================================== Main ==========================================
args = cfg
assert args.classes > 1
assert (args.train_h - 1) % 8 == 0 and (args.train_w - 1) % 8 == 0
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
if args.manual_seed is not None:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
    random.seed(args.manual_seed)

### multi-processing training is deprecated
if args.dist_url == "env://" and args.world_size == -1:
    args.world_size = int(os.environ["WORLD_SIZE"])
args.distributed = args.world_size > 1 or args.multiprocessing_distributed
args.ngpus_per_node = len(args.train_gpu)
if len(args.train_gpu) == 1:
    args.sync_bn = False  # sync_bn is deprecated
    args.distributed = False
    args.multiprocessing_distributed = False

# main_worker(args.train_gpu, args.ngpus_per_node, args)

model = CyCTR(layers=args.layers, shot=args.shot, reduce_dim=args.hidden_dims, with_transformer=args.with_transformer)

param_dicts = [{"params": [p for n, p in model.named_parameters() if "transformer" not in n
                and p.requires_grad]},]
transformer_param_dicts = [{
        "params": [p for n, p in model.named_parameters() if "transformer" in n and "bias" not in n and p.requires_grad],
        "lr": 1e-4,
        "weight_decay": 1e-2,},
        {"params": [p for n, p in model.named_parameters() if
        "transformer" in n and "bias" in n and p.requires_grad],
        "lr": 1e-4,
        "weight_decay": 0,}]
optimizer = torch.optim.SGD(
    param_dicts,
    lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
base_lrs = [pg['lr'] for pg in optimizer.param_groups]
transformer_optimizer = torch.optim.AdamW(transformer_param_dicts, lr=1e-4, weight_decay=1e-4)

global logger, writer
logger = get_logger()
writer = SummaryWriter(args.save_path)
logger.info("=> creating model ...")
logger.info("Classes: {}".format(args.classes))
logger.info(model)
print(args)

model = torch.nn.DataParallel(model.cuda())

if args.weight:
    if os.path.isfile(args.weight):
        logger.info("=> loading weight '{}'".format(args.weight))
        checkpoint = torch.load(args.weight)
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("=> loaded weight '{}'".format(args.weight))
    else:
        logger.info("=> no weight found at '{}'".format(args.weight))

if args.resume:
    if os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))


value_scale = 255
mean = [0.485, 0.456, 0.406]
mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]
std = [item * value_scale for item in std]

assert args.split in [0, 1, 2, 3, 999]
train_transform = [
    transform.RandScale([args.scale_min, args.scale_max]),
    transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.padding_label),
    transform.RandomGaussianBlur(),
    transform.RandomHorizontalFlip(),
    transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.padding_label),
    transform.ToTensor(),
    transform.Normalize(mean=mean, std=std)]
train_transform = transform.Compose(train_transform)
train_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, \
                             data_list=args.train_list, transform=train_transform, mode='train', \
                             use_coco=args.use_coco, use_split_coco=args.use_split_coco)

train_sampler = None
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                           num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                           drop_last=True)

if args.evaluate:
    if args.resized_val:
        val_transform = transform.Compose([
            transform.Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    else:
        val_transform = transform.Compose([
            transform.test_Resize(size=args.val_size),
            transform.ToTensor(),
            transform.Normalize(mean=mean, std=std)])
    val_data = dataset.SemData(split=args.split, shot=args.shot, data_root=args.data_root, \
                               data_list=args.val_list, transform=val_transform, mode='val', \
                               use_coco=args.use_coco, use_split_coco=args.use_split_coco)
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                             num_workers=args.workers, pin_memory=True, sampler=val_sampler)

max_iou = 0.
filename = ''

for epoch in range(args.start_epoch, args.epochs):
    if args.fix_random_seed_val:
        torch.cuda.manual_seed(args.manual_seed + epoch)
        np.random.seed(args.manual_seed + epoch)
        torch.manual_seed(args.manual_seed + epoch)
        torch.cuda.manual_seed_all(args.manual_seed + epoch)
        random.seed(args.manual_seed + epoch)

    epoch_log = epoch + 1
    loss_train, mIoU_train, mAcc_train, allAcc_train = train(train_loader, model, optimizer, transformer_optimizer,
                                                             epoch, base_lrs)
    if main_process():
        writer.add_scalar('loss_train', loss_train, epoch_log)
        writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
        writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
        writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

    if args.evaluate and epoch > args.epochs // 5:
        loss_val, mIoU_val, mAcc_val, allAcc_val, class_miou = validate(val_loader, model)
        if main_process():
            writer.add_scalar('loss_val', loss_val, epoch_log)
            writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
            writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
            writer.add_scalar('class_miou_val', class_miou, epoch_log)
            writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
        if class_miou > max_iou:
            max_iou = class_miou
            if os.path.exists(filename):
                os.remove(filename)
            filename = args.save_path + '/train_epoch_' + str(epoch) + '_' + str(max_iou) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       filename)

filename = args.save_path + '/final.pth'
logger.info('Saving checkpoint to: ' + filename)
torch.save({'epoch': args.epochs, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, filename)
