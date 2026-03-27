import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
from collections import OrderedDict
import json
import math
import os
import sys
import time
from torch.utils.tensorboard import SummaryWriter


import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from data import ShortVideoDataset, SYNLVideoDataset
from models import CLIPModel
import losses
import utils

from transformers import AutoProcessor
from torch.utils.data.distributed import DistributedSampler



def get_args_parser():
    parser = argparse.ArgumentParser(description='OneClip training', add_help=False)

    parser.add_argument('--output-dir', default='./output', type=str, help='output dir')

    parser.add_argument('--model', default='CLIP_VITB32', type=str)
    parser.add_argument('--resume', default='', type=str, help='path to resume from')

    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--warmup-epochs', default=0, type=int)
    parser.add_argument('--start-epoch', default=0, type=int)
    parser.add_argument('--batch-size', default=64, type=int,
                        help='number of samples per-gpu')
    parser.add_argument('--lr', default=3e-3, type=float)
    parser.add_argument('--lr-start', default=1e-6, type=float,
                        help='initial warmup lr')
    parser.add_argument('--lr-end', default=1e-5, type=float,
                        help='minimum final lr')
    parser.add_argument('--update-freq', default=1, type=int,
                        help='optimizer update frequency (i.e. gradient accumulation steps)')
    parser.add_argument('--wd', default=0.1, type=float)
    parser.add_argument('--betas', default=(0.9, 0.98), nargs=2, type=float)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable mixed-precision training (requires more memory and compute)')

    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

    parser.add_argument('--clip_frames', default=16, type=int)
    parser.add_argument('--neg_frames', default=100, type=int)
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers per process')
    parser.add_argument('--temporal_method', default=None, type=str)

    return parser


best_acc1 = 0


def main(args):
    utils.init_distributed_mode(args)

    global best_acc1

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # create model
    print("=> creating model: {}".format(args.model))
    model = CLIPModel.from_pretrained(args.model)
    
    
    clip_processor = AutoProcessor.from_pretrained(args.model)
    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], bucket_cap_mb=200) 

    # define loss function (criterion) and optimizer
    if args.temporal_method == 'longvideo':
        criterion = losses.INTRACLIPLoss().cuda(args.gpu)
    else:
        criterion = losses.INTERCLIPLoss().cuda(args.gpu)

    p_wd, p_non_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights
        if p.ndim < 2 or 'bias' in n or 'ln' in n or 'bn' in n:
            p_non_wd.append(p)
        else:
            p_wd.append(p)

    optim_params = [{"params": p_wd, "weight_decay": args.wd},
                    {"params": p_non_wd, "weight_decay": 0}]

    optimizer = torch.optim.AdamW(optim_params, lr=args.lr, betas=args.betas,
                                    eps=args.eps, weight_decay=args.wd)
    
    scaler = amp.GradScaler(enabled=not args.disable_amp)

    # optionally resume from a checkpoint (takes precedence over autoresume)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading resume checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
            args.start_epoch = epoch
            result = model.load_state_dict(checkpoint['state_dict'], strict=False)
            print(result)
            optimizer.load_state_dict(checkpoint['optimizer']) if 'optimizer' in checkpoint else ()
            scaler.load_state_dict(checkpoint['scaler']) if 'scaler' in checkpoint else ()
            best_acc1 = checkpoint['best_acc1']
            print("=> loaded resume checkpoint '{}' (epoch {})"
                  .format(args.resume, epoch))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        # auto-resume from latest checkpoint in output directory
        latest = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.isfile(latest):
            print("=> loading latest checkpoint '{}'".format(latest))
            latest_checkpoint = torch.load(latest, map_location='cpu')
            args.start_epoch = latest_checkpoint['epoch']
            model.load_state_dict(latest_checkpoint['state_dict'])
            optimizer.load_state_dict(latest_checkpoint['optimizer'])
            scaler.load_state_dict(latest_checkpoint['scaler'])
            best_acc1 = latest_checkpoint['best_acc1']
            best_acc1 = 0
            print("=> loaded latest checkpoint '{}' (epoch {})"
                  .format(latest, latest_checkpoint['epoch']))

    cudnn.benchmark = True

    # Data loading code 
    print("=> creating dataset")
    if args.temporal_method == 'longvideo':
        train_dataset = SYNLVideoDataset(
            anno_path = '/path/to/annotations',
            video_path = "/path/to/video",
            processor=clip_processor, 
            clip_frames=args.clip_frames, 
            neg_frames=args.neg_frames,
        )
    else:
        train_dataset = ShortVideoDataset(
            nextqa_anno_path='/path/to/train.csv',
            nextqa_video_path='/path/to/video',
            qaego4d_anno_path='/path/to/annotations.train.json', 
            qaego4d_video_path='/path/to/video',
            processor=clip_processor, 
            clip_frames=args.clip_frames, 
        )


    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=False)

    train_loader.num_samples = len(train_dataset) 
    train_loader.num_batches = len(train_loader)

    lr_schedule = None

    if utils.is_main_process() and args.output_dir is not None:
        args.log_dir = os.path.join(args.output_dir, 'tb_logs')
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    print(args)

    print("=> beginning training")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed and train_sampler is not None and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(epoch)

        # train for one epoch
        train_stats = train(train_loader, log_writer, model, criterion, optimizer, scaler, epoch, lr_schedule, args, clip_processor)
        
        best_acc1 = 0
        is_best = False
        print("=> saving checkpoint")
        utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'best_acc1': best_acc1,
                'args': args,
            }, is_best, args.output_dir, epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    'epoch': epoch}

        # log test stats to log_writer (tensorboard)
        if log_writer is not None:
            for k, v in log_stats.items():
                if k.startswith('test'):
                    log_writer.add_scalar(k, v, epoch)

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train(train_loader, log_writer, model, criterion, optimizer, scaler, epoch, lr_schedule, args, processor):
    batch_time = AverageMeter('Time', ':6.2f')
    data_time = AverageMeter('Data', ':6.2f')
    mem = AverageMeter('Mem (GB)', ':6.1f')
    metric_names = ['loss']

    loader_len = train_loader.num_batches
    iters_per_epoch = loader_len // args.update_freq
    metrics = OrderedDict([(name, AverageMeter(name, ':.2f')) for name in metric_names])
    progress = ProgressMeter(
        iters_per_epoch,
        [batch_time, data_time, mem, *metrics.values()],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()


    end = time.time()
    print(f'=> training epoch {epoch}')
    for data_iter, inputs in enumerate(train_loader):
        optim_iter = data_iter // args.update_freq

        # measure data loading time
        data_time.update(time.time() - end)

        # update weight decay and learning rate according to their schedule
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        if lr_schedule is not None:
            for k, param_group in enumerate(optimizer.param_groups):
                param_group['lr'] = lr_schedule[it]

        frames = inputs['frames'].cuda(args.gpu, non_blocking=True)
        bsz, n_frames, c, w, h = frames.shape
        frames = frames.view(-1, c, w, h)
        questions = inputs['questions'].cuda(args.gpu, non_blocking=True)
        attention_mask = inputs['attention_mask'].cuda(args.gpu, non_blocking=True)
        if args.temporal_method == 'longvideo':
            labels = inputs['labels'].cuda(args.gpu, non_blocking=True)


        # compute output
        with amp.autocast(enabled=not args.disable_amp):
            outputs = model(pixel_values=frames, input_ids=questions, attention_mask=attention_mask)
            if args.temporal_method == 'longvideo':
                outputs['labels'] = labels
            loss_dict = criterion(outputs, args)
            loss = loss_dict['loss'] / args.update_freq
        
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        scaler.scale(loss).backward()

        if (data_iter + 1) % args.update_freq != 0:
            continue

        # compute gradient and do SGD step
        scaler.step(optimizer)
        scaler.update()
        model.zero_grad(set_to_none=True)

        # clamp logit scale to [0, 100]
        if hasattr(model, 'module'):
            model.module.logit_scale.data.clamp_(0, 4.6052)
            logit_scale = model.module.logit_scale.exp().item()
        else:
            model.logit_scale.data.clamp_(0, 4.6052)
            logit_scale = model.logit_scale.exp().item()


        metrics['loss'].update(loss.item(), args.batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        mem.update(torch.cuda.max_memory_allocated() // 1e9)

        # save to log_writer (tensorboard)
        if log_writer is not None:
            log_writer.add_scalar('loss', loss.item(), it)
            log_writer.add_scalar('scaler', scaler.get_scale(), it)
            log_writer.add_scalar('logit', logit_scale, it)
            log_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], it)

        if optim_iter % args.print_freq == 0:
            progress.display(optim_iter)

    progress.synchronize()
    return {**{k: v.avg for k, v in metrics.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'logit_scale': logit_scale}


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def synchronize(self):
        if not utils.is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.sum, self.count], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.sum = int(t[0])
        self.count = t[1]
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def synchronize(self):
        for meter in self.meters:
            meter.synchronize()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



if __name__ == '__main__':
    parser = argparse.ArgumentParser('OneClip training', parents=[get_args_parser()])
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)