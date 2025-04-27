import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
import utils
from functools import partial

from FSLiteNet import FSLiteNet, _cfg


def get_args_parser():
    parser = argparse.ArgumentParser('GFNet evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--arch', default='FSLiteNet', type=str, help='gfnet-xs,combined-gfnet,combined-gfnet-residual,FSLiteNet')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--data-path', default=r'D:\数据集\NWPU45', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='NWPU45', choices=['AID', 'UCM', 'NWPU45', 'INAT19', 'TG2'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model-path',
                        default=r'D:\PyTorch动手学\GFNet-master_123\GFNet-master_1\FSLiteNet-UCM-300-95.71.pth',
                        help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    return parser


def main(args):
    cudnn.benchmark = True
    dataset_val, _ = build_dataset(is_train=False, args=args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    num_classes = 45
    if args.arch == 'FSLiteNet':
        model = FSLiteNet(
            img_size=args.input_size, num_classes=num_classes,
            patch_size=2, embed_dim=384, fdf_net_depth=6, mlp_ratio=4,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            convnext_dims=[32, 64, 128, 256], convnext_depths=[2, 2, 6, 2]
        )
    else:
        raise NotImplementedError

    model_path = args.model_path
    model.default_cfg = _cfg()

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    print('## model has been successfully loaded')

    model = model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)

    criterion = torch.nn.CrossEntropyLoss().cuda()
    validate(data_loader_val, model, criterion)


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

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # 获取前五个类别及其概率
            top5_prob, top5_class = torch.topk(torch.softmax(output, dim=1), 5)

            # 容差值定义，用于判断概率是否接近
            tolerance = 0.1

            for j in range(images.size(0)):  # 遍历每个样本
                true_label = target[j].item()  # 获取真实标签
                true_prob = torch.softmax(output[j], dim=0)[true_label].item()  # 真实标签对应的概率
                print(f"Sample {j} (True Label: {true_label}, Probability: {true_prob:.4f}):")

                similar_classes = []
                for k in range(5):  # 遍历前五个类别
                    prob = top5_prob[j][k].item()
                    class_id = top5_class[j][k].item()

                    # 仅考虑概率大于0.1的类别
                    if prob > 0.1:
                        print(f"Class {class_id} with probability {prob}")

                        # 比较该类与其他类的概率是否相近
                        for m in range(k + 1, 5):
                            if top5_prob[j][m].item() > 0.1 and abs(top5_prob[j][k] - top5_prob[j][m]) < tolerance:
                                similar_classes.append((class_id, top5_class[j][m].item()))

                if similar_classes:
                    print(f"Classes with similar probabilities: {similar_classes}")

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg



if __name__ == '__main__':
    parser = argparse.ArgumentParser('GFNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
