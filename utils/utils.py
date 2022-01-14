from __future__ import print_function
import os
import shutil
import torch
import numpy as np


def save_checkpoint(state, is_best, root, filename):
    torch.save(state, os.path.join(root, filename))
    if is_best:
        shutil.copyfile(os.path.join(root, filename), os.path.join(root, 'best_' + filename))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


def evaluate_seg(pred, gt):
    pred_binary = (pred >= 0.5).float().cuda()
    pred_binary_inverse = (pred_binary == 0).float().cuda()

    gt_binary = (gt >= 0.5).float().cuda()
    gt_binary_inverse = (gt_binary == 0).float().cuda()

    MAE = torch.abs(pred_binary - gt_binary).mean().cuda(0)
    TP = pred_binary.mul(gt_binary).sum().cuda(0)
    FP = pred_binary.mul(gt_binary_inverse).sum().cuda(0)
    TN = pred_binary_inverse.mul(gt_binary_inverse).sum().cuda(0)
    FN = pred_binary_inverse.mul(gt_binary).sum().cuda(0)

    if TP.item() == 0:
        TP = torch.Tensor([1]).cuda(0)
    # recall
    Recall = TP / (TP + FN)
    # Precision or positive predictive value
    Precision = TP / (TP + FP)
    # F1 score = Dice
    Dice = 2 * Precision * Recall / (Precision + Recall)
    # Overall accuracy
    Accuracy = (TP + TN) / (TP + FP + FN + TN)
    # IoU for poly
    IoU_polyp = TP / (TP + FP + FN)

    return MAE.data.cpu().numpy().squeeze(), \
           Recall.data.cpu().numpy().squeeze(), \
           Precision.data.cpu().numpy().squeeze(), \
           Accuracy.data.cpu().numpy().squeeze(), \
           Dice.data.cpu().numpy().squeeze(), \
           IoU_polyp.data.cpu().numpy().squeeze()


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch, consistency=0.1, consistency_rampup=5, start_epoch=0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    if epoch <= start_epoch:
        return 0.0
    return consistency * sigmoid_rampup(epoch - start_epoch, consistency_rampup - start_epoch)
