"""Lovasz-Softmax and Jaccard hinge loss in PyTorch.
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from itertools import filterfalse
import torch
import torch.nn.functional as F
import numpy as np


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        ious.append(float(intersection) / float(union) if union else EMPTY)
    return 100 * mean(ious)


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou_list = []
        for i in range(C):
            if i == ignore:
                continue
            intersection = ((label == i) & (pred == i)).sum()
            union = ((label == i) | ((pred == i) & (label != ignore))).sum()
            iou_list.append(float(intersection) / float(union) if union else EMPTY)
        ious.append(iou_list)
    return 100 * np.array([mean(iou_col) for iou_col in zip(*ious)])


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    if per_image:
        return mean(
            lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
            for log, lab in zip(logits, labels)
        )
    return lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))


def lovasz_hinge_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = 1. - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm.data]
    grad = lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def flatten_binary_scores(scores, labels, ignore=None):
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    return scores[valid], labels[valid]


class StableBCELoss(torch.nn.Module):
    def forward(self, input, target):
        neg_abs = -input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    return StableBCELoss()(logits, labels.float())


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    if per_image:
        return mean(
            lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
            for prob, lab in zip(probas, labels)
        )
    return lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)


def lovasz_softmax_flat(probas, labels, classes='present'):
    if probas.numel() == 0:
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if classes == 'present' and fg.sum() == 0:
            continue
        class_pred = probas[:, 0] if C == 1 else probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm.data]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid.nonzero().squeeze()], labels[valid]


def xloss(logits, labels, ignore=None):
    return F.cross_entropy(logits, labels, ignore_index=255)


def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    l = iter(l)
    if ignore_nan:
        l = filterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    return acc if n == 1 else acc / n
