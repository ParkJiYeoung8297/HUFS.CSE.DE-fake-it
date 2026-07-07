import numpy as np
import torch
from sklearn.metrics import auc, roc_curve


class AverageMeter:
    """Computes and stores the average and current value."""

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


def calculate_accuracy(outputs, targets):
    batch_size = targets.size(0)
    _, pred = outputs.topk(1, 1, True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))
    return 100 * correct.float().sum().item() / batch_size


def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    return fpr[eer_idx], thresholds[eer_idx]


def compute_pauc(y_true, y_scores, fpr_limit=0.1):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    mask = fpr <= fpr_limit
    return auc(fpr[mask], tpr[mask]) / fpr_limit


def fake_positive_scores(video_real_scores, labels):
    video_fake_scores = [1 - score for score in video_real_scores]
    true_bin = [1 if label == "FAKE" else 0 for label in labels]
    return torch.tensor(true_bin), torch.tensor(video_fake_scores)

