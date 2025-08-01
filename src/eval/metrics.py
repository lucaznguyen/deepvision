import torch, numpy as np
from sklearn.metrics import confusion_matrix   # add scikit‑learn to requirements

@torch.no_grad()
def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
    """Compute Top‑k accuracy for the specified values of k."""
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred   = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        res.append(correct[:k].reshape(-1).float().sum(0).item())
    return [r / target.size(0) for r in res]

def confusion_matrix_np(preds, labels, num_classes):
    return confusion_matrix(labels, preds, labels=np.arange(num_classes))
