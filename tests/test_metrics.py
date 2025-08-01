import torch
from src.eval.metrics import accuracy

def test_topk_accuracy():
    logits = torch.tensor([[0.1, 2.0, 0.3],
                           [4.0, 0.2, 0.1]])
    targets = torch.tensor([1, 0])
    top1, top2 = accuracy(logits, targets, topk=(1,2))
    assert abs(top1 - 1.0) < 1e-4
    assert abs(top2 - 1.0) < 1e-4
