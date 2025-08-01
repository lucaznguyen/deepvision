import torch.nn as nn

class Module(nn.Module):
    """Light wrapper that exposes a count_params utility."""

    def num_params(self, trainable_only: bool = True) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
