from typing import Callable, Tuple
from torch.utils.data import DataLoader

class DataModule:
    """
    Wraps two DataLoader objects (`train_loader`, `val_loader`).
    Accepts a loader‑factory to keep interface uniform across datasets.
    """

    def __init__(self, loader_fn: Callable[..., Tuple[DataLoader, DataLoader]], **loader_kwargs):
        self.train_loader, self.val_loader = loader_fn(**loader_kwargs)
