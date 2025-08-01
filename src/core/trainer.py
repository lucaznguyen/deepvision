import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loaders,
        device="cuda",
        lr=0.1,
        epochs=50,
        mixprecision=True
    ):
        self.model = model.to(device)
        self.train_loader, self.val_loader = loaders
        self.device = device
        self.epochs = epochs
        self.scaler = torch.cuda.amp.GradScaler(enabled=mixprecision)
        self.optim = SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        self.sched = CosineAnnealingLR(self.optim, T_max=epochs)

    def _loop(self, loader, train=True):
        phase = "Train" if train else "Eval "
        self.model.train(train)
        total, correct, loss_sum = 0, 0, 0.0
        criterion = nn.CrossEntropyLoss()
        pbar = tqdm(loader, desc=phase, leave=False)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            with torch.cuda.amp.autocast(self.scaler.is_enabled()):
                logits = self.model(x)
                loss = criterion(logits, y)
            if train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad()

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
            loss_sum += loss.item() * y.size(0)
            pbar.set_postfix(acc=f"{100*correct/total:5.2f}%", loss=f"{loss_sum/total:6.4f}")

        return correct / total, loss_sum / total

    def fit(self):
        for epoch in range(1, self.epochs + 1):
            train_acc, _ = self._loop(self.train_loader, train=True)
            val_acc,   _ = self._loop(self.val_loader,   train=False)
            self.sched.step()
            print(f"Epoch {epoch:03d}: train acc={train_acc:.4f}  val acc={val_acc:.4f}")
