from torchvision.models import resnet18, resnet34
from core.module import Module

class ResNet(Module):
    def __init__(self, variant: str = "resnet18", num_classes: int = 100):
        super().__init__()
        fn = resnet18 if variant == "resnet18" else resnet34
        self.net = fn(num_classes=num_classes)
    def forward(self, x): return self.net(x)
