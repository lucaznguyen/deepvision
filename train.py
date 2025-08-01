import hydra, torch
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from utils.registry import Registry

DATASETS = Registry()
MODELS   = Registry()

# Register Week‑2 datasets
from data.cifar100       import cifar100_loaders     ; DATASETS["cifar100"]       = cifar100_loaders
from data.tiny_imagenet  import tiny_imagenet_loaders; DATASETS["tiny_imagenet"]  = tiny_imagenet_loaders
# Retain Week‑1 loaders
from data.mnist          import mnist_loaders        ; DATASETS["mnist"]          = mnist_loaders
from data.cifar10        import cifar10_loaders      ; DATASETS["cifar10"]        = cifar10_loaders

# Register Week‑2 models
from models.resnet       import ResNet               ; MODELS["resnet18"]         = lambda num_classes: ResNet("resnet18", num_classes)
from models.resnet       import ResNet               ; MODELS["resnet34"]         = lambda num_classes: ResNet("resnet34", num_classes)
# from models.vgg          import VGG16                ; MODELS["vgg16"]            = VGG16
# from models.wide_resnet  import WideResNet28x10      ; MODELS["wide_resnet"]      = WideResNet28x10
# from models.vit          import ViTTiny              ; MODELS["vit_tiny"]         = ViTTiny

# Retain Week‑1 SimpleCNN
from models.simple_cnn   import SimpleCNN            ; MODELS["simple_cnn"]       = lambda num_classes: SimpleCNN(in_channels=3, num_classes=num_classes)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def _run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    loaders = DATASETS[cfg.dataset.name](batch_size=cfg.run.batch_size, root=cfg.dataset.root)
    model   = MODELS[cfg.model.name](num_classes=cfg.dataset.num_classes)

    from core.trainer import Trainer
    trainer = Trainer(model, loaders, device=cfg.run.device, lr=cfg.run.lr, epochs=cfg.run.epochs)
    trainer.fit()

if __name__ == "__main__":
    _run()
