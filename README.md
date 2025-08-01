# DeepVision – A Modular PyTorch Code-Base for Vision & Sequence Models

> **Status** | Main branch: ![Build](https://img.shields.io/badge/CI-passing-brightgreen) ![License](https://img.shields.io/badge/License-Apache--2.0-blue)

**DeepVision** is a modular, teaching-oriented yet production-ready PyTorch framework evolving from simple CNNs (MNIST/CIFAR) to advanced architectures (RNNs, Transformers, distributed training), organized in **four structured milestones**.

---

## 📚 Table of Contents
1. [Features](#features)
2. [Project Layout](#project-layout)
3. [Requirements & Installation](#requirements--installation)
4. [Quickstart](#quickstart)
5. [Configuration with Hydra](#configuration-with-hydra)
6. [Adding New Components](#adding-new-components)
7. [Benchmarks](#benchmarks)
8. [Testing & CI](#testing--ci)
9. [Contributing](#contributing)
10. [License](#license)

---

## ✨ Features
- **Datasets:** MNIST, CIFAR-10/100, Tiny-ImageNet (with HDF5 caching).
- **Models:** SimpleCNN, VGG-16, ResNet-18/34, WideResNet-28-10, ViT-Tiny.
- **Training Engine:** Single-GPU, AMP, Cosine LR schedule, easily extensible to DDP.
- **Configuration:** Hydra-based hierarchical configuration system.
- **Evaluation:** Top-k accuracy, confusion matrix, parameter count/FLOPs estimates.
- **DevOps:** Integrated CI via GitHub Actions, pre-commit hooks (`ruff`, `black`).
- **Pedagogy:** Weekly exercises, structured progress reports.

---

## 📂 Project Layout
deepvision/
├── README.md
├── pyproject.toml
├── configs/
├── src/
│ ├── cli/
│ ├── core/
│ ├── data/
│ ├── models/
│ ├── eval/
│ └── utils/
├── tests/
└── reports/

---

## 🚀 Requirements & Installation
- Python ≥ 3.9
- PyTorch ≥ 2.2 (CUDA recommended)
- Dependencies: `torchvision`, `hydra-core`, `omegaconf`, `scikit-learn`, `timm`, `tqdm`, `tensorboard`, `pytest`

```bash
git clone https://github.com/<user>/deepvision.git
cd deepvision

python -m venv .venv && source .venv/bin/activate
pip install -e .

---

## 🖥️ Quickstart

### Train SimpleCNN on CIFAR-10:

```bash
python -m src.cli.train dataset=cifar10 model=simple_cnn run.device=gpu
```

### Train ResNet-18 on CIFAR-100:

```bash
python -m src.cli.train dataset=cifar100 model=resnet18 run.batch_size=256 run.lr=0.05 run.epochs=120
```

### Train ResNet-34 on Tiny-ImageNet:

```bash
python -m src.cli.train dataset=tiny_imagenet model=resnet34
```

---

## ⚙️ Configuration with Hydra

Defined in `configs/config.yaml`:

```yaml
defaults:
  - _self_
  - dataset: cifar100
  - model: resnet18
  - override hydra/job_logging: disabled
  - override hydra/hydra_logging: disabled
```

CLI Overrides:

```bash
python -m src.cli.train dataset=tiny_imagenet model=vit_tiny run.device=cpu
```

---

## 🛠️ Adding New Components

### ➡️ Dataset

* **Loader:** Create in `src/data/mydataset.py`
* **Register:** Add to `src/cli/train.py`
* **Configure:** Create YAML in `configs/dataset/mydataset.yaml`

### ➡️ Model

* **Wrapper:** Create in `src/models/mymodel.py`
* **Register:** Add to model registry in `src/cli/train.py`
* **Configure:** YAML in `configs/model/mymodel.yaml`

---

## 📈 Benchmarks

| Model               | Dataset   | Epochs | Top-1 Acc | Params | GPU Time/Epoch\* |
| ------------------- | --------- | ------ | --------- | ------ | ---------------- |
| SimpleCNN           | CIFAR-10  | 50     | 71%       | 0.9M   | 12s              |
| ResNet-34           | Tiny-IN   | 90     | 46%       | 21M    | 45s              |
| ViT-Tiny (pretrain) | CIFAR-100 | 100    | 74%       | 5.7M   | 18s              |

\*Benchmarked on RTX 3080 (AMP enabled, batch size=128).

---

## 🧪 Testing & CI

```bash
pytest -q
ruff . --fix
pre-commit run -a
```

* GitHub Actions automatically run tests and linting on every commit.

---

## 🤝 Contributing

* Fork, branch, PR workflow.
* Adhere to existing style (`ruff`, `black`).
* Provide unit tests for new features.
* Document clearly with meaningful comments.

---

## 📜 License

* **Apache-2.0 License** (see [LICENSE](./LICENSE)).
* If using academically, please cite the repository URL.

© 2025 LucazNguyen