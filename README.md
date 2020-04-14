# FixMatch experiments in PyTorch

Experiments with "FixMatch" method on Cifar10 dataset.

Based on ["FixMatch: Simplifying Semi-Supervised Learning withConsistency and Confidence"](https://arxiv.org/abs/2001.07685)

## Requirements

```bash
pip install --upgrade --pre pytorch-ignite
```

## Training

### Download dataset (Optional)
```bash
python -c "import torchvision as t; t.datasets.cifar.CIFAR10('/path/to/cifar10', download=True)"
```

```bash
python -u main_fixmatch.py --params "data_path=/path/to/cifar10"
```

## Training with MLflow

