# FixMatch experiments in PyTorch

Experiments with "FixMatch" on Cifar10 dataset.

Based on ["FixMatch: Simplifying Semi-Supervised Learning withConsistency and Confidence"](https://arxiv.org/abs/2001.07685)

## Requirements

```bash
pip install --upgrade --pre pytorch-ignite
```

## Training

```bash
python -u main_fixmatch.py
# or python -u main_fixmatch.py --params "data_path=/path/to/cifar10"
```

## TODO

* [ ] Resume training from existing checkpoint:
    * [x] save/load CTA
    * [x] save ema model
* [ ] Logging to online platform: NeptuneML or Trains or W&B
* [ ] DDP: Synchronize CTA across processes