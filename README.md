# FixMatch experiments in PyTorch and Ignite

Experiments with "FixMatch" on Cifar10 dataset.

Based on ["FixMatch: Simplifying Semi-Supervised Learning withConsistency and Confidence"](https://arxiv.org/abs/2001.07685)
and its official [code](https://github.com/google-research/fixmatch).

**Data-augmentations policy is CTA**

## Requirements

```bash
pip install --upgrade --pre hydra-core tensorboardX
pip install --upgrade git+https://github.com/pytorch/ignite
# pip install --upgrade --pre pytorch-ignite
```

Optionally, we can install `wandb` for online experiments tracking.
```bash
pip install wandb
```

We can also opt to replace `Pillow` by `Pillow-SIMD` to accelerate image processing part:
```bash
pip uninstall -y pillow && CC="cc -mavx2" pip install --no-cache-dir --force-reinstall pillow-simd
```

## Training

```bash
python -u main_fixmatch.py model=WRN-28-2
```

- Default output folder: "/tmp/output-fixmatch-cifar10". 
- For complete list of options: `python -u main_fixmatch.py --help` 

This script automatically trains on multiple GPUs (`torch.nn.DistributedParallel`). 

If it is needed to specify input/output folder :  
```
python -u main_fixmatch.py dataflow.data_path=/data/cifar10/ hydra.run.dir=/output-fixmatch model=WRN-28-2
```

To use wandb logger, we need login and run with `online_exp_tracking.wandb=true`:
```bash
wandb login <token>
python -u main_fixmatch.py model=WRN-28-2 online_exp_tracking.wandb=true
```

To see other options:
```bash
python -u main_fixmatch.py --help
```

### Training curves visualization

By default, we use Tensorboard to log training curves

```bash
tensorboard --logdir=/tmp/output-fixmatch-cifar10/
```

Online logging on W&B: https://app.wandb.ai/vfdev-5/fixmatch-pytorch


### Distributed Data Parallel (DDP) on multiple GPUs (Experimental)

For example, training on 2 GPUs 
```bash
python -u -m torch.distributed.launch --nproc_per_node=2 main_fixmatch.py model=WRN-28-2 distributed.backend=nccl
```

### TPU(s) on Colab (Experimental)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ZoWz1-a3bpj1xMxpM2K2qQ4Y9xvtdGWO) 
For example, training on 8 TPUs in distributed mode: 
```bash
python -u main_fixmatch.py model=resnet18 distributed.backend=xla-tpu distributed.nproc_per_node=8
# or python -u main_fixmatch.py model=WRN-28-2 distributed.backend=xla-tpu distributed.nproc_per_node=8
```


## Experimentations

### Faster Resnet-18 training

- reduce the number of epochs
- reduce the number of CTA

```bash
python main_fixmatch.py distributed.backend=nccl online_exp_tracking.wandb=true solver.num_epochs=100 \
    ssl.confidence_threshold=0.7 ema_decay=0.9 ssl.cta_update_every=15
``` 
