# FixMatch experiments in PyTorch

Experiments with "FixMatch" on Cifar10 dataset.

Based on ["FixMatch: Simplifying Semi-Supervised Learning withConsistency and Confidence"](https://arxiv.org/abs/2001.07685)
and its official [code](https://github.com/google-research/fixmatch).

## Requirements

```bash
pip install --upgrade --pre hydra-core tensorboardX
pip install --upgrade --pre pytorch-ignite
```

## Training

```bash
python -u main_fixmatch.py model=WRN-28-2
```

This script automatically trains in multiple GPUs (`torch.nn.DistributedParallel`). 

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
tensorboard --logdir=/tmp/output-fixmatch-cifar10-hydra/
```


### Distributed Data Parallel (DDP) on multiple GPUs (Experimental)

For example, training on 2 GPUs 
```bash
python -u -m torch.distributed.launch --nproc_per_node=2 main_fixmatch.py model=WRN-28-2 distributed.backend=nccl
```

### TPU(s) on Colab (Experimental)

#### 8 TPUs on Colab

```bash
python -u main_fixmatch.py model=resnet18 distributed.backend=xla-tpu distributed.nproc_per_node=8
# or python -u main_fixmatch.py model=WRN-28-2 distributed.backend=xla-tpu distributed.nproc_per_node=8
```
