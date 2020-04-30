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

This script automatically trains in multiple GPUs (`torch.nn.DistributedParallel`). 

### Distributed Data Parallel (DDP) on multiple GPUs (Experimental)

For example, training on 2 GPUs 
```bash
python -u -m torch.distributed.launch --nproc_per_node=2 main_fixmatch.py --params="distributed=True"
```

### TPU(s) on Colab (Experimental)

#### Installation
```bash
VERSION = "1.5"
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version $VERSION
```

#### Single TPU
```bash
python -u main_fixmatch.py --params="device='xla'"
```

#### 8 TPUs on Colab

```bash
python -u main_fixmatch.py --params="device='xla';distributed=True"
```

## TODO

* [x] Resume training from existing checkpoint:
    * [x] save/load CTA
    * [x] save ema model

* [ ] DDP: 
    * [x] Synchronize CTA across processes
    * [x] Unified GPU and TPU approach    
    * [ ] Bug: DDP performances are worse than DP on the first epochs        

* [ ] Logging to an online platform: NeptuneML or Trains or W&B

* [ ] Replace PIL augmentations with Albumentations

```python
class BlurLimitSampler:
    def __init__(self, blur, weights):
        self.blur = blur # [3, 5, 7]
        self.weights = weights # [0.1, 0.5, 0.4]    
    def get_params(self):
        return {"ksize": int(random.choice(self.blur, p=self.weights))}
        
class Blur(ImageOnlyTransform):
    def __init__(self, blur_limit, always_apply=False, p=0.5):
        super(Blur, self).__init__(always_apply, p)
        self.blur_limit = blur_limit    
        
    def apply(self, image, ksize=3, **params):
        return F.blur(image, ksize)    
    
    def get_params(self):
        if isinstance(self.blur_limit, BlurLimitSampler):
            return self.blur_limit.get_params()
        return {"ksize": int(random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2)))}    
    
    def get_transform_init_args_names(self):
        return ("blur_limit",)
```