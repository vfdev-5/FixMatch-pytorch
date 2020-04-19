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

BUGS:
* [x] Memory leak    

* [x] Resume training from existing checkpoint:
    * [x] save/load CTA
    * [x] save ema model

* [ ] DDP: Synchronize CTA across processes

* [ ] Logging to online platform: NeptuneML or Trains or W&B

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