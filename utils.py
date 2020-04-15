from functools import partial
import json
import random
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import Subset, Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets.cifar import CIFAR10

from ignite.utils import convert_tensor

from ctaugment import OPS, CTAugment, OP


weak_transforms = T.Compose([
    T.Pad(4),
    T.RandomCrop(32),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
])


test_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25))
])


cutout_image_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.25, 0.25, 0.25)),
    T.RandomErasing(scale=(0.02, 0.15))
])


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


class TransformedDataset(Dataset):

    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, i):
        dp = self.dataset[i]
        return self.transforms(dp)

    def __len__(self):
        return len(self.dataset)


def get_supervised_trainset(root, num_train_samples_per_class=25, download=True):
    num_classes = 10
    full_train_dataset = CIFAR10(root, train=True, download=download)

    supervised_train_indices = []
    counter = [0] * num_classes

    indices = list(range(len(full_train_dataset)))
    random_indices = np.random.permutation(indices)

    for i in random_indices:
        dp = full_train_dataset[i]
        if len(supervised_train_indices) >= num_classes * num_train_samples_per_class:
            break
        if counter[dp[1]] < num_train_samples_per_class:
            counter[dp[1]] += 1
            supervised_train_indices.append(i)

    return Subset(full_train_dataset, supervised_train_indices)


def get_supervised_trainset_0_250(root, download=False):
    full_train_dataset = CIFAR10(root, train=True, download=download)

    supervised_train_indices = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
        30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 
        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 
        58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 
        72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 
        86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 
        111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 
        122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 
        133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 
        144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 
        155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 
        166, 167, 169, 170, 171, 172, 173, 174, 175, 177, 178, 
        179, 180, 181, 182, 183, 185, 186, 187, 188, 189, 190, 
        191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 202, 
        203, 204, 205, 207, 209, 210, 211, 213, 215, 216, 217,
        218, 220, 221, 222, 223, 224, 228, 229, 230, 231, 233, 
        237, 239, 240, 241, 244, 246, 247, 252, 254, 256, 259, 
        260, 263, 264, 268, 271, 272, 276, 277, 279, 280, 281, 
        284, 285, 290, 293, 296, 308, 317
    ]
    return Subset(full_train_dataset, supervised_train_indices)


def get_supervised_train_loader(supervised_train_dataset, transforms=weak_transforms, **dataloader_kwargs):

    dataloader_kwargs['pin_memory'] = True
    dataloader_kwargs['drop_last'] = True
    dataloader_kwargs['shuffle'] = dataloader_kwargs.get("sampler", None) is None

    supervised_train_loader = DataLoader(
        TransformedDataset(
            supervised_train_dataset,
            transforms=lambda d: {"image": transforms(d[0]), "target": d[1]}
        ),
        **dataloader_kwargs
    )
    return supervised_train_loader


def get_test_loader(root, transforms=test_transforms, **dataloader_kwargs):

    full_test_dataset = CIFAR10(root, train=False, download=False)

    dataloader_kwargs['pin_memory'] = True
    dataloader_kwargs['drop_last'] = False
    dataloader_kwargs['shuffle'] = False

    test_loader = DataLoader(
        TransformedDataset(
            full_test_dataset,
            transforms=lambda dp: {"image": transforms(dp[0]), "target": dp[1]}
        ),
        **dataloader_kwargs
    )
    return test_loader


class StorableCTAugment(CTAugment):    

    def load_state_dict(self, state):
        for k in ["decay", "depth", "th", "rates"]:
            assert k in state, "{} not in {}".format(k, state.keys())
            setattr(self, k, state[k])

    def state_dict(self):
        return OrderedDict([(k, getattr(self, k)) for k in ["decay", "depth", "th", "rates"]])


def get_default_cta():
    return StorableCTAugment()


def cta_apply(pil_img, ops):
    if ops is None:
        return pil_img
    for op, args in ops:
        pil_img = OPS[op].f(pil_img, *args)
    return pil_img


def deserialize(policy_str):
    return [OP(f=x[0], bins=x[1]) for x in json.loads(policy_str)]


def cta_image_transforms(pil_img, cta, transform=cutout_image_transforms):
    policy = cta.policy(probe=False)
    pil_img = cta_apply(pil_img, policy)
    return transform(pil_img)


def cta_probe_transforms(dp, cta, image_transforms=cutout_image_transforms):
    policy = cta.policy(probe=True)
    probe = cta_apply(dp[0], policy)
    probe = image_transforms(probe)
    return {
        "image": probe,
        "target": dp[1],
        "policy": json.dumps(policy)
    }


def get_cta_probe_loader(supervised_train_dataset, cta, **dataloader_kwargs):

    dataloader_kwargs['pin_memory'] = True
    dataloader_kwargs['drop_last'] = False
    dataloader_kwargs['shuffle'] = dataloader_kwargs.get("sampler", None) is None

    cta_probe_loader = DataLoader(
        TransformedDataset(
            supervised_train_dataset,
            transforms=partial(cta_probe_transforms, cta=cta)
        ),
        **dataloader_kwargs
    )

    return cta_probe_loader


def get_unsupervised_train_loader(raw_dataset, transforms_weak, transforms_strong, **dataloader_kwargs):
    
    unsupervised_train_dataset = TransformedDataset(
        raw_dataset,
        transforms=lambda dp: {"image": transforms_weak(dp[0]), "strong_aug": transforms_strong(dp[0])}
    )

    dataloader_kwargs['pin_memory'] = True
    dataloader_kwargs['drop_last'] = True
    dataloader_kwargs['shuffle'] = dataloader_kwargs.get("sampler", None) is None

    unsupervised_train_loader = DataLoader(
        unsupervised_train_dataset,
        **dataloader_kwargs
    )
    return unsupervised_train_loader


def sup_prepare_batch(batch, device, non_blocking):
    x = convert_tensor(batch["image"], device, non_blocking)
    y = convert_tensor(batch["target"], device, non_blocking)
    return x, y


def cycle(dataloader):
    while True:
        for b in dataloader:
            yield b


def stats(cta):
    return '\n'.join('%-16s    %s' % (k, ' / '.join(' '.join('%.2f' % x for x in cta.rate_to_p(rate))
                                                    for rate in cta.rates[k]))
                     for k in sorted(OPS.keys()))


def interleave(x, batch, inverse=False):
    # def interleave(x, batch):
    #     s = x.get_shape().as_list()
    #     return tf.reshape(tf.transpose(tf.reshape(x, [-1, batch] + s[1:]), [1, 0] + list(range(2, 1+len(s)))), [-1] + s[1:])
    shape = x.shape
    axes = [batch, -1] if inverse else [-1, batch]
    return x.reshape(*axes, *shape[1:]).transpose(0, 1).reshape(-1, *shape[1:])


def deinterleave(x, batch):
    return interleave(x, batch, inverse=True)


def setup_ema(ema_model, ref_model):
    ema_model.load_state_dict(ref_model.state_dict())
    for param in ema_model.parameters():
        param.detach_()
    # set EMA model's BN buffers as base model BN buffers:
    for m1, m2 in zip(ref_model.modules(), ema_model.modules()):
        if isinstance(m1, nn.BatchNorm2d) and isinstance(m2, nn.BatchNorm2d):
            m2.running_mean = m1.running_mean
            m2.running_var = m1.running_var
