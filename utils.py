from functools import partial
import json
import random

import numpy as np

import torch

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


def get_supervised_trainset(root, num_train_samples_per_class=25, download=False):
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


def get_supervised_train_loader(supervised_train_dataset, transforms=weak_transforms, **dataloader_kwargs):

    dataloader_kwargs['pin_memory'] = True
    dataloader_kwargs['drop_last'] = False
    dataloader_kwargs['shuffle'] = True

    supervised_train_loader = DataLoader(
        TransformedDataset(
            supervised_train_dataset,
            transforms=lambda d: {"image": transforms(d[0]), "target": d[1]}
        ),
        **dataloader_kwargs
    )
    return supervised_train_loader


def get_test_loader(root, transforms=test_transforms, **dataloader_kwargs):

    full_test_dataset = CIFAR10(root, train=False, download=True)

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


def get_default_cta():
    return CTAugment()


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


def get_cta_probe_train_loader(supervised_train_dataset, cta, **dataloader_kwargs):

    dataloader_kwargs['pin_memory'] = True
    dataloader_kwargs['drop_last'] = False
    dataloader_kwargs['shuffle'] = True

    cta_probe_loader = DataLoader(
        TransformedDataset(
            supervised_train_dataset,
            transforms=partial(cta_probe_transforms, cta=cta)
        ),
        **dataloader_kwargs
    )

    return cta_probe_loader


def get_unsupervised_train_loader(root, transforms_weak, transforms_strong, **dataloader_kwargs):

    full_train_dataset = CIFAR10(root, train=True, download=False)
    unsupervised_train_dataset = TransformedDataset(
        full_train_dataset,
        transforms=lambda dp: {"image": transforms_weak(dp[0]), "strong_aug": transforms_strong(dp[0])}
    )

    dataloader_kwargs['pin_memory'] = True
    dataloader_kwargs['drop_last'] = True
    dataloader_kwargs['shuffle'] = True

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
