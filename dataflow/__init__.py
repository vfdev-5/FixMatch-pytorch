from functools import partial

from torch.utils.data import Dataset

from ignite.utils import convert_tensor


class TransformedDataset(Dataset):
    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __getitem__(self, i):
        dp = self.dataset[i]
        return self.transforms(dp)

    def __len__(self):
        return len(self.dataset)


def sup_prepare_batch(batch, device, non_blocking):
    x = convert_tensor(batch["image"], device, non_blocking)
    y = convert_tensor(batch["target"], device, non_blocking)
    return x, y


def cycle(dataloader):
    while True:
        for b in dataloader:
            yield b


def get_supervised_train_loader(
    dataset_name, root, num_train_samples_per_class, download=True, **dataloader_kwargs
):
    if dataset_name == "cifar10":
        from dataflow.cifar10 import (
            get_supervised_trainset,
            get_supervised_train_loader,
            weak_transforms,
        )

        train_dataset = get_supervised_trainset(
            root,
            num_train_samples_per_class=num_train_samples_per_class,
            download=download,
        )

        return get_supervised_train_loader(train_dataset, **dataloader_kwargs)

    else:
        raise ValueError("Unhandled dataset: {}".format(dataset_name))


def get_test_loader(dataset_name, root, download=True, **dataloader_kwargs):
    if dataset_name == "cifar10":
        from dataflow.cifar10 import get_test_loader

        return get_test_loader(root=root, download=download, **dataloader_kwargs)

    else:
        raise ValueError("Unhandled dataset: {}".format(dataset_name))


def get_unsupervised_train_loader(
    dataset_name, root, cta, download=True, **dataloader_kwargs
):
    if dataset_name == "cifar10":
        from dataflow import cifar10

        full_train_dataset = cifar10.get_supervised_trainset(
            root, num_train_samples_per_class=None, download=download
        )

        strong_transforms = partial(cifar10.cta_image_transforms, cta=cta)

        return cifar10.get_unsupervised_train_loader(
            full_train_dataset,
            transforms_weak=cifar10.weak_transforms,
            transforms_strong=strong_transforms,
            **dataloader_kwargs
        )

    else:
        raise ValueError("Unhandled dataset: {}".format(dataset_name))


def get_cta_probe_loader(
    dataset_name,
    root,
    num_train_samples_per_class,
    cta,
    download=True,
    **dataloader_kwargs
):
    if dataset_name == "cifar10":
        from dataflow.cifar10 import get_supervised_trainset, get_cta_probe_loader

        train_dataset = get_supervised_trainset(
            root,
            num_train_samples_per_class=num_train_samples_per_class,
            download=download,
        )

        return get_cta_probe_loader(train_dataset, cta=cta, **dataloader_kwargs)

    else:
        raise ValueError("Unhandled dataset: {}".format(dataset_name))
