
import torch.nn as nn

import ignite.distributed as idist

from hydra.utils import instantiate

from models import setup_model, setup_ema
from dataflow import get_supervised_train_loader, get_test_loader, get_unsupervised_train_loader, get_cta_probe_loader


def initialize(cfg):
    model = setup_model(cfg.model, num_classes=cfg.num_classes)
    ema_model = setup_model(cfg.model, num_classes=cfg.num_classes)

    model.to(idist.device())
    ema_model.to(idist.device())
    setup_ema(ema_model, model)

    model = idist.auto_model(model)

    if isinstance(model, nn.parallel.DataParallel):
        ema_model = nn.parallel.DataParallel(ema_model)

    optimizer = instantiate(cfg.solver.optimizer, model.parameters())
    optimizer = idist.auto_optim(optimizer)

    sup_criterion = instantiate(cfg.solver.supervised_criterion)

    total_num_iters = cfg.solver.num_epochs * cfg.solver.epoch_length
    lr_scheduler = instantiate(cfg.solver.lr_scheduler, optimizer, T_max=total_num_iters)

    return model, ema_model, optimizer, sup_criterion, lr_scheduler


def get_dataflow(cfg, cta=None, with_unsup=False):

    num_workers = cfg.dataflow.num_workers if cta is None else cfg.dataflow.num_workers // 2

    sup_train_loader = get_supervised_train_loader(
        cfg.dataflow.name,
        root=cfg.dataflow.data_path,
        num_train_samples_per_class=cfg.ssl.num_train_samples_per_class,
        batch_size=cfg.dataflow.batch_size,
        num_workers=num_workers,
    )

    test_loader = get_test_loader(
        cfg.dataflow.name,
        root=cfg.dataflow.data_path,
        batch_size=cfg.dataflow.batch_size,
        num_workers=cfg.dataflow.num_workers,
    )

    unsup_train_loader = None
    if with_unsup:
        if cta is None:
            raise ValueError("If with_unsup=True, cta should be defined, but given None")
        unsup_train_loader = get_unsupervised_train_loader(
            cfg.dataflow.name,
            root=cfg.dataflow.data_path,
            cta=cta,
            batch_size=int(cfg.dataflow.batch_size * cfg.ssl.mu_ratio),
            num_workers=num_workers,
        )

    cta_probe_loader = None
    if cta is not None:
        cta_probe_loader = get_cta_probe_loader(
            cfg.dataflow.name,
            root=cfg.dataflow.data_path,
            num_train_samples_per_class=cfg.ssl.num_train_samples_per_class,
            cta=cta,
            batch_size=int(cfg.dataflow.batch_size * cfg.ssl.mu_ratio),
            num_workers=num_workers,
        )

    return sup_train_loader, test_loader, unsup_train_loader, cta_probe_loader
