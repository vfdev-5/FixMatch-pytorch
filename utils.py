import torch.nn as nn

import ignite.distributed as idist

from hydra.utils import instantiate

from models import setup_model, setup_ema
from dataflow import (
    get_supervised_train_loader,
    get_test_loader,
    get_unsupervised_train_loader,
    get_cta_probe_loader,
)


def load_module(filepath):
    """Method to load module from file path

    Args:
        filepath: path to module to load
    """
    # Taken from https://github.com/vfdev-5/py_config_runner/blob/2d64af869cd89d4fd1f01da1465a913e160cf135/
    # py_config_runner/utils.py#L60

    import importlib.util
    from pathlib import Path

    if not Path(filepath).exists():
        raise ValueError("File '{}' is not found".format(filepath))

    spec = importlib.util.spec_from_file_location(Path(filepath).stem, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initialize(cfg):
    model = setup_model(cfg.model, num_classes=cfg.num_classes)
    ema_model = setup_model(cfg.model, num_classes=cfg.num_classes)

    model.to(idist.device())
    ema_model.to(idist.device())
    setup_ema(ema_model, model)

    if cfg.ssl.get("fmaugs", None) is not None:
        import fmaugs
        from pathlib import Path

        fmaugs_ops_path = Path(__file__).parent / "config" / "ssl" / "fmaugs" / cfg.ssl.fmaugs
        fmaugs_conf = load_module(fmaugs_ops_path)

        fmaugs_type = fmaugs_conf.fmaugs_type
        random_fmaugs = fmaugs_conf.ops

        model.fmaugs_enabled = True

        def transform_feature_map(module, x):
            if module.training and model.fmaugs_enabled:
                return random_fmaugs(x[0])

        fmaugs.augment_model(fmaugs_type, model, transform_feature_map)

    model = idist.auto_model(model)

    if isinstance(model, nn.parallel.DataParallel):
        ema_model = nn.parallel.DataParallel(ema_model)     

    if hasattr(model, "module") and hasattr(model.module, "fmaugs_enabled"):
        # DP or DDP => add fmaugs_enabled as a property
        type(model).fmaugs_enabled = property(
            fget=lambda self: self.module.fmaugs_enabled
        )

    optimizer = instantiate(cfg.solver.optimizer, model.parameters())
    optimizer = idist.auto_optim(optimizer)

    sup_criterion = instantiate(cfg.solver.supervised_criterion)

    total_num_iters = cfg.solver.num_epochs * cfg.solver.epoch_length
    lr_scheduler = instantiate(
        cfg.solver.lr_scheduler, optimizer, T_max=total_num_iters
    )

    return model, ema_model, optimizer, sup_criterion, lr_scheduler


def get_dataflow(cfg, cta=None, with_unsup=False):

    num_workers = (
        cfg.dataflow.num_workers if cta is None else cfg.dataflow.num_workers // 2
    )

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
        unsup_train_loader = get_unsupervised_train_loader(
            cfg.dataflow.name,
            root=cfg.dataflow.data_path,
            cta=cta,  # cta can also be None -> random_strong_transforms is used instead of CTA
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


def interleave(x, batch, inverse=False):
    """
    TF code
    def interleave(x, batch):
        s = x.get_shape().as_list()
        return tf.reshape(
            tf.transpose(tf.reshape(x, [-1, batch] + s[1:]), [1, 0] + list(range(2, 1+len(s)))), [-1] + s[1:]
        )
    """
    shape = x.shape
    axes = [batch, -1] if inverse else [-1, batch]
    return x.reshape(*axes, *shape[1:]).transpose(0, 1).reshape(-1, *shape[1:])


def deinterleave(x, batch):
    return interleave(x, batch, inverse=True)
