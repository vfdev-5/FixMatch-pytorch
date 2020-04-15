import argparse
from functools import partial
from pathlib import Path
import yaml
import hashlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import ignite
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall
from ignite.handlers import Checkpoint
from ignite.utils import convert_tensor

from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.time_profilers import BasicTimeProfiler


from wrn import WideResNet
import utils


device = "cuda"


def run(trainer, config):
    assert isinstance(trainer, BaseTrainer)
    
    local_rank = config["local_rank"]
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        torch.cuda.set_device(local_rank)
    rank = dist.get_rank() if distributed else 0

    torch.manual_seed(config["seed"] + rank)
    
    debug = config["debug"]

    cta = utils.get_default_cta()

    supervised_train_loader_iter, unsupervised_train_loader_iter, cta_probe_loader_iter = \
        get_dataflow_iters(config, cta, distributed)

    test_loader = utils.get_test_loader(
        config["data_path"],
        transforms=utils.test_transforms,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    model, ema_model, optimizer = get_models_optimizer(config, distributed)

    sup_criterion = nn.CrossEntropyLoss().to(device)
    unsup_criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    num_epochs = config["num_epochs"]
    epoch_length = config["epoch_length"]
    total_num_iters = num_epochs * epoch_length
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_num_iters, eta_min=0.0)

    # Setup trainer
    trainer.setup(
        config=config,
        model=model, ema_model=ema_model, optimizer=optimizer,
        sup_criterion=sup_criterion, unsup_criterion=unsup_criterion,
        cta=cta,
        device=device
    )

    # Setup handler to prepare data batches
    @trainer.on(Events.ITERATION_STARTED)
    def prepare_batch(e):
        sup_batch = next(supervised_train_loader_iter)
        unsup_batch = next(unsupervised_train_loader_iter)
        cta_probe_batch = next(cta_probe_loader_iter)        
        e.state.batch = {
            "sup_batch": utils.sup_prepare_batch(sup_batch, device, non_blocking=True),
            "unsup_batch": (
                convert_tensor(unsup_batch["image"], device, non_blocking=True),
                convert_tensor(unsup_batch["strong_aug"], device, non_blocking=True)
            ),
            "cta_probe_batch": (
                *utils.sup_prepare_batch(cta_probe_batch, device, non_blocking=True),
                cta_probe_batch['policy']
            )
        }

    # Setup other common handlers for the trainer
    to_save = {
        "model": model, 
        "ema_model": ema_model, 
        "optimizer": optimizer, 
        "trainer": trainer, 
        "lr_scheduler": lr_scheduler,
        "cta": cta
    }

    if config["with_amp_level"] is not None:
        from apex import amp
        to_save["amp"] = amp

    common.setup_common_training_handlers(
        trainer,
        # to_save=None if debug else to_save,
        save_every_iters=config["checkpoint_every"],
        # output_path=config["output_path"],
        output_names=trainer.output_names,
        lr_scheduler=lr_scheduler,
        with_pbar_on_iters=config["display_iters"],
        log_every_iters=2
    )

    # Temporary fix until: https://github.com/pytorch/ignite/pull/923
    if rank == 0:
        from ignite.handlers import ModelCheckpoint
        checkpoint_handler = ModelCheckpoint(
            dirname=config["output_path"], 
            filename_prefix="training", 
            require_empty=False
        )
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(every=config["checkpoint_every"]), 
            checkpoint_handler, 
            to_save
        )

    # Setup handler to update EMA model
    @trainer.on(Events.ITERATION_COMPLETED, config["ema_decay"])
    def update_ema_model(ema_decay):
        # EMA on parametes
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(ema_decay).add_(param.data, alpha=1.0 - ema_decay)

    # Setup handlers for debugging
    if debug and rank == 0:

        @trainer.on(Events.STARTED | Events.ITERATION_COMPLETED(every=100))
        def log_weights_norms(_):
            wn = []
            ema_wn = []
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                wn.append(torch.mean(param.data))
                ema_wn.append(torch.mean(ema_param.data))

            print("\n\nWeights norms")
            print("\n- Raw model: {}".format(to_list_str(torch.tensor(wn[:10] + wn[-10:]))))
            print("- EMA model: {}\n".format(to_list_str(torch.tensor(ema_wn[:10] + ema_wn[-10:]))))

        profiler = BasicTimeProfiler()
        profiler.attach(trainer)
        
        @trainer.on(Events.ITERATION_COMPLETED(every=200))
        def log_profiling(_):
            results = profiler.get_results()
            profiler.print_results(results)

    # Setup validation engine
    metrics = {
        "accuracy": Accuracy(),
        "precision": Precision(average=False),
        "recall": Recall(average=False),
    }

    evaluator = create_supervised_evaluator(
        model, metrics,
        prepare_batch=utils.sup_prepare_batch, device=device, non_blocking=True
    )
    ema_evaluator = create_supervised_evaluator(
        ema_model, metrics,
        prepare_batch=utils.sup_prepare_batch, device=device, non_blocking=True
    )

    def log_results(epoch, max_epochs, metrics, ema_metrics):
        msg1 = "\n".join(["{:16s}: {}".format(k, to_list_str(v)) for k, v in metrics.items()])
        msg2 = "\n".join(["{:16s}: {}".format(k, to_list_str(v)) for k, v in ema_metrics.items()])
        print("\nEpoch {}/{}\nRaw:\n{}\nEMA:\n{}\n".format(epoch, max_epochs, msg1, msg2))
        print(utils.stats(cta))

    def run_evaluation():
        evaluator.run(test_loader)
        ema_evaluator.run(test_loader)
        if rank == 0:
            log_results(
                trainer.state.epoch, 
                trainer.state.max_epochs,
                evaluator.state.metrics,
                ema_evaluator.state.metrics
            )

    ev = Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.STARTED | Events.COMPLETED
    trainer.add_event_handler(ev, run_evaluation)

    # setup TB logging
    if rank == 0:
        tb_logger = common.setup_tb_logging(
            config["output_path"],
            trainer,
            optimizers=optimizer,
            evaluators={"validation": evaluator, "ema validation": ema_evaluator},
            log_every_iters=1
        )

        if config["display_iters"]:
            ProgressBar(persist=False, desc="Test evaluation").attach(evaluator)

    data = list(range(epoch_length))

    resume_from = list(Path(config["output_path"]).rglob("training_checkpoint*.pt*"))
    if len(resume_from) > 0:
        # get latest
        checkpoint_fp = max(resume_from, key=lambda p: p.stat().st_mtime)
        assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
        if rank == 0:
            print("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
        checkpoint = torch.load(checkpoint_fp.as_posix())
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    try:
        trainer.run(data, epoch_length=epoch_length, max_epochs=config["num_epochs"] if not debug else 1)
    except Exception as e:
        import traceback
        print(traceback.format_exc())

    if rank == 0:
        tb_logger.close()


def main(trainer, config):
    parser = argparse.ArgumentParser("Semi-Supervised Learning - FixMatch with CTA: Train WRN-28-2 on CIFAR10 dataset")
    parser.add_argument(
        "--params",
        type=str,
        help="Override default configuration with parameters: "
        "data_path=/path/to/dataset;batch_size=64;num_workers=12 ...",
    )
    parser.add_argument("--local_rank", type=int, help="Local process rank in distributed computation")

    args = parser.parse_args()

    assert torch.cuda.is_available(), "Training is no GPU only"
    torch.backends.cudnn.benchmark = True

    config["local_rank"] = args.local_rank if args.local_rank is not None else 0

    # Override config:
    if args.params is not None:
        for param in args.params.split(";"):
            key, value = param.split("=")
            if "/" not in value:
                value = eval(value)
            config[key] = value

    if config["local_rank"] == 0:
        ds_id = "{}".format(config["num_train_samples_per_class"] * 10)
        print("SSL Training of {} on CIFAR10@{}".format(config["model"], ds_id))
        print("- PyTorch version: {}".format(torch.__version__))
        print("- Ignite version: {}".format(ignite.__version__))
        print("- CUDA version: {}".format(torch.version.cuda))

        print("\n")
        print("Configuration:")
        for key, value in config.items():
            print("\t{}: {}".format(key, value))
        print("\n")

    backend = config["dist_backend"]
    distributed = backend is not None
    if distributed:
        dist.init_process_group(backend, init_method=config["dist_url"])
        # let each node print the info
        if config["local_rank"] == 0:
            print("\nDistributed setting:")
            print("\tbackend: {}".format(dist.get_backend()))
            print("\tworld size: {}".format(dist.get_world_size()))
            print("\trank: {}".format(dist.get_rank()))
            print("\n")

    conf_hash = hashlib.md5(repr(config).encode("utf-8")).hexdigest()
    prefix = "training" if not config["debug"] else "debug-training"
    output_path = Path(config["output_path"]) / "{}-{}-{}".format(prefix, ds_id, conf_hash)

    if (not distributed) or (dist.get_rank() == 0):

        if not output_path.exists():
            output_path.mkdir(parents=True)

        # dump config:
        with open((output_path / "config.yaml"), "w") as h:
            yaml.dump(config, h)
        
        print("Output path: {}".format(output_path.as_posix()))

    output_path = output_path.as_posix()
    config["output_path"] = output_path

    try:
        run(trainer, config)
    except KeyboardInterrupt:
        print("Catched KeyboardInterrupt -> exit")
    except Exception as e:
        if distributed:
            dist.destroy_process_group()
        raise e

    if distributed:
        dist.destroy_process_group()


class BaseTrainer(Engine):

    def __init__(self):
        super(BaseTrainer, self).__init__(self.train_step)

    def setup(self, **kwargs):
        for k, v in kwargs.items():
            if k != 'self' and not k.startswith('_'):
                setattr(self, k, v)

    def train_step(self, engine, batch):
        raise NotImplementedError("This is the base class")


def get_default_config():
    batch_size = 64

    config = {
        "seed": 12,
        "data_path": "/tmp/cifar10",
        "output_path": "/tmp/output-fixmatch-cifar10",
        "model": "WRN-28-2",
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "batch_size": batch_size,
        "num_workers": 12,
        "num_epochs": 1024,
        "epoch_length": 2 ** 16 // batch_size,  # epoch_length * num_epochs == 2 ** 20
        "learning_rate": 0.03,
        "validate_every": 1,

        # AMP
        "with_amp_level": None,  # if "O1" or "O2" -> train with apex/amp, otherwise fp32 (None)

        # DDP        
        "dist_url": "env://",
        "dist_backend": None,  # if None distributed option is disabled, set to "nccl" to enable

        # SSL settings
        "num_train_samples_per_class": 25,
        "mu_ratio": 7,
        "ema_decay": 0.999,

        # FixMatch settings
        "confidence_threshold": 0.95,
        "lambda_u": 1.0,

        # Logging:
        "display_iters": True,
        "checkpoint_every": 200,
        "debug": False
    }
    return config


def to_list_str(v):
    if isinstance(v, torch.Tensor):
        return " ".join(["%.2f" % i for i in v.tolist()])
    return "%.2f" % v


def get_dataflow_iters(config, cta, distributed=False):

    # Rescale batch_size and num_workers
    ngpus_per_node = torch.cuda.device_count()
    ngpus = dist.get_world_size() if distributed else 1
    batch_size = config["batch_size"] // ngpus
    num_workers = int((config["num_workers"] + ngpus_per_node - 1) / ngpus_per_node)
    num_workers //= 3  # 3 dataloaders

    # Setup dataflow
    if config["num_train_samples_per_class"] == 25:
        supervised_train_dataset = utils.get_supervised_trainset_0_250(config["data_path"])
    else:
        supervised_train_dataset = utils.get_supervised_trainset(
            config["data_path"],
            config["num_train_samples_per_class"]
        )

    supervised_train_loader = utils.get_supervised_train_loader(
        supervised_train_dataset,
        transforms=utils.weak_transforms,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=DistributedSampler(supervised_train_dataset) if distributed else None
    )

    cta_probe_loader = utils.get_cta_probe_loader(
        supervised_train_dataset,
        cta=cta,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=DistributedSampler(supervised_train_dataset) if distributed else None
    )
    
    full_train_dataset = utils.CIFAR10(config["data_path"], train=True)
    unsupervised_train_loader = utils.get_unsupervised_train_loader(
        full_train_dataset,
        transforms_weak=utils.weak_transforms,
        transforms_strong=partial(utils.cta_image_transforms, cta=cta),
        batch_size=batch_size * config["mu_ratio"],
        num_workers=num_workers,
        sampler=DistributedSampler(full_train_dataset) if distributed else None
    )

    # Setup training/validation loops
    supervised_train_loader_iter = utils.cycle(supervised_train_loader)
    unsupervised_train_loader_iter = utils.cycle(unsupervised_train_loader)
    cta_probe_loader_iter = utils.cycle(cta_probe_loader)

    return supervised_train_loader_iter, unsupervised_train_loader_iter, cta_probe_loader_iter


def get_models_optimizer(config, distributed=False):
    
    # Setup model, optimizer
    model = WideResNet(num_classes=10)
    model = model.to(device)

    # Setup EMA model
    ema_model = WideResNet(num_classes=10).to(device)
    utils.setup_ema(ema_model, model)

    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        nesterov=True,
    )

    if config["with_amp_level"] is not None:
        assert config["with_amp_level"] in ("O1", "O2")
        from apex import amp
        models, optimizer = amp.initialize([model, ema_model], optimizer, opt_level=config["with_amp_level"])
        model, ema_model = models

    if distributed:
        model = DDP(model, device_ids=[config["local_rank"],])
        # ema_model has no grads => DDP wont work        
    elif config["with_amp_level"] in ("O1", None) and torch.cuda.device_count() > 0:
        model = nn.parallel.DataParallel(model)
        ema_model = nn.parallel.DataParallel(ema_model)

    return model, ema_model, optimizer
