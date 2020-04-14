import argparse
from functools import partial
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import ignite
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall
from ignite.handlers import Checkpoint

from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar

from wrn import WideResNet
import utils


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
        "num_workers": 10,
        "num_epochs": 1024,
        "epoch_length": 2 ** 16 // batch_size,  # epoch_length * num_epochs == 2 ** 20
        "learning_rate": 0.03,
        "validate_every": 1,

        "with_amp_level": None,  # if "O1" or "O2" -> train with apex/amp, otherwise fp32

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
    }
    return config


def run(trainer, output_path, config):
    assert isinstance(trainer, BaseTrainer)
    device = "cuda"
    torch.manual_seed(config["seed"])

    batch_size = config["batch_size"]
    num_workers = config["num_workers"]

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
        num_workers=num_workers
    )

    test_loader = utils.get_test_loader(
        config["data_path"],
        transforms=utils.test_transforms,
        batch_size=batch_size,
        num_workers=num_workers
    )

    cta = utils.get_default_cta()

    cta_probe_loader = utils.get_cta_probe_loader(
        supervised_train_dataset,
        cta=cta,
        batch_size=batch_size,
        num_workers=num_workers
    )

    unsupervised_train_loader = utils.get_unsupervised_train_loader(
        config["data_path"],
        transforms_weak=utils.weak_transforms,
        transforms_strong=partial(utils.cta_image_transforms, cta=cta),
        batch_size=batch_size * config["mu_ratio"],
        num_workers=num_workers
    )

    # Setup model, optimizer, criterion
    model = WideResNet(num_classes=10)
    model = model.to(device)

    # Setup EMA model
    ema_model = WideResNet(num_classes=10).to(device)
    ema_model.load_state_dict(model.state_dict())
    for param in ema_model.parameters():
        param.detach_()

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
        model, optimizer = amp.initialize(model, optimizer, opt_level=config["with_amp_level"])

    if torch.cuda.device_count() > 0:
        model = nn.parallel.DataParallel(model)

    sup_criterion = nn.CrossEntropyLoss().to(device)
    unsup_criterion = nn.CrossEntropyLoss(reduction='none').to(device)

    num_epochs = config["num_epochs"]
    epoch_length = config["epoch_length"]
    total_num_iters = num_epochs * epoch_length
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_num_iters, eta_min=0.0)

    # Setup training/validation loops
    supervised_train_loader_iter = utils.cycle(supervised_train_loader)
    unsupervised_train_loader_iter = utils.cycle(unsupervised_train_loader)
    cta_probe_loader_iter = utils.cycle(cta_probe_loader)

    trainer.setup(
        config=config,
        model=model, ema_model=ema_model, optimizer=optimizer,
        sup_criterion=sup_criterion, unsup_criterion=unsup_criterion,
        supervised_train_loader_iter=supervised_train_loader_iter, 
        unsupervised_train_loader_iter=unsupervised_train_loader_iter, 
        cta_probe_loader_iter=cta_probe_loader_iter,
        cta=cta,
        device=device
    )

    to_save = {
        "model": model, "optimizer": optimizer, "trainer": trainer, "lr_scheduler": lr_scheduler
    }

    if config["with_amp_level"] is not None:
        from apex import amp
        to_save["amp"] = amp

    common.setup_common_training_handlers(
        trainer,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        output_path=output_path,
        output_names=trainer.output_names,
        lr_scheduler=lr_scheduler,
        with_pbar_on_iters=config["display_iters"],
        log_every_iters=10
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_ema_model():
        ema_decay = config["ema_decay"]
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(ema_decay).add_(param.data, alpha=1.0 - ema_decay)

    # Setup validation
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

    def log_results(epoch, max_epochs, metrics):
        def to_list_str(v):
            if isinstance(v, torch.Tensor):
                return " ".join(["%.2f" % i for i in v.tolist()])
            return "%.2f" % v

        msg = "\n".join(["{:16s}: {}".format(k, to_list_str(v)) for k, v in metrics.items()])
        print("\nEpoch {}/{}\n{}".format(epoch, max_epochs, msg))
        print(utils.stats(cta))

    def run_evaluation():
        evaluator.run(test_loader)
        ema_evaluator.run(test_loader)
        log_results(trainer.state.epoch, trainer.state.max_epochs, ema_evaluator.state.metrics)

    ev = Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.STARTED | Events.COMPLETED
    trainer.add_event_handler(ev, run_evaluation)

    # setup TB logging
    tb_logger = common.setup_tb_logging(
        output_path,
        trainer,
        optimizers=optimizer,
        evaluators={"validation": evaluator, "ema validation": ema_evaluator},
        log_every_iters=1
    )

    if config["display_iters"]:
        ProgressBar(persist=False, desc="Test evaluation").attach(evaluator)

    resume_from = list(Path(output_path).rglob("training_checkpoint*.pt*"))
    if len(resume_from) > 0:
        # get latest
        checkpoint_fp = max(resume_from, key=lambda p: p.stat().st_mtime)
        assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(checkpoint_fp.as_posix())
        print("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
        checkpoint = torch.load(checkpoint_fp.as_posix())
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    data = list(range(epoch_length))
    try:
        trainer.run(data, max_epochs=config["num_epochs"])
    except Exception as e:
        import traceback
        print(traceback.format_exc())

    tb_logger.close()


def main(trainer, config):
    parser = argparse.ArgumentParser("Semi-Supervised Learning - FixMatch with CTA: Train WRN-28-2 on CIFAR10 dataset")
    parser.add_argument(
        "--params",
        type=str,
        help="Override default configuration with parameters: "
        "data_path=/path/to/dataset;batch_size=64;num_workers=12 ...",
    )

    args = parser.parse_args()

    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True

    # Override config:
    if args.params is not None:
        for param in args.params.split(";"):
            key, value = param.split("=")
            if "/" not in value:
                value = eval(value)
            config[key] = value

    print("Train {} on CIFAR10".format(config["model"]))
    print("- PyTorch version: {}".format(torch.__version__))
    print("- Ignite version: {}".format(ignite.__version__))
    print("- CUDA version: {}".format(torch.version.cuda))

    print("\n")
    print("Configuration:")
    for key, value in config.items():
        print("\t{}: {}".format(key, value))
    print("\n")

    now = datetime.now().strftime("%Y%m%d-%H%M%S")
    gpu_conf = "-{}".format(torch.cuda.device_count())
    output_path = Path(config["output_path"]) / "{}{}".format(now, gpu_conf)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    output_path = output_path.as_posix()
    print("Output path: {}".format(output_path))

    try:
        run(trainer, output_path, config)
    except KeyboardInterrupt:
        print("Catched KeyboardInterrupt -> exit")
    except Exception as e:
        raise e


class BaseTrainer(Engine):

    def __init__(self):
        super(BaseTrainer, self).__init__(self.train_step)

    def setup(self, **kwargs):
        for k, v in kwargs.items():
            if k != 'self' and not k.startswith('_'):
                setattr(self, k, v)

    def train_step(self, engine, _):
        raise NotImplementedError("This is the base class")
