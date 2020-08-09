import os
from pathlib import Path

import torch
import torch.nn as nn

import ignite.distributed as idist
from ignite.contrib.engines import common
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall
from ignite.handlers import Checkpoint

from dataflow import sup_prepare_batch, cycle
from ctaugment import stats, deserialize


def to_list_str(v):
    if isinstance(v, torch.Tensor):
        return " ".join(["%.2f" % i for i in v.tolist()])
    return "%.2f" % v


def create_trainer(
    train_step,
    output_names,
    model,
    ema_model,
    optimizer,
    lr_scheduler,
    supervised_train_loader,
    test_loader,
    cfg,
    logger,
    cta=None,
    unsup_train_loader=None,
    cta_probe_loader=None,
):

    trainer = Engine(train_step)
    trainer.logger = logger

    output_path = os.getcwd()

    to_save = {
        "model": model,
        "ema_model": ema_model,
        "optimizer": optimizer,
        "trainer": trainer,
        "lr_scheduler": lr_scheduler,
    }
    if cta is not None:
        to_save["cta"] = cta

    common.setup_common_training_handlers(
        trainer,
        train_sampler=supervised_train_loader.sampler,
        to_save=to_save,
        save_every_iters=cfg.solver.checkpoint_every,
        output_path=output_path,
        output_names=output_names,
        lr_scheduler=lr_scheduler,
        with_pbars=False,
        clear_cuda_cache=False,
    )

    ProgressBar(persist=False).attach(
        trainer, metric_names="all", event_name=Events.ITERATION_COMPLETED
    )

    unsupervised_train_loader_iter = None
    if unsup_train_loader is not None:
        unsupervised_train_loader_iter = cycle(unsup_train_loader)

    cta_probe_loader_iter = None
    if cta_probe_loader is not None:
        cta_probe_loader_iter = cycle(cta_probe_loader)

    # Setup handler to prepare data batches
    @trainer.on(Events.ITERATION_STARTED)
    def prepare_batch(e):
        sup_batch = e.state.batch
        e.state.batch = {
            "sup_batch": sup_batch,
        }
        if unsupervised_train_loader_iter is not None:
            unsup_batch = next(unsupervised_train_loader_iter)
            e.state.batch["unsup_batch"] = unsup_batch

        if cta_probe_loader_iter is not None:
            cta_probe_batch = next(cta_probe_loader_iter)
            cta_probe_batch["policy"] = [
                deserialize(p) for p in cta_probe_batch["policy"]
            ]
            e.state.batch["cta_probe_batch"] = cta_probe_batch

    # Setup handler to update EMA model
    @trainer.on(Events.ITERATION_COMPLETED, cfg.ema_decay)
    def update_ema_model(ema_decay):
        # EMA on parametes
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(ema_decay).add_(param.data, alpha=1.0 - ema_decay)

    # Setup handlers for debugging
    if cfg.debug:

        @trainer.on(Events.STARTED | Events.ITERATION_COMPLETED(every=100))
        @idist.one_rank_only()
        def log_weights_norms():
            wn = []
            ema_wn = []
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                wn.append(torch.mean(param.data))
                ema_wn.append(torch.mean(ema_param.data))

            msg = "\n\nWeights norms"
            msg += "\n- Raw model: {}".format(
                to_list_str(torch.tensor(wn[:10] + wn[-10:]))
            )
            msg += "\n- EMA model: {}\n".format(
                to_list_str(torch.tensor(ema_wn[:10] + ema_wn[-10:]))
            )
            logger.info(msg)

            rmn = []
            rvar = []
            ema_rmn = []
            ema_rvar = []
            for m1, m2 in zip(model.modules(), ema_model.modules()):
                if isinstance(m1, nn.BatchNorm2d) and isinstance(m2, nn.BatchNorm2d):
                    rmn.append(torch.mean(m1.running_mean))
                    rvar.append(torch.mean(m1.running_var))
                    ema_rmn.append(torch.mean(m2.running_mean))
                    ema_rvar.append(torch.mean(m2.running_var))

            msg = "\n\nBN buffers"
            msg += "\n- Raw mean: {}".format(to_list_str(torch.tensor(rmn[:10])))
            msg += "\n- Raw var: {}".format(to_list_str(torch.tensor(rvar[:10])))
            msg += "\n- EMA mean: {}".format(to_list_str(torch.tensor(ema_rmn[:10])))
            msg += "\n- EMA var: {}\n".format(to_list_str(torch.tensor(ema_rvar[:10])))
            logger.info(msg)

        # TODO: Need to inspect a bug
        # if idist.get_rank() == 0:
        #     from ignite.contrib.handlers import ProgressBar
        #
        #     profiler = BasicTimeProfiler()
        #     profiler.attach(trainer)
        #
        #     @trainer.on(Events.ITERATION_COMPLETED(every=200))
        #     def log_profiling(_):
        #         results = profiler.get_results()
        #         profiler.print_results(results)

    # Setup validation engine
    metrics = {
        "accuracy": Accuracy(),
    }

    if not (idist.has_xla_support and idist.backend() == idist.xla.XLA_TPU):
        metrics.update({
            "precision": Precision(average=False),
            "recall": Recall(average=False),
        })

    eval_kwargs = dict(
        metrics=metrics,
        prepare_batch=sup_prepare_batch,
        device=idist.device(),
        non_blocking=True,
    )

    evaluator = create_supervised_evaluator(model, **eval_kwargs)
    ema_evaluator = create_supervised_evaluator(ema_model, **eval_kwargs)

    def log_results(epoch, max_epochs, metrics, ema_metrics):
        msg1 = "\n".join(
            ["\t{:16s}: {}".format(k, to_list_str(v)) for k, v in metrics.items()]
        )
        msg2 = "\n".join(
            ["\t{:16s}: {}".format(k, to_list_str(v)) for k, v in ema_metrics.items()]
        )
        logger.info(
            "\nEpoch {}/{}\nRaw:\n{}\nEMA:\n{}\n".format(epoch, max_epochs, msg1, msg2)
        )
        if cta is not None:
            logger.info("\n" + stats(cta))

    @trainer.on(
        Events.EPOCH_COMPLETED(every=cfg.solver.validate_every)
        | Events.STARTED
        | Events.COMPLETED
    )
    def run_evaluation():
        evaluator.run(test_loader)
        ema_evaluator.run(test_loader)
        log_results(
            trainer.state.epoch,
            trainer.state.max_epochs,
            evaluator.state.metrics,
            ema_evaluator.state.metrics,
        )

    # setup TB logging
    if idist.get_rank() == 0:
        tb_logger = common.setup_tb_logging(
            output_path,
            trainer,
            optimizers=optimizer,
            evaluators={"validation": evaluator, "ema validation": ema_evaluator},
            log_every_iters=15,
        )
        if cfg.online_exp_tracking.wandb:
            from ignite.contrib.handlers import WandBLogger

            wb_dir = Path("/tmp/output-fixmatch-wandb")
            if not wb_dir.exists():
                wb_dir.mkdir()

            wb_name = "{}-{}".format(cfg.name, cfg.model)

            if "fmaugs" in cfg.name:
                wb_name += "-fmaugs-{}".format(cfg.ssl.fmaugs)

            _ = WandBLogger(
                project="fixmatch-pytorch",
                name=wb_name,
                config=cfg,
                sync_tensorboard=True,
                dir=wb_dir.as_posix(),
                reinit=True,
            )

    resume_from = cfg.solver.resume_from
    if resume_from is not None:
        resume_from = list(Path(resume_from).rglob("training_checkpoint*.pt*"))
        if len(resume_from) > 0:
            # get latest
            checkpoint_fp = max(resume_from, key=lambda p: p.stat().st_mtime)
            assert checkpoint_fp.exists(), "Checkpoint '{}' is not found".format(
                checkpoint_fp.as_posix()
            )
            logger.info("Resume from a checkpoint: {}".format(checkpoint_fp.as_posix()))
            checkpoint = torch.load(checkpoint_fp.as_posix())
            Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    @trainer.on(Events.COMPLETED)
    def release_all_resources():
        nonlocal unsupervised_train_loader_iter, cta_probe_loader_iter

        if idist.get_rank() == 0:
            tb_logger.close()

        if unsupervised_train_loader_iter is not None:
            unsupervised_train_loader_iter = None

        if cta_probe_loader_iter is not None:
            cta_probe_loader_iter = None

    return trainer
