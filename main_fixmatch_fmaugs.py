import torch

import ignite.distributed as idist
from ignite.engine import Events
from ignite.utils import manual_seed, setup_logger

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import utils
import trainers


def training(local_rank, cfg):

    logger = setup_logger("FixMatch Training with FMAugs", distributed_rank=idist.get_rank())

    if local_rank == 0:
        logger.info(cfg.pretty())

    rank = idist.get_rank()
    manual_seed(cfg.seed + rank)
    device = idist.device()

    model, ema_model, optimizer, sup_criterion, lr_scheduler = utils.initialize(cfg)

    # If using FMAugs, we added 'fmaugs_enabled' attributed
    assert hasattr(model, "fmaugs_enabled"), "No FMAugs added to the model"

    unsup_criterion = instantiate(cfg.solver.unsupervised_criterion)

    (
        supervised_train_loader,
        test_loader,
        unsup_train_loader,
        _,
    ) = utils.get_dataflow(cfg, with_unsup=True)

    # Enable FMAugs
    model.fmaugs_enabled = not cfg.disable_fmaugs

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch["sup_batch"]["image"], batch["sup_batch"]["target"]
        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        weak_x, strong_x = (
            batch["unsup_batch"]["image"],
            batch["unsup_batch"]["strong_aug"],
        )
        if weak_x.device != device:
            weak_x = weak_x.to(device, non_blocking=True)
            strong_x = strong_x.to(device, non_blocking=True)

        le = 2 * engine.state.mu_ratio + 1
        x_cat = utils.interleave(torch.cat([x, weak_x, strong_x], dim=0), le)
        y_pred_cat = model(x_cat)
        y_pred_cat = utils.deinterleave(y_pred_cat, le)

        idx1 = len(x)
        idx2 = idx1 + len(weak_x)
        y_pred = y_pred_cat[:idx1, ...]
        y_weak_preds = y_pred_cat[idx1:idx2, ...]  # logits_weak
        y_strong_preds = y_pred_cat[idx2:, ...]  # logits_strong

        # supervised learning:
        sup_loss = sup_criterion(y_pred, y)

        # unsupervised learning:
        y_weak_probas = torch.softmax(y_weak_preds, dim=1).detach()
        y_pseudo = y_weak_probas.argmax(dim=1)
        max_y_weak_probas, _ = y_weak_probas.max(dim=1)
        unsup_loss_mask = (
            max_y_weak_probas >= engine.state.confidence_threshold
        ).float()
        unsup_loss = (
            unsup_criterion(y_strong_preds, y_pseudo) * unsup_loss_mask
        ).mean()

        total_loss = sup_loss + engine.state.lambda_u * unsup_loss

        total_loss.backward()

        optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "sup_loss": sup_loss.item(),
            "unsup_loss": unsup_loss.item(),
            "mask": unsup_loss_mask.mean().item(),  # this should not be averaged for DDP
        }

    output_names = ["total_loss", "sup_loss", "unsup_loss", "mask"]

    trainer = trainers.create_trainer(
        train_step,
        output_names=output_names,
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        supervised_train_loader=supervised_train_loader,
        test_loader=test_loader,
        cfg=cfg,
        logger=logger,
        unsup_train_loader=unsup_train_loader,
    )

    trainer.state.confidence_threshold = cfg.ssl.confidence_threshold
    trainer.state.lambda_u = cfg.ssl.lambda_u
    trainer.state.mu_ratio = cfg.ssl.mu_ratio

    # @trainer.on(Events.ITERATION_COMPLETED(every=cfg.ssl.fmaugs_update_every))
    # def update_fmaugs_rates():
    #     batch = trainer.state.batch

    epoch_length = cfg.solver.epoch_length
    num_epochs = cfg.solver.num_epochs if not cfg.debug else 2
    try:
        trainer.run(
            supervised_train_loader, epoch_length=epoch_length, max_epochs=num_epochs
        )
    except Exception as e:

        logger.exception("")
        raise e


@hydra.main(config_path="config", config_name="fixmatch_fmaugs")
def main(cfg: DictConfig) -> None:

    with idist.Parallel(
        backend=cfg.distributed.backend, nproc_per_node=cfg.distributed.nproc_per_node
    ) as parallel:
        parallel.run(training, cfg)


if __name__ == "__main__":
    main()
