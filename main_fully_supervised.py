import hydra
from omegaconf import DictConfig

import ignite.distributed as idist
from ignite.utils import manual_seed, setup_logger

import utils
import trainers


def training(local_rank, cfg):

    logger = setup_logger(
        "Fully-Supervised Training", distributed_rank=idist.get_rank()
    )

    if local_rank == 0:
        logger.info(cfg.pretty())

    rank = idist.get_rank()
    manual_seed(cfg.seed + rank)
    device = idist.device()

    model, ema_model, optimizer, sup_criterion, lr_scheduler = utils.initialize(cfg)

    supervised_train_loader, test_loader, *_ = utils.get_dataflow(cfg)

    def train_step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x = batch["sup_batch"]["image"]
        y = batch["sup_batch"]["target"]
        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        y_pred = model(x)
        sup_loss = sup_criterion(y_pred, y)
        sup_loss.backward()

        optimizer.step()

        return {
            "sup_loss": sup_loss.item(),
        }

    trainer = trainers.create_trainer(
        train_step,
        output_names=["sup_loss",],
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        supervised_train_loader=supervised_train_loader,
        test_loader=test_loader,
        cfg=cfg,
        logger=logger,
    )

    epoch_length = cfg.solver.epoch_length
    num_epochs = cfg.solver.num_epochs if not cfg.debug else 2
    try:
        trainer.run(
            supervised_train_loader, epoch_length=epoch_length, max_epochs=num_epochs
        )
    except Exception as e:
        import traceback

        print(traceback.format_exc())


@hydra.main(config_path="config", config_name="fully_supervised")
def main(cfg: DictConfig) -> None:

    with idist.Parallel(
        backend=cfg.distributed.backend, nproc_per_node=cfg.distributed.nproc_per_node
    ) as parallel:
        parallel.run(training, cfg)


if __name__ == "__main__":
    main()
