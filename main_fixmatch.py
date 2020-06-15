import torch

import ignite.distributed as idist
from ignite.engine import Events
from ignite.utils import manual_seed, setup_logger

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import utils
import trainers
from ctaugment import get_default_cta, OPS, interleave, deinterleave


sorted_op_names = sorted(list(OPS.keys()))


def pack_as_tensor(k, bins, error, size=5, pad_value=-555.0):
    out = torch.empty(size).fill_(pad_value).to(error)
    out[0] = sorted_op_names.index(k)
    le = len(bins)
    out[1] = le
    out[2 : 2 + le] = torch.tensor(bins).to(error)
    out[2 + le] = error
    return out


def unpack_from_tensor(t):
    k_index = int(t[0].item())
    le = int(t[1].item())
    bins = t[2 : 2 + le].tolist()
    error = t[2 + le].item()
    return sorted_op_names[k_index], bins, error


def training(local_rank, cfg):

    logger = setup_logger("FixMatch Training", distributed_rank=idist.get_rank())

    if local_rank == 0:
        logger.info(cfg.pretty())

    rank = idist.get_rank()
    manual_seed(cfg.seed + rank)
    device = idist.device()

    model, ema_model, optimizer, sup_criterion, lr_scheduler = utils.initialize(cfg)

    unsup_criterion = instantiate(cfg.solver.unsupervised_criterion)

    cta = get_default_cta()

    (
        supervised_train_loader,
        test_loader,
        unsup_train_loader,
        cta_probe_loader,
    ) = utils.get_dataflow(cfg, cta=cta, with_unsup=True)

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

        # according to TF code: single forward pass on concat data: [x, weak_x, strong_x]
        le = 2 * engine.state.mu_ratio + 1
        # Why interleave: https://github.com/google-research/fixmatch/issues/20#issuecomment-613010277
        # We need to interleave due to multiple-GPU batch norm issues. Let's say we have to GPUs, and our batch is
        # comprised of labeled (L) and unlabeled (U) images. Let's use a batch size of 2 for making easier visually
        # in my following example.
        #
        # - Without interleaving, we have a batch LLUUUUUU...U (there are 14 U). When the batch is split to be passed
        # to both GPUs, we'll have two batches LLUUUUUU and UUUUUUUU. Note that all labeled examples ended up in batch1
        # sent to GPU1. The problem here is that batch norm will be computed per batch and the moments will lack
        # consistency between batches.
        #
        # - With interleaving, by contrast, the two batches will be LUUUUUUU and LUUUUUUU. As you can notice the
        # batches have the same distribution of labeled and unlabeled samples and will therefore have more consistent
        # moments.
        #
        x_cat = interleave(torch.cat([x, weak_x, strong_x], dim=0), le)
        y_pred_cat = model(x_cat)
        y_pred_cat = deinterleave(y_pred_cat, le)

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
        cta=cta,
        unsup_train_loader=unsup_train_loader,
        cta_probe_loader=cta_probe_loader,
    )

    trainer.state.confidence_threshold = cfg.ssl.confidence_threshold
    trainer.state.lambda_u = cfg.ssl.lambda_u
    trainer.state.mu_ratio = cfg.ssl.mu_ratio

    distributed = idist.get_world_size() > 1

    @trainer.on(Events.ITERATION_COMPLETED)
    def update_cta_rates():
        batch = trainer.state.batch
        x, y = batch["cta_probe_batch"]["image"], batch["cta_probe_batch"]["target"]
        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        policies = batch["cta_probe_batch"]["policy"]

        ema_model.eval()
        with torch.no_grad():
            y_pred = ema_model(x)
            y_probas = torch.softmax(y_pred, dim=1)  # (N, C)

            if distributed:
                for y_proba, t, policy in zip(y_probas, y, policies):
                    error = y_proba
                    error[t] -= 1
                    error = torch.abs(error).sum()
                    cta.update_rates(policy, 1.0 - 0.5 * error.item())
            else:
                error_per_op = []
                for y_proba, t, policy in zip(y_probas, y, policies):
                    error = y_proba
                    error[t] -= 1
                    error = torch.abs(error).sum()
                    for k, bins in policy:
                        error_per_op.append(pack_as_tensor(k, bins, error))
                error_per_op = torch.stack(error_per_op)
                # all gather
                tensor_list = idist.all_gather(error_per_op)
                # update cta rates
                for t in tensor_list:
                    k, bins, error = unpack_from_tensor(t)
                    cta.update_rates([(k, bins),], 1.0 - 0.5 * error)

    epoch_length = cfg.solver.epoch_length
    num_epochs = cfg.solver.num_epochs if not cfg.debug else 2
    try:
        trainer.run(
            supervised_train_loader, epoch_length=epoch_length, max_epochs=num_epochs
        )
    except Exception as e:
        import traceback

        print(traceback.format_exc())


@hydra.main(config_path="config", config_name="fixmatch")
def main(cfg: DictConfig) -> None:

    with idist.Parallel(
        backend=cfg.distributed.backend, nproc_per_node=cfg.distributed.nproc_per_node
    ) as parallel:
        parallel.run(training, cfg)


if __name__ == "__main__":
    main()
