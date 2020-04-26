import torch.distributed as dist
from base_train import main, BaseTrainer, get_default_config


class FullySupervisedTrainer(BaseTrainer):

    output_names = ["sup_loss", ]

    def train_step(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()

        x, y = batch["sup_batch"]

        y_pred = self.model(x)

        # supervised learning:
        sup_loss = self.sup_criterion(y_pred, y)

        if self.config["with_amp_level"] is not None:
            from apex import amp
            with amp.scale_loss(sup_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            sup_loss.backward()

        self.optimizer.step()

        return {
            "sup_loss": sup_loss.item(),
        }

    def setup(self, **kwargs):
        super(FullySupervisedTrainer, self).setup(**kwargs)
        self.distributed = dist.is_available() and dist.is_initialized()


if __name__ == "__main__":
    main(FullySupervisedTrainer(), get_default_config())
