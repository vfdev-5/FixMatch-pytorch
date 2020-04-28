from base_train import main, BaseTrainer
from configs import get_default_config
import dist_utils


class FullySupervisedTrainer(BaseTrainer):

    output_names = ["sup_loss", ]

    def train_step(self, engine, batch):
        self.model.train()
        self.optimizer.zero_grad()

        x, y = batch["sup_batch"]

        y_pred = self.model(x)

        # supervised learning:
        sup_loss = self.sup_criterion(y_pred, y)

        if self.config["with_nv_amp_level"] is not None:
            from apex import amp
            with amp.scale_loss(sup_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            sup_loss.backward()

        if dist_utils.is_tpu_distributed():
            dist_utils.xm.optimizer_step(self.optimizer)
        else:
            self.optimizer.step()

        return {
            "sup_loss": sup_loss.item(),
        }


if __name__ == "__main__":
    main(FullySupervisedTrainer(), get_default_config())
