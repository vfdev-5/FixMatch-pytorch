
import torch

from ignite.engine import Events
from ignite.utils import convert_tensor

import utils
from base_main import main, get_default_config
from main_fixmatch import FixMatchTrainer


def get_config():
    config = get_default_config()
    config["num_sup_substeps"] = 2
    config["num_unsup_substeps"] = 1
    return config


class FixMatchTwoStepsTrainer(FixMatchTrainer):

    def train_step(self, *args, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()

        # supervised part
        total_loss = 0
        for _ in range(config["num_sup_substeps"]):
            sup_batch = next(self.supervised_train_loader_iter)
            x, y = utils.sup_prepare_batch(sup_batch, self.device, non_blocking=True)
            y_pred = self.model(x)
            sup_loss = self.sup_criterion(y_pred, y)

            if self.config["with_nv_amp_level"] is not None:
                from apex import amp
                with amp.scale_loss(sup_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                sup_loss.backward()
            total_loss += sup_loss

        # pseudo-labeling
        for _ in range(config["num_unsup_substeps"]):
            unsup_batch = next(self.unsupervised_train_loader_iter)
            weak_x = convert_tensor(unsup_batch["image"], self.device, non_blocking=True)
            strong_x = convert_tensor(unsup_batch["strong_aug"], self.device, non_blocking=True)

            y_strong_preds = self.model(strong_x)
            y_weak_preds = self.model(weak_x).detach()
            y_pseudo = y_weak_preds.argmax(dim=1)
            y_weak_probas = torch.softmax(y_weak_preds, dim=1)
            max_y_weak_probas, _ = y_weak_probas.max(dim=1)
            unsup_loss_mask = (max_y_weak_probas > self.confidence_threshold).float()
            unsup_loss = (self.unsup_criterion(y_strong_preds, y_pseudo) * unsup_loss_mask).mean()
            if self.config["with_nv_amp_level"] is not None:
                from apex import amp
                with amp.scale_loss(unsup_loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                unsup_loss.backward()
            total_loss += self.lambda_u * unsup_loss

        self.optimizer.step()

        return {
            "total_loss": total_loss,
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "mask": unsup_loss_mask.mean()
        }


if __name__ == "__main__":
    main(FixMatchTwoStepsTrainer(), get_config())
