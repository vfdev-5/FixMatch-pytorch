
import torch

from ignite.engine import Events
from ignite.utils import convert_tensor

import utils
from base_main import main, BaseTrainer, get_default_config


class FixMatchTrainer(BaseTrainer):

    output_names = ["total_loss", "sup_loss", "unsup_loss", "mask"]

    def train_step(self, *args, **kwargs):
        self.model.train()
        self.optimizer.zero_grad()

        # supervised data:
        sup_batch = next(self.supervised_train_loader_iter)
        x, y = utils.sup_prepare_batch(sup_batch, self.device, non_blocking=True)
        # unsupervised data:
        unsup_batch = next(self.unsupervised_train_loader_iter)
        weak_x = convert_tensor(unsup_batch["image"], self.device, non_blocking=True)
        strong_x = convert_tensor(unsup_batch["strong_aug"], self.device, non_blocking=True)

        # according to TF code: single forward pass on concat data: [x, weak_x, strong_x]
        x_cat = torch.cat([x, weak_x, strong_x], dim=0)
        y_pred_cat = self.model(x_cat)

        idx1 = len(x)
        idx2 = idx1 + len(weak_x)
        y_pred = y_pred_cat[:idx1, ...]
        y_weak_preds = y_pred_cat[idx1:idx2, ...]  # logits_weak
        y_strong_preds = y_pred_cat[idx2:, ...]    # logits_strong

        # supervised learning:
        sup_loss = self.sup_criterion(y_pred, y)

        # unsupervised learning:
        y_weak_probas = torch.softmax(y_weak_preds, dim=1).detach()
        y_pseudo = y_weak_probas.argmax(dim=1)
        max_y_weak_probas, _ = y_weak_probas.max(dim=1)
        unsup_loss_mask = (max_y_weak_probas >= self.confidence_threshold).float()
        unsup_loss = (self.unsup_criterion(y_strong_preds, y_pseudo) * unsup_loss_mask).mean()

        total_loss = sup_loss + self.lambda_u * unsup_loss

        if self.config["with_amp_level"] is not None:
            from apex import amp
            with amp.scale_loss(total_loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()

        self.optimizer.step()

        return {
            "total_loss": total_loss,
            "sup_loss": sup_loss,
            "unsup_loss": unsup_loss,
            "mask": unsup_loss_mask.mean()
        }

    def setup(self, **kwargs):
        super(FixMatchTrainer, self).setup(**kwargs)
        self.confidence_threshold = self.config["confidence_threshold"]
        self.lambda_u = self.config["lambda_u"]
        self.add_event_handler(Events.ITERATION_COMPLETED, self.update_cta_rates)

    def update_cta_rates(self, _):
        cta_probe_batch = next(self.cta_probe_loader_iter)
        x, y = utils.sup_prepare_batch(cta_probe_batch, self.device, non_blocking=True)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x)
            y_probas = torch.softmax(y_pred, dim=1)  # (N, C)

            for y_proba, t, policy_str in zip(y_probas, y, cta_probe_batch['policy']):
                policy = utils.deserialize(policy_str)
                error = y_proba
                error[t] -= 1
                error = torch.abs(error).sum()
                self.cta.update_rates(policy, 1.0 - 0.5 * error.item())


if __name__ == "__main__":
    main(FixMatchTrainer(), get_default_config())
