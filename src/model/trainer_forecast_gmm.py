import datetime
from pathlib import Path
import time
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.metrics import MR, minADE, minFDE, brier_minFDE
from src.utils.optim import WarmupCosLR
from src.utils.submission_av2 import SubmissionAv2
from src.utils.LaplaceNLLLoss import LaplaceNLLLoss
from .model_forecast_gmm import ModelForecast


class Trainer(pl.LightningModule):
    """PyTorch Lightning trainer for DeMo motion forecasting model with GMM predictor.

    Handles training, validation, and testing of trajectory prediction models
    with GMM-based multi-modal predictions and Laplace loss.
    """

    def __init__(
        self,
        model: dict,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
    ) -> None:
        """Initialize the trainer with model configuration and training parameters.

        Args:
            model: Dictionary containing model type and configuration
            pretrained_weights: Path to pretrained checkpoint file
            lr: Learning rate for optimizer
            warmup_epochs: Number of epochs for learning rate warmup
            epochs: Total number of training epochs
            weight_decay: Weight decay coefficient for regularization
        """
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        model_type = model.pop('type')

        self.net = self.get_model(model_type)(**model)

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)
            print('Pretrained weights have been loaded.')

        metrics = MetricCollection(
            {
                "minADE1": minADE(k=1),
                "minADE6": minADE(k=6),
                "minFDE1": minFDE(k=1),
                "minFDE6": minFDE(k=6),
                "MR": MR(),
                "b-minFDE6": brier_minFDE(k=6),
            }
        )
        self.laplace_loss = LaplaceNLLLoss()
        self.val_metrics = metrics.clone(prefix="val_")
    
    def get_model(self, model_type):
        """Get the appropriate model class based on model type.

        Args:
            model_type: String specifying which model to use

        Returns:
            Model class (ModelForecast)
        """
        model_dict = {
            'ModelForecast': ModelForecast,  # DeMo with GMM predictor (Mode Localization Module only)
        }
        assert model_type in model_dict
        return model_dict[model_type]

    def forward(self, data):
        """Forward pass through the model.

        Args:
            data: Input batch containing agent trajectories and scene context

        Returns:
            Model predictions including trajectories, probabilities, and auxiliary outputs
        """
        return self.net(data)

    def predict(self, data):
        """Generate predictions for inference.

        Args:
            data: List of data batches for sequential timesteps

        Returns:
            Tuple of (predictions, probabilities) formatted for submission
        """
        predictions = []
        probs = []
        for i in range(len(data)):
            cur_data = data[i]
            out = self(cur_data)
            prediction, prob = self.submission_handler.format_data(
                cur_data, out["y_hat"], out["pi"], inference=True)
            predictions.append(prediction)
            probs.append(prob)

        return predictions, probs

    def cal_loss(self, out, data, tag=''):
        """Calculate loss for trajectory prediction with GMM predictor.

        Computes loss components:
        - Regression loss for trajectory predictions from Mode Localization Module
        - Classification loss for mode selection  
        - Laplace loss with scale parameters from GMM
        - Loss for other agents in the scene

        Args:
            out: Model outputs containing predictions and auxiliary information
            data: Ground truth data including target trajectories
            tag: Prefix for loss logging

        Returns:
            Tuple of (total_loss, loss_dict) for logging
        """
        y_hat, pi, scal, y_hat_others = out["y_hat"], out["pi"], out["scal"], out["y_hat_others"]

        # gt
        y, y_others = data["target"][:, 0], data["target"][:, 1:]

        # loss for output of Mode Localization Module (GMM predictor)
        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)

        # loss for other agents
        others_reg_mask = data["target_mask"][:, 1:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        # Laplace loss with GMM scale parameters
        predictions = {}
        predictions['traj'] = y_hat
        predictions['scale'] = scal
        predictions['probs'] = pi
        laplace_loss = self.laplace_loss.compute(predictions, y)

        # total loss
        loss = agent_reg_loss + agent_cls_loss + others_reg_loss + laplace_loss

        disp_dict = {
            f"{tag}loss": loss.item(),
            f"{tag}reg_loss": agent_reg_loss.item(),
            f"{tag}cls_loss": agent_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
            f"{tag}laplace_loss": laplace_loss.item(),
        }

        return loss, disp_dict

    def training_step(self, data, batch_idx):
        """Single training step for one batch.

        Args:
            data: Input batch or list of batches
            batch_idx: Index of current batch

        Returns:
            Training loss for backpropagation
        """
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        loss, loss_dict = self.cal_loss(out, data)

        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return loss

    def validation_step(self, data, batch_idx):
        """Single validation step for one batch.

        Args:
            data: Input batch or list of batches
            batch_idx: Index of current batch
        """
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        _, loss_dict = self.cal_loss(out, data, tag='val_')
        metrics = self.val_metrics(out, data['target'][:, 0])

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

        for k, v in loss_dict.items():
            self.log(
                k,
                v,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def test_step(self, data, batch_idx):
        """Single test step for one batch.

        Generates predictions and saves them for submission.

        Args:
            data: Input batch or list of batches
            batch_idx: Index of current batch
        """
        if isinstance(data, list):
            predictions, probs = self.predict(data)
        else:
            out = self(data)
            predictions, probs = self.submission_handler.format_data(
                data, out["y_hat"], out["pi"], inference=True)

        # save predictions for submission
        self.submission_handler.append(data, predictions, probs)

    def on_test_end(self):
        """Called at the end of testing to save all predictions."""
        output_dir = Path(self.logger.log_dir) / "submission"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        submission_file = output_dir / f"single_agent_{timestamp}.parquet"
        
        self.submission_handler.generate_submission_file(submission_file)
        print(f"Submission file saved to {submission_file}")

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        Uses AdamW optimizer with warmup and cosine annealing schedule.

        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-6,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        """Custom learning rate scheduler step.

        Args:
            scheduler: Learning rate scheduler
            metric: Metric value (unused)
        """
        scheduler.step(epoch=self.current_epoch)
