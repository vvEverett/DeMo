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
from .model_forecast import ModelForecast, StreamModelForecast


class Trainer(pl.LightningModule):
    """PyTorch Lightning trainer for DeMo motion forecasting model.

    Handles training, validation, and testing of trajectory prediction models
    with support for multi-modal predictions and various loss functions.
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
        self.val_metrics_new = metrics.clone(prefix="val_new_")
    
    def get_model(self, model_type):
        """Get the appropriate model class based on model type.

        Args:
            model_type: String specifying which model to use

        Returns:
            Model class (ModelForecast or StreamModelForecast)
        """
        model_dict = {
            'ModelForecast': ModelForecast,  # only 'DeMo'
            'StreamModelForecast': StreamModelForecast,  # integrate 'DeMo' with 'RealMotion'
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
        """Generate predictions for inference with memory persistence across timesteps.

        Used for streaming prediction where previous timestep information
        is maintained in memory for temporal consistency.

        Args:
            data: List of data batches for sequential timesteps

        Returns:
            Tuple of (predictions, probabilities) formatted for submission
        """
        memory_dict = None
        predictions = []
        probs = []
        for i in range(len(data)):
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            memory_dict = out['memory_dict']
            prediction, prob = self.submission_handler.format_data(
                cur_data, out["y_hat"], out["pi"], inference=True)
            predictions.append(prediction)
            probs.append(prob)

        return predictions, probs

    def cal_loss(self, out, data, tag=''):
        """Calculate comprehensive loss for trajectory prediction.

        Computes multiple loss components:
        - Regression loss for trajectory predictions
        - Classification loss for mode selection
        - Loss for other agents in the scene
        - Laplace NLL loss for uncertainty estimation

        Args:
            out: Model outputs containing predictions and auxiliary information
            data: Ground truth data including target trajectories
            tag: Prefix for loss logging

        Returns:
            Tuple of (total_loss, loss_dict) for logging
        """
        y_hat, pi, y_hat_others = out["y_hat"], out["pi"], out["y_hat_others"]
        scal, scal_new = out["scal"], out["scal_new"]
        new_y_hat = out.get("new_y_hat", None)
        new_pi = out.get("new_pi", None)
        dense_predict = out.get("dense_predict", None)

        # gt
        y, y_others = data["target"][:, 0], data["target"][:, 1:]

        # loss for output of state query
        if dense_predict is not None:
            dense_reg_loss = F.smooth_l1_loss(dense_predict, y)
        else:
            dense_reg_loss = 0

        # loss for output of mode query
        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
        best_mode = torch.argmin(l2_norm, dim=-1)
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)
        
        # loss for final output
        if new_y_hat is not None:
            l2_norm_new = torch.norm(new_y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)
            best_mode_new = torch.argmin(l2_norm_new, dim=-1)
            new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), best_mode_new]
            new_agent_reg_loss = F.smooth_l1_loss(new_y_hat_best[..., :2], y)
        else:
            new_agent_reg_loss = 0
        if new_pi is not None:
            new_pi_reg_loss = F.cross_entropy(new_pi, best_mode_new.detach(), label_smoothing=0.2)
        else:
            new_pi_reg_loss = 0

        # loss for other agents
        others_reg_mask = data["target_mask"][:, 1:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        # Laplace loss, which is not necessary
        predictions = {}
        predictions['traj'] = y_hat
        predictions['scale'] = scal
        predictions['probs'] = pi
        laplace_loss = self.laplace_loss.compute(predictions, y)

        predictions['traj'] = new_y_hat
        predictions['scale'] = scal_new
        predictions['probs'] = new_pi
        laplace_loss_new = self.laplace_loss.compute(predictions, y)

        # total loss
        loss = agent_reg_loss + agent_cls_loss + others_reg_loss + \
                new_agent_reg_loss + dense_reg_loss + new_pi_reg_loss
        loss = loss + laplace_loss + laplace_loss_new

        disp_dict = {
            f"{tag}loss": loss.item(),
            f"{tag}reg_loss": agent_reg_loss.item(),
            f"{tag}cls_loss": agent_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
            f"{tag}laplace_loss": laplace_loss.item(),
            f"{tag}laplace_loss_new": laplace_loss_new.item(),
        }
        if new_y_hat is not None:
            disp_dict[f"{tag}reg_loss_refine"] = new_agent_reg_loss.item()
        if new_pi is not None:
            disp_dict[f"{tag}reg_loss_new_pi"] = new_pi_reg_loss.item()
        if dense_predict is not None:
            disp_dict[f"{tag}reg_loss_dense"] = dense_reg_loss.item()

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
        """Single validation step for model evaluation.

        Computes validation metrics including minADE, minFDE, and MR.
        Evaluates both base predictions and refined predictions if available.

        Args:
            data: Validation batch or list of batches
            batch_idx: Index of current batch
        """
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        _, loss_dict = self.cal_loss(out, data)
        metrics = self.val_metrics(out, data['target'][:, 0])
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        if out['new_y_hat'] is not None:
            metrics_new = self.val_metrics_new(out, data['target'][:, 0])

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        if out['new_y_hat'] is not None:
            self.log_dict(
                metrics_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )

    def on_test_start(self) -> None:
        """Initialize submission handler at the start of testing.

        Creates submission directory and sets up handler for
        formatting predictions for competition submission.
        """
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        self.submission_handler = SubmissionAv2(
            save_dir=save_dir
        )

    def test_step(self, data, batch_idx) -> None:
        """Single test step for generating competition submissions.

        Generates predictions and formats them for Argoverse 2
        motion forecasting challenge submission.

        Args:
            data: Test batch or list of batches
            batch_idx: Index of current batch
        """
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        if out['new_y_hat'] is not None:
            out['y_hat'] = out['new_y_hat']
            out['pi'] = out['new_pi']
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        """Finalize submission file after all test steps complete.

        Generates the final submission file containing all predictions
        in the required format for competition evaluation.
        """
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler.

        Sets up AdamW optimizer with weight decay regularization,
        separating parameters that should and shouldn't have weight decay.
        Uses cosine annealing schedule with warmup.

        Returns:
            Tuple of (optimizers, schedulers) for PyTorch Lightning
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                full_param_name = (
                    "%s.%s" % (module_name, param_name) if module_name else param_name
                )
                if "bias" in param_name:
                    no_decay.add(full_param_name)
                elif "weight" in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ("weight" in param_name or "bias" in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters()
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0

        optim_groups = [
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.lr,
            min_lr=1e-5,
            warmup_epochs=self.warmup_epochs,
            epochs=self.epochs,
        )
        return [optimizer], [scheduler]


# integrate 'DeMo' with 'RealMotion'
class StreamTrainer(Trainer):
    """Streaming trainer for temporal sequence processing.

    Extends base trainer to handle streaming data where model
    processes sequences of frames with memory persistence.
    Uses gradient accumulation over multiple frames.
    """

    def __init__(self,
                 num_grad_frame=2,
                 **kwargs):
        """Initialize streaming trainer.

        Args:
            num_grad_frame: Number of frames to accumulate gradients over
            **kwargs: Arguments passed to parent Trainer class
        """
        super().__init__(**kwargs)
        self.num_grad_frame = num_grad_frame
    
    def training_step(self, data, batch_idx):
        """Training step for streaming data with temporal memory.

        Processes a sequence of frames where initial frames run without
        gradients to warm up memory, then accumulates gradients over
        the final frames for efficient training.

        Args:
            data: List of sequential data frames
            batch_idx: Index of current batch

        Returns:
            Accumulated loss over gradient frames
        """
        total_step = len(data)
        num_grad_frames = min(self.num_grad_frame, total_step)
        num_no_grad_frames = total_step - num_grad_frames

        memory_dict = None
        self.eval()
        with torch.no_grad():
            for i in range(num_no_grad_frames):
                cur_data = data[i]
                cur_data['memory_dict'] = memory_dict
                out = self(cur_data)
                memory_dict = out['memory_dict']

        self.train()
        sum_loss = 0
        loss_dict = {}
        for i in range(num_grad_frames):
            cur_data = data[i + num_no_grad_frames]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            cur_loss, cur_loss_dict = self.cal_loss(out, cur_data, tag=f'step{i + num_no_grad_frames}_')
            loss_dict.update(cur_loss_dict)
            sum_loss += cur_loss
            memory_dict = out['memory_dict']
        loss_dict['loss'] = sum_loss.item()
        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )

        return sum_loss
    
    def validation_step(self, data, batch_idx):
        """Validation step for streaming data sequences.

        Processes the entire sequence while maintaining memory state
        across frames. Evaluates metrics on the final frame prediction
        which incorporates temporal information from all previous frames.

        Args:
            data: List of sequential validation frames
            batch_idx: Index of current batch
        """
        memory_dict = None
        all_outs = []
        for i in range(len(data)):
            cur_data = data[i]
            if cur_data['x_positions_diff'].size(1) == 1:
                return
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            _, cur_loss_dict = self.cal_loss(out, cur_data, tag=f'step{i}_')
            memory_dict = out['memory_dict']
            all_outs.append(out)

        metrics = self.val_metrics(all_outs[-1], data[-1]['target'][:, 0])
        if all_outs[-1]['new_y_hat'] is not None:
            all_outs[-1]['y_hat'] = all_outs[-1]['new_y_hat']
            all_outs[-1]['pi'] = all_outs[-1]['new_pi']
        if all_outs[-1]['new_y_hat'] is not None:
            metrics_new = self.val_metrics_new(all_outs[-1], data[-1]['target'][:, 0])

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        if all_outs[-1]['new_y_hat'] is not None:
            self.log_dict(
                metrics_new,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                batch_size=1,
                sync_dist=True,
            )
    
    def test_step(self, data, batch_idx) -> None:
        """Test step for streaming data with temporal context.

        Processes the complete sequence maintaining memory across frames
        and generates final predictions for submission. Uses the refined
        predictions if available for better accuracy.

        Args:
            data: List of sequential test frames
            batch_idx: Index of current batch
        """
        memory_dict = None
        all_outs = []
        for i in range(len(data)):
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            memory_dict = out['memory_dict']
            all_outs.append(out)

        if all_outs[-1]['new_y_hat'] is not None:
            all_outs[-1]['y_hat'] = all_outs[-1]['new_y_hat']
            all_outs[-1]['pi'] = all_outs[-1]['new_pi']

        self.submission_handler.format_data(data[-1], all_outs[-1]["y_hat"], all_outs[-1]["pi"])
