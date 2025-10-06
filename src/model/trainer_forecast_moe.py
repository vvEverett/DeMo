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
from .model_forecast_moe import ModelForecast


class Trainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for DeMo motion forecasting with Mixture of Experts (MoE).
    
    Handles training, validation, and testing of trajectory prediction models
    with MoE architecture for learning diverse driving patterns.
    
    Key features:
    - 6 specialized experts for different driving behaviors
    - Top-2 expert activation with sparse gating
    - Load balancing loss for uniform expert utilization
    - Unsupervised learning of driving patterns through MoE
    """

    def __init__(
        self,
        model: dict,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        aux_loss_weight: float = 0.01,
        diversity_loss_weight: float = 0.1,
    ) -> None:
        """
        Initialize the MoE trainer with model configuration and training parameters.

        Args:
            model: Dictionary containing model type and configuration
            pretrained_weights: Path to pretrained checkpoint file
            lr: Learning rate for optimizer
            warmup_epochs: Number of epochs for learning rate warmup
            epochs: Total number of training epochs
            weight_decay: Weight decay coefficient for regularization
            aux_loss_weight: Weight for auxiliary load balancing loss
            diversity_loss_weight: Weight for expert diversity loss
        """
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.aux_loss_weight = aux_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        model_type = model.pop('type')
        assert model_type == 'ModelForecast', "Only ModelForecast is supported for MoE"

        self.net = ModelForecast(**model)

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
        
        # For tracking expert utilization
        self.expert_usage_buffer = []
    
    def forward(self, data):
        """
        Forward pass through the MoE model.

        Args:
            data: Input batch containing agent trajectories and scene context

        Returns:
            Model predictions including trajectories, probabilities, auxiliary loss, and expert weights
        """
        return self.net(data)

    def predict(self, data):
        """
        Generate predictions for inference.

        Args:
            data: Input batch or list of batches

        Returns:
            Tuple of (predictions, probabilities) formatted for submission
        """
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        prediction, prob = self.submission_handler.format_data(
            data, out["y_hat"], out["pi"], inference=True)
        return prediction, prob

    def cal_loss(self, out, data, tag=''):
        """
        Calculate comprehensive loss for MoE trajectory prediction.

        Computes loss components:
        - Regression loss for trajectory predictions
        - Classification loss for mode selection  
        - Loss for other agents in the scene
        - Auxiliary load balancing loss for MoE
        - Diversity loss to encourage expert specialization

        Args:
            out: Model outputs containing predictions and auxiliary information
            data: Ground truth data including target trajectories
            tag: Prefix for loss logging

        Returns:
            Tuple of (total_loss, loss_dict) for logging
        """
        y_hat = out["y_hat"]  # [B, M, T, 2]
        pi = out["pi"]  # [B, M]
        y_hat_others = out["y_hat_others"]  # [B, N-1, T, 2]
        aux_loss = out["aux_loss"]  # scalar
        expert_weights = out["expert_weights"]  # [B, num_experts]

        # Ground truth
        y = data["target"][:, 0]  # [B, T, 2]
        y_others = data["target"][:, 1:]  # [B, N-1, T, 2]

        # ============ Main trajectory prediction loss ============
        # Find best mode based on L2 distance
        l2_norm = torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)  # [B, M]
        best_mode = torch.argmin(l2_norm, dim=-1)  # [B]
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]  # [B, T, 2]
        
        # Regression loss
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[..., :2], y)
        
        # Classification loss with label smoothing
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach(), label_smoothing=0.2)

        # ============ Other agents loss ============
        others_reg_mask = data["target_mask"][:, 1:]
        others_reg_loss = F.smooth_l1_loss(
            y_hat_others[others_reg_mask], y_others[others_reg_mask]
        )

        # ============ MoE-specific losses ============
        # 1. Auxiliary load balancing loss (already computed in model)
        moe_aux_loss = aux_loss * self.aux_loss_weight
        
        # 2. Expert diversity loss to encourage specialization
        # Encourage different experts to be selected for different samples
        diversity_loss = self._compute_diversity_loss(expert_weights) * self.diversity_loss_weight

        # ============ Total loss ============
        total_loss = (
            agent_reg_loss + 
            agent_cls_loss + 
            others_reg_loss + 
            moe_aux_loss + 
            diversity_loss
        )

        # ============ Loss dictionary for logging ============
        disp_dict = {
            f"{tag}loss": total_loss.item(),
            f"{tag}reg_loss": agent_reg_loss.item(),
            f"{tag}cls_loss": agent_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
            f"{tag}moe_aux_loss": moe_aux_loss.item(),
            f"{tag}diversity_loss": diversity_loss.item(),
        }
        
        # Add expert utilization statistics
        if self.training:
            expert_stats = self._compute_expert_statistics(expert_weights)
            disp_dict.update({f"{tag}{k}": v for k, v in expert_stats.items()})

        return total_loss, disp_dict
    
    def _compute_diversity_loss(self, expert_weights):
        """
        Compute diversity loss to encourage expert specialization.
        
        We want different samples to use different experts, which encourages
        experts to specialize in different driving behaviors.
        
        Args:
            expert_weights: [B, num_experts] - expert selection probabilities
            
        Returns:
            diversity_loss: scalar tensor
        """
        # Compute pairwise cosine similarity between expert weight distributions
        # High similarity means experts are being used similarly (bad)
        # Low similarity means experts are specialized (good)
        
        # Normalize expert weights
        expert_weights_norm = F.normalize(expert_weights, p=2, dim=1)  # [B, num_experts]
        
        # Compute cosine similarity matrix [B, B]
        similarity_matrix = torch.mm(expert_weights_norm, expert_weights_norm.t())
        
        # We want low similarity (encourage diversity)
        # Remove diagonal (self-similarity is always 1)
        B = expert_weights.size(0)
        mask = ~torch.eye(B, dtype=torch.bool, device=expert_weights.device)
        off_diagonal_sim = similarity_matrix[mask]
        
        # Penalize high similarity
        diversity_loss = off_diagonal_sim.mean()
        
        return diversity_loss
    
    def _compute_expert_statistics(self, expert_weights):
        """
        Compute statistics about expert utilization for logging.
        
        Args:
            expert_weights: [B, num_experts]
            
        Returns:
            dict with statistics
        """
        with torch.no_grad():
            # Average usage per expert across batch
            avg_usage = expert_weights.mean(dim=0)  # [num_experts]
            
            # Find most and least used experts
            max_usage = avg_usage.max().item()
            min_usage = avg_usage.min().item()
            
            # Standard deviation of usage (measure of imbalance)
            usage_std = avg_usage.std().item()
            
            # Entropy of average distribution (higher = more uniform)
            eps = 1e-8
            entropy = -(avg_usage * torch.log(avg_usage + eps)).sum().item()
            
            stats = {
                'expert_max_usage': max_usage,
                'expert_min_usage': min_usage,
                'expert_usage_std': usage_std,
                'expert_entropy': entropy,
            }
            
            # Individual expert usage
            for i in range(avg_usage.size(0)):
                stats[f'expert_{i}_usage'] = avg_usage[i].item()
        
        return stats

    def training_step(self, data, batch_idx):
        """
        Single training step for one batch.

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

        # Log all losses
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
        """
        Single validation step for model evaluation.

        Computes validation metrics including minADE, minFDE, and MR.

        Args:
            data: Validation batch or list of batches
            batch_idx: Index of current batch
        """
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        _, loss_dict = self.cal_loss(out, data, tag='val_')
        
        # Log losses
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
        
        # Compute metrics
        metrics = self.val_metrics(out, data['target'][:, 0])

        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        """
        Called at the end of validation epoch.
        Log expert utilization statistics.
        """
        pass  # Can add epoch-level statistics here if needed

    def on_test_start(self) -> None:
        """
        Initialize submission handler at the start of testing.
        """
        save_dir = Path("./submission")
        save_dir.mkdir(exist_ok=True)
        self.submission_handler = SubmissionAv2(save_dir=save_dir)

    def test_step(self, data, batch_idx) -> None:
        """
        Single test step for generating competition submissions.

        Args:
            data: Test batch or list of batches
            batch_idx: Index of current batch
        """
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        self.submission_handler.format_data(data, out["y_hat"], out["pi"])

    def on_test_end(self) -> None:
        """
        Finalize submission file after all test steps complete.
        """
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        """
        Configure optimizer and learning rate scheduler.

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
