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
from .model_forecast_moe_supervised import ModelForecast


class Trainer(pl.LightningModule):
    """
    PyTorch Lightning trainer for DeMo motion forecasting with Supervised Mixture of Experts.
    
    Architecture:
    - 1 Shared Expert: learns general patterns from ALL data (30% weight)
    - 5 Unshared Experts: specialize in specific behaviors (70% weight, top-2 activation)
      * Expert 1: Lane Keeping (Straight/Lane Change)
      * Expert 2: Turn Left
      * Expert 3: Turn Right
      * Expert 4: Constraint-Driven Deceleration (Stop/Yield/Junction)
      * Expert 5: Others (Long-tail behaviors)
    
    Training strategy:
    - Shared expert trained on all samples
    - Unshared experts trained only on their assigned samples via supervised routing
    - Router supervised by ground truth expert labels
    """

    def __init__(
        self,
        model: dict,
        pretrained_weights: str = None,
        lr: float = 1e-3,
        warmup_epochs: int = 10,
        epochs: int = 60,
        weight_decay: float = 1e-4,
        router_loss_weight: float = 1.0,
        aux_loss_weight: float = 0.01,
        expert_loss_weight: float = 1.0,  # NEW: Weight for expert-specific supervision
    ) -> None:
        """
        Initialize the Supervised MoE trainer.

        Args:
            model: Dictionary containing model type and configuration
            pretrained_weights: Path to pretrained checkpoint file
            lr: Learning rate for optimizer
            warmup_epochs: Number of epochs for learning rate warmup
            epochs: Total number of training epochs
            weight_decay: Weight decay coefficient for regularization
            router_loss_weight: Weight for router supervision loss
            aux_loss_weight: Weight for load balancing auxiliary loss
            expert_loss_weight: Weight for expert-specific supervision loss (CRITICAL!)
        """
        super(Trainer, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.router_loss_weight = router_loss_weight
        self.aux_loss_weight = aux_loss_weight
        self.expert_loss_weight = expert_loss_weight  # NEW
        self.save_hyperparameters()
        self.submission_handler = SubmissionAv2()

        model_type = model.pop('type')
        assert model_type == 'ModelForecast', "Only ModelForecast is supported for Supervised MoE"

        self.net = ModelForecast(**model)

        if pretrained_weights is not None:
            self.net.load_from_checkpoint(pretrained_weights)

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
        
        # Class weights for imbalanced expert labels (based on inverse frequency)
        # Expert distribution: E1(45.97%), E2(7.68%), E3(6.96%), E4(10.34%), E5(29.05%)
        # Weights calculated as: total_samples / (num_classes * class_count)
        self.register_buffer(
            'class_weights',
            torch.tensor([0.4351, 2.6055, 2.8716, 1.9339, 0.6885], dtype=torch.float32)
        )
        
        # For tracking expert utilization (per-mode)
        self.expert_usage_stats = {i: 0 for i in range(5)}  # 5 unshared experts
        self.mode_expert_stats = {m: {i: 0 for i in range(5)} for m in range(6)}  # per-mode stats
        self.total_samples = 0
    
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
        Calculate comprehensive loss for Supervised MoE trajectory prediction.

        Loss components:
        - Regression loss for trajectory predictions
        - Classification loss for mode selection  
        - Loss for other agents in the scene
        - Router supervision loss (uses ground truth expert labels, soft supervision per-mode)
        - Load balancing auxiliary loss (encourages uniform expert utilization)
        - Expert-specific supervision loss (CRITICAL: direct gradient to each expert on its samples)

        Args:
            out: Model outputs containing predictions and routing information
            data: Ground truth data including target trajectories and expert labels
            tag: Prefix for loss logging

        Returns:
            Tuple of (total_loss, loss_dict) for logging
        """
        y_hat = out["y_hat"]  # [B, M, T, 2]
        pi = out["pi"]  # [B, M]
        y_hat_others = out["y_hat_others"]  # [B, N-1, T, 2]
        router_logits = out["router_logits"]  # [B, M, num_unshared_experts] - per-mode routing
        selected_experts = out["selected_experts"]  # [B, M, top_k]
        expert_labels = out["expert_labels"]  # [B] - ground truth expert IDs (1-5)
        aux_loss = out["aux_loss"]  # Load balancing loss
        expert_predictions = out["expert_predictions"]  # Dict[expert_idx -> (traj, score)]

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

        # ============ Router supervision loss (only for best mode) ============
        # Strategy: Only supervise the best mode's router to learn correct routing
        # This prevents mode homogenization and maintains diversity across modes
        router_loss = torch.tensor(0.0, device=y_hat.device)
        if expert_labels is not None:
            # Convert expert labels from 1-5 to 0-4
            expert_targets = (expert_labels - 1).long()  # [B]
            
            # Get batch size and number of modes
            B, M = y_hat.shape[0], y_hat.shape[1]
            
            # Only supervise the router for the best mode (closest to ground truth)
            # best_mode: [B] - index of the mode with minimum L2 distance to GT
            best_mode_router_logits = router_logits[torch.arange(B), best_mode, :]  # [B, num_experts]
            
            # Apply weighted cross-entropy loss with class weights to handle imbalance
            # class_weights: [num_experts] - inverse frequency weights for each expert class
            router_loss = F.cross_entropy(
                best_mode_router_logits, 
                expert_targets,
                weight=self.class_weights
            )
            router_loss = router_loss * self.router_loss_weight
            
            # Optional: Add entropy regularization to other modes to encourage diversity
            # Compute entropy for non-best modes to encourage exploration
            other_modes_mask = torch.ones(B, M, dtype=torch.bool, device=y_hat.device)
            other_modes_mask[torch.arange(B), best_mode] = False  # Exclude best mode
            
            if other_modes_mask.any():
                other_modes_logits = router_logits[other_modes_mask]  # [B*(M-1), num_experts]
                other_modes_probs = F.softmax(other_modes_logits, dim=-1)  # [B*(M-1), num_experts]
                
                # Entropy regularization: encourage uniform distribution for diversity
                eps = 1e-8
                entropy = -(other_modes_probs * torch.log(other_modes_probs + eps)).sum(dim=-1).mean()
                
                # We want to maximize entropy (uniform distribution), so minimize negative entropy
                diversity_loss = -entropy * 0.01  # Small weight for diversity
                router_loss = router_loss + diversity_loss

        # ============ Expert-specific supervision loss (CRITICAL!) ============
        # This is the key to making experts learn their specializations!
        # For each sample, we compute loss between the ground truth expert's prediction and GT trajectory
        expert_specific_loss = torch.tensor(0.0, device=y_hat.device)
        if expert_labels is not None and expert_predictions is not None:
            # Convert expert labels from 1-5 to 0-4
            expert_targets = (expert_labels - 1).long()  # [B]
            
            # For each sample in the batch, get its assigned expert's prediction
            B = y.shape[0]
            
            # Stack all expert predictions: [num_experts, B, M, T, 2]
            all_expert_trajs = torch.stack([expert_predictions[i][0] for i in range(5)], dim=0)
            
            # For each sample, select the trajectory from its assigned expert
            # expert_targets: [B] - which expert (0-4) is responsible for each sample
            # We want to get: assigned_expert_traj[b] = all_expert_trajs[expert_targets[b], b]
            
            # Method: Use advanced indexing
            # For each batch index b, we want all_expert_trajs[expert_targets[b], b, :, :, :]
            batch_indices = torch.arange(B, device=y.device)
            
            # Get predictions from assigned experts for all modes: [B, M, T, 2]
            assigned_expert_trajs = all_expert_trajs[expert_targets, batch_indices]  # [B, M, T, 2]
            
            # Find the best mode for each sample (same as before)
            # This ensures we train the expert on the mode that best matches GT
            l2_norm_expert = torch.norm(assigned_expert_trajs[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1)  # [B, M]
            best_mode_expert = torch.argmin(l2_norm_expert, dim=-1)  # [B]
            
            # Get the best mode's prediction from assigned expert
            assigned_expert_best = assigned_expert_trajs[batch_indices, best_mode_expert]  # [B, T, 2]
            
            # Compute regression loss between assigned expert's prediction and ground truth
            # This directly trains each expert on its assigned samples!
            expert_specific_loss = F.smooth_l1_loss(assigned_expert_best, y)
            expert_specific_loss = expert_specific_loss * self.expert_loss_weight
            
            # Optional: Also add classification loss for expert's score prediction
            # This helps the expert learn to be confident on its assigned samples
            all_expert_scores = torch.stack([expert_predictions[i][1] for i in range(5)], dim=0)  # [num_experts, B, M]
            assigned_expert_scores = all_expert_scores[expert_targets, batch_indices]  # [B, M]
            expert_cls_loss = F.cross_entropy(assigned_expert_scores, best_mode_expert.detach(), label_smoothing=0.2)
            expert_specific_loss = expert_specific_loss + 0.1 * expert_cls_loss  # Small weight for score loss

        # ============ Load balancing auxiliary loss ============
        weighted_aux_loss = aux_loss * self.aux_loss_weight

        # ============ Total loss ============
        total_loss = (
            agent_reg_loss + 
            agent_cls_loss + 
            others_reg_loss + 
            router_loss +
            weighted_aux_loss +
            expert_specific_loss  # NEW: Direct expert supervision!
        )

        # ============ Loss dictionary for logging ============
        disp_dict = {
            f"{tag}loss": total_loss.item(),
            f"{tag}reg_loss": agent_reg_loss.item(),
            f"{tag}cls_loss": agent_cls_loss.item(),
            f"{tag}others_reg_loss": others_reg_loss.item(),
            f"{tag}router_loss": router_loss.item(),
            f"{tag}aux_loss": weighted_aux_loss.item(),
            f"{tag}expert_loss": expert_specific_loss.item(),  # NEW
        }
        
        # Add expert utilization statistics (per-mode)
        if self.training and expert_labels is not None:
            with torch.no_grad():
                for b in range(selected_experts.shape[0]):
                    for m in range(selected_experts.shape[1]):
                        for k in range(selected_experts.shape[2]):
                            expert_idx = selected_experts[b, m, k].item()
                            self.expert_usage_stats[expert_idx] += 1
                            self.mode_expert_stats[m][expert_idx] += 1
                self.total_samples += selected_experts.shape[0] * selected_experts.shape[1] * selected_experts.shape[2]

        return total_loss, disp_dict
    
    def _compute_diversity_loss(self, expert_weights):
        """
        Compute diversity loss to encourage expert specialization.
        
        We want different samples and modes to use different experts, which encourages
        experts to specialize in different driving behaviors.
        
        Args:
            expert_weights: [B, M, num_experts] - expert selection probabilities for each mode
            
        Returns:
            diversity_loss: scalar tensor
        """
        # Flatten to [B*M, num_experts] to treat each (batch, mode) pair as independent sample
        B, M, num_experts = expert_weights.shape
        expert_weights_flat = expert_weights.view(B * M, num_experts)  # [B*M, num_experts]
        
        # Compute pairwise cosine similarity between expert weight distributions
        # High similarity means experts are being used similarly (bad)
        # Low similarity means experts are specialized (good)
        
        # Normalize expert weights along expert dimension
        expert_weights_norm = F.normalize(expert_weights_flat, p=2, dim=1)  # [B*M, num_experts]
        
        # Compute cosine similarity matrix [B*M, B*M]
        similarity_matrix = torch.mm(expert_weights_norm, expert_weights_norm.t())
        
        # We want low similarity (encourage diversity)
        # Remove diagonal (self-similarity is always 1)
        total_samples = B * M
        mask = ~torch.eye(total_samples, dtype=torch.bool, device=expert_weights.device)
        off_diagonal_sim = similarity_matrix[mask]
        
        # Penalize high similarity - want different samples/modes to use different experts
        diversity_loss = off_diagonal_sim.mean()
        
        return diversity_loss
    
    def _compute_expert_statistics(self, expert_weights):
        """
        Compute statistics about expert utilization for logging.
        
        Args:
            expert_weights: [B, M, num_experts] - expert selection probabilities for each mode
            
        Returns:
            dict with statistics
        """
        with torch.no_grad():
            # Flatten to [B*M, num_experts] to compute overall expert usage
            B, M, num_experts = expert_weights.shape
            expert_weights_flat = expert_weights.view(B * M, num_experts)
            
            # Average usage per expert across all samples and modes
            avg_usage = expert_weights_flat.mean(dim=0)  # [num_experts]
            
            # Find most and least used experts
            max_usage = avg_usage.max().item()
            min_usage = avg_usage.min().item()
            max_expert_idx = avg_usage.argmax().item()
            min_expert_idx = avg_usage.argmin().item()
            
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
                'most_used_expert': max_expert_idx,
                'least_used_expert': min_expert_idx,
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

    def on_train_epoch_end(self):
        """
        Called at the end of training epoch.
        Log expert utilization statistics per mode.
        """
        if self.total_samples > 0:
            # Overall expert usage
            print("\n" + "="*80)
            print(f"Epoch {self.current_epoch} Expert Utilization Statistics:")
            print("="*80)
            
            # Overall expert usage
            print("\nOverall Expert Usage:")
            for expert_idx in range(5):
                usage_pct = (self.expert_usage_stats[expert_idx] / self.total_samples) * 100
                print(f"  Expert {expert_idx} (Label {expert_idx+1}): {usage_pct:.2f}%")
                self.log(f"train/expert_{expert_idx}_usage", usage_pct, sync_dist=True)
            
            # Per-mode expert usage
            print("\nPer-Mode Expert Usage:")
            mode_samples = self.total_samples // 6  # Assuming 6 modes
            for m in range(6):
                print(f"\n  Mode {m}:")
                for expert_idx in range(5):
                    if mode_samples > 0:
                        usage_pct = (self.mode_expert_stats[m][expert_idx] / mode_samples) * 100
                        print(f"    Expert {expert_idx}: {usage_pct:.2f}%")
                        self.log(f"train/mode_{m}_expert_{expert_idx}_usage", usage_pct, sync_dist=True)
            
            print("="*80 + "\n")
            
            # Reset statistics for next epoch
            self.expert_usage_stats = {i: 0 for i in range(5)}
            self.mode_expert_stats = {m: {i: 0 for i in range(5)} for m in range(6)}
            self.total_samples = 0

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
