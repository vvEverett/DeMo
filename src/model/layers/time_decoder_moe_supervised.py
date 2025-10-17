import torch
import torch.nn as nn
from .transformer_blocks import Cross_Block, Block
import torch.nn.functional as F


class ExpertMLP(nn.Module):
    """Single Expert MLP for trajectory prediction"""
    def __init__(self, future_len=60, dim=128):
        super(ExpertMLP, self).__init__()
        self._future_len = future_len
        # Trajectory prediction MLP
        self.trajectory_head = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, self._future_len*2)
        )
        # Confidence score MLP
        self.score_head = nn.Sequential(
            nn.Linear(dim, 128), 
            nn.GELU(), 
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, input):
        """
        Args:
            input: [B, M, D] - batch, modes, dim
        Returns:
            trajectories: [B, M, T, 2]
            scores: [B, M]
        """
        B, M, _ = input.shape
        # Predict trajectories [B, M, T, 2]
        trajectories = self.trajectory_head(input).view(B, M, self._future_len, 2) 
        # Predict confidence scores [B, M]
        scores = self.score_head(input).squeeze(-1)
        
        return trajectories, scores


class SupervisedRouter(nn.Module):
    """
    Supervised MLP-based router for unshared expert selection
    Routes each mode independently to experts (like unsupervised MoE)
    Uses ground truth labels for supervision during training
    """
    def __init__(self, dim=128, num_unshared_experts=5):
        super(SupervisedRouter, self).__init__()
        self.num_unshared_experts = num_unshared_experts
        self.router = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_unshared_experts)
        )
        
    def forward(self, context):
        """
        Args:
            context: [B*M, D] - context features from all modes (flattened)
        Returns:
            router_logits: [B*M, num_unshared_experts] - expert selection logits
            router_probs: [B*M, num_unshared_experts] - expert selection probabilities
        """
        router_logits = self.router(context)  # [B*M, num_unshared_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        return router_logits, router_probs


class SupervisedMoEPredictor(nn.Module):
    """
    Supervised Mixture of Experts Predictor with:
    - 1 shared expert: trained on all data (30% weight)
    - 5 unshared experts: trained on specific data (70% weight, top-2 activation)
    - Supervised router: guided by ground truth expert labels, but routes each mode independently
    
    Expert definitions:
    - Expert 1: Lane Keeping (Straight/Lane Change)
    - Expert 2: Turn Left (from stop/moving)
    - Expert 3: Turn Right (from stop/moving)
    - Expert 4: Constraint-Driven Deceleration (Stop/Yield/Junction/Congestion)
    - Expert 5: Others (Long-tail behaviors)
    """
    def __init__(self, future_len=60, dim=128, num_modes=6, num_unshared_experts=5, 
                 top_k=2, shared_weight=0.3, load_balance_weight=0.01):
        super(SupervisedMoEPredictor, self).__init__()
        self._future_len = future_len
        self.num_modes = num_modes
        self.num_unshared_experts = num_unshared_experts
        self.top_k = top_k
        self.shared_weight = shared_weight
        self.unshared_weight = 1.0 - shared_weight
        self.load_balance_weight = load_balance_weight
        
        # Shared expert: learns general patterns from all data
        self.shared_expert = ExpertMLP(future_len, dim)
        
        # Unshared experts: specialize in specific driving behaviors
        # Expert 0: Lane Keeping (label 1)
        # Expert 1: Turn Left (label 2)
        # Expert 2: Turn Right (label 3)
        # Expert 3: Constraint-Driven Deceleration (label 4)
        # Expert 4: Others (label 5)
        self.unshared_experts = nn.ModuleList([
            ExpertMLP(future_len, dim) for _ in range(num_unshared_experts)
        ])
        
        # Supervised router for unshared expert selection (per-mode routing)
        self.router = SupervisedRouter(dim, num_unshared_experts)
        
    def forward(self, mode_features, expert_labels=None):
        """
        Args:
            mode_features: [B, M, D] where M is num_modes (6)
            expert_labels: [B] ground truth expert labels (1-5), None during inference
        Returns:
            trajectories: [B, M, T, 2]
            scores: [B, M]
            router_logits: [B, M, num_unshared_experts] for supervision
            selected_expert_indices: [B, M, top_k] selected unshared experts
            aux_loss: load balancing loss
            expert_predictions: Dict[expert_idx -> (traj, score)] for expert-specific supervision
        """
        B, M, D = mode_features.shape
        assert M == self.num_modes, f"Expected {self.num_modes} modes, got {M}"
        
        # CRITICAL: Route each mode independently (like unsupervised MoE)
        # Each of the M=6 modes should independently select which experts to use
        mode_features_flat = mode_features.view(B * M, D)  # [B*M, D]
        
        # Get router predictions for unshared experts (per-mode)
        router_logits, router_probs = self.router(mode_features_flat)  # [B*M, num_unshared_experts]
        
        # Reshape back to [B, M, num_unshared_experts]
        router_logits = router_logits.view(B, M, self.num_unshared_experts)
        router_probs = router_probs.view(B, M, self.num_unshared_experts)
        
        # Top-K sparse gating for unshared experts (per-mode)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # [B, M, top_k]
        top_k_probs = F.softmax(top_k_logits, dim=-1)  # [B, M, top_k]
        
        # ============ Get predictions from shared expert (30%) ============
        shared_traj, shared_score = self.shared_expert(mode_features)  # [B, M, T, 2], [B, M]
        
        # ============ Store individual expert predictions for expert-specific supervision ============
        # We need to get predictions from ALL unshared experts (not just top-k)
        # This allows us to compute expert-specific losses during training
        expert_predictions = {}
        
        # Get predictions from all unshared experts
        for expert_idx in range(self.num_unshared_experts):
            expert = self.unshared_experts[expert_idx]
            expert_traj, expert_score = expert(mode_features)  # [B, M, T, 2], [B, M]
            expert_predictions[expert_idx] = (expert_traj, expert_score)
        
        # ============ Get predictions from top-k unshared experts (70%) - SPARSE ACTIVATION ============
        # Strategy: Only compute selected top-k experts (sparse gating)
        # Flatten and batch process to reduce number of expert forward calls
        
        # Flatten top_k_indices and top_k_probs: [B, M, top_k] -> [B*M*top_k]
        top_k_indices_flat = top_k_indices.reshape(-1)  # [B*M*top_k]
        top_k_probs_flat = top_k_probs.reshape(-1)  # [B*M*top_k]
        
        # Create a mapping: for each unique expert, collect all (batch, mode) pairs that use it
        # This allows us to batch process inputs for each expert
        unshared_traj = torch.zeros(B, M, self._future_len, 2, device=mode_features.device)
        unshared_score = torch.zeros(B, M, device=mode_features.device)
        
        # Process each expert that has been selected by at least one (batch, mode) pair
        unique_experts = torch.unique(top_k_indices_flat)
        
        for expert_idx in unique_experts:
            # Find which (batch, mode, k) positions selected this expert
            # Create indices in the original [B, M, top_k] space
            mask = (top_k_indices == expert_idx)  # [B, M, top_k]
            
            # Get the positions where this expert is selected
            batch_indices, mode_indices, k_indices = torch.where(mask)
            
            if len(batch_indices) == 0:
                continue
                
            # Gather the corresponding mode features for this expert
            # mode_features[batch_indices, mode_indices] gives us all inputs for this expert
            expert_input = mode_features[batch_indices, mode_indices].unsqueeze(1)  # [N, 1, D]
            
            # Forward through the selected expert (single forward call per expert)
            expert = self.unshared_experts[expert_idx]
            expert_traj, expert_score = expert(expert_input)  # [N, 1, T, 2], [N, 1]
            expert_traj = expert_traj.squeeze(1)  # [N, T, 2]
            expert_score = expert_score.squeeze(1)  # [N]
            
            # Get the corresponding weights
            expert_weights = top_k_probs[batch_indices, mode_indices, k_indices]  # [N]
            
            # Accumulate weighted predictions back to the correct positions (vectorized)
            # Weight the expert predictions
            weighted_traj = expert_traj * expert_weights.unsqueeze(-1).unsqueeze(-1)  # [N, T, 2]
            weighted_score = expert_score * expert_weights  # [N]
            
            # Use index_add_ for efficient in-place accumulation
            # Convert 2D indices (batch, mode) to flat indices for unshared_traj and unshared_score
            flat_indices = batch_indices * M + mode_indices  # [N]
            
            # Flatten unshared_traj to [B*M, T, 2] for index_add_
            unshared_traj_flat = unshared_traj.view(B * M, self._future_len, 2)
            unshared_traj_flat.index_add_(0, flat_indices, weighted_traj)
            unshared_traj = unshared_traj_flat.view(B, M, self._future_len, 2)
            
            # Flatten unshared_score to [B*M] for index_add_
            unshared_score_flat = unshared_score.view(B * M)
            unshared_score_flat.index_add_(0, flat_indices, weighted_score)
            unshared_score = unshared_score_flat.view(B, M)
        
        # ============ Final prediction: 30% shared + 70% unshared ============
        final_traj = self.shared_weight * shared_traj + self.unshared_weight * unshared_traj
        final_score = self.shared_weight * shared_score + self.unshared_weight * unshared_score
        
        # ============ Compute load balancing loss ============
        aux_loss = self._compute_load_balance_loss(router_probs)
        
        return final_traj, final_score, router_logits, top_k_indices, aux_loss, expert_predictions
    
    def _compute_load_balance_loss(self, router_probs):
        """
        Compute load balancing auxiliary loss
        Encourages uniform expert utilization during training
        
        Args:
            router_probs: [B, M, num_unshared_experts] - routing probabilities for each mode
        Returns:
            loss: scalar tensor
        """
        # Compute average usage probability for each expert across all samples
        B, M, num_experts = router_probs.shape
        router_probs_flat = router_probs.view(B * M, num_experts)
        avg_probs = router_probs_flat.mean(dim=0)  # [num_experts] - average usage per expert
        
        # Compute entropy to encourage uniform distribution
        eps = 1e-8
        entropy = -(avg_probs * torch.log(avg_probs + eps)).sum()
        
        # We want to maximize entropy (uniform distribution)
        # So we minimize negative entropy
        load_balance_loss = -entropy * self.load_balance_weight
        
        # Alternative: L2 distance from uniform distribution
        uniform_dist = torch.ones_like(avg_probs) / num_experts
        l2_loss = F.mse_loss(avg_probs, uniform_dist)
        
        # Combine both losses
        total_aux_loss = load_balance_loss + 0.01 * l2_loss
        
        return total_aux_loss


class TimeDecoder(nn.Module):
    """
    Time Decoder with Supervised Mixture of Experts
    Uses 1 shared + 5 unshared experts with supervised routing
    Each mode independently routes to experts
    """
    def __init__(self, future_len=60, dim=128, num_unshared_experts=5, top_k=2, 
                 shared_weight=0.3, load_balance_weight=0.01):
        super(TimeDecoder, self).__init__()

        ###### Mode Localization Module ######
        # mode self attention
        self.self_block_mode = nn.ModuleList(
            Block()
            for i in range(3)
        )

        # mode cross attention
        self.cross_block_mode = nn.ModuleList(
            Cross_Block()
            for i in range(3)
        )

        # mode query initialization
        self.multi_modal_query_embedding = nn.Embedding(6, dim)
        self.register_buffer('modal', torch.arange(6).long())

        # Supervised MoE Predictor (with per-mode routing)
        self.predictor = SupervisedMoEPredictor(
            future_len=future_len,
            dim=dim,
            num_modes=6,
            num_unshared_experts=num_unshared_experts,
            top_k=top_k,
            shared_weight=shared_weight,
            load_balance_weight=load_balance_weight
        )

    def forward(self, mode, encoding, expert_labels=None, mask=None):
        """
        Args:
            mode: placeholder (not used, kept for compatibility)
            encoding: [B, N, C] - encoded scene features
            expert_labels: [B] ground truth expert labels (1-5), None during inference
            mask: [B, N] - attention mask
        Returns:
            y_hat: [B, M, T, 2] - predicted trajectories
            pi: [B, M] - mode probabilities
            router_logits: [B, M, num_unshared_experts] for supervision loss
            selected_experts: [B, M, top_k] selected unshared experts
            expert_labels: [B] ground truth expert labels (passed through)
            aux_loss: load balancing loss
            expert_predictions: Dict[expert_idx -> (traj, score)] for expert-specific supervision
        """
        # Directional intention localization (Mode Localization Module)
        multi_modal_query = self.multi_modal_query_embedding(self.modal)  # [K, C] K=6
        mode_query = encoding[:, 0]  # [B, C]
        mode = mode_query[:, None] + multi_modal_query  # [B, K, C]

        for blk in self.cross_block_mode:
            mode = blk(mode, encoding, key_padding_mask=mask)
        for blk in self.self_block_mode:
            mode = blk(mode)

        # Supervised MoE prediction (per-mode routing)
        y_hat, pi, router_logits, selected_experts, aux_loss, expert_predictions = self.predictor(mode, expert_labels)

        # Return expert_labels for trainer to use
        return y_hat, pi, router_logits, selected_experts, expert_labels, aux_loss, expert_predictions
