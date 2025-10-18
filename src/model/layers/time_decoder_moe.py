import torch
import torch.nn as nn
from .transformer_blocks import Cross_Block, Block
import torch.nn.functional as F
from .mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


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


class MoERouter(nn.Module):
    """
    MLP-based router for context-aware expert selection
    Uses context features to determine expert weights
    """
    def __init__(self, dim=128, num_experts=6):
        super(MoERouter, self).__init__()
        self.num_experts = num_experts
        self.router = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_experts)
        )
        
    def forward(self, context):
        """
        Args:
            context: [B, D] - context features from encoder
        Returns:
            router_logits: [B, num_experts] - expert selection logits
            router_probs: [B, num_experts] - expert selection probabilities
        """
        router_logits = self.router(context)  # [B, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        return router_logits, router_probs


class MoEPredictor(nn.Module):
    """
    Mixture of Experts Predictor with:
    - 6 experts: specialized for different driving behaviors
    - Top-2 expert activation with sparse gating
    - MLP router for context-aware expert selection
    - Unsupervised learning of driving patterns
    """
    def __init__(self, future_len=60, dim=128, num_modes=6, num_experts=6, top_k=2):
        super(MoEPredictor, self).__init__()
        self._future_len = future_len
        self.num_modes = num_modes
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Expert network: 6 specialized experts
        # Experts 0-5: specialized for different driving behaviors
        self.experts = nn.ModuleList([
            ExpertMLP(future_len, dim) for _ in range(num_experts)
        ])
        
        # Context-aware router
        self.router = MoERouter(dim, num_experts)
        
    def forward(self, mode_features):
        """
        Args:
            mode_features: [B, M, D] where M is num_modes (6)
        Returns:
            trajectories: [B, M, T, 2]
            scores: [B, M]
            aux_loss: load balancing loss
            expert_weights: [B, M, num_experts] for analysis
        """
        B, M, D = mode_features.shape
        assert M == self.num_modes, f"Expected {self.num_modes} modes, got {M}"
        
        # CRITICAL: Route each mode independently to experts
        # Each of the M=6 modes (e.g., go straight, turn left, turn right, etc.) 
        # should independently select which experts to use based on its own features.
        # 
        # WHY reshape to [B*M, D]?
        # - Conceptually: we want to route each mode separately (like a loop over M modes)
        # - Efficiently: reshape to [B*M, D] allows batch processing all B*M samples at once
        # - Example: B=2, M=6 -> [12, D] where rows 0-5 are batch0's 6 modes, 
        #                                      rows 6-11 are batch1's 6 modes
        # - Router processes each row independently -> each mode gets its own routing decision
        mode_features_flat = mode_features.view(B * M, D)  # [B*M, D]
        
        # Get expert routing weights for each mode independently
        # Router treats each of the B*M samples as independent inputs
        router_logits, router_probs = self.router(mode_features_flat)  # [B*M, num_experts]
        
        # Reshape back to [B, M, num_experts] for easier indexing
        # Now router_probs[i, m, :] contains expert weights for batch i, mode m
        router_logits = router_logits.view(B, M, self.num_experts)  # [B, M, num_experts]
        router_probs = router_probs.view(B, M, self.num_experts)  # [B, M, num_experts]
        
        # Top-K sparse gating for each mode
        # Select top-k experts for each mode independently
        # Shape: [B, M, top_k] - each mode gets its own top-k expert indices and weights
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # [B, M, top_k]
        top_k_probs = F.softmax(top_k_logits, dim=-1)  # [B, M, top_k]
        
        # ======== OPTIMIZED PARALLEL EXPERT COMPUTATION ========
        # Instead of nested loops, we use batch operations for GPU parallelization
        
        # Step 1: Get predictions from ALL experts for ALL mode features in parallel
        # Expand mode_features_flat to [B*M, num_experts, D] by repeating
        mode_features_expanded = mode_features_flat.unsqueeze(1).expand(
            B * M, self.num_experts, D
        )  # [B*M, num_experts, D]
        
        # Reshape to [B*M*num_experts, D] to batch process through experts
        mode_features_for_experts = mode_features_expanded.reshape(B * M * self.num_experts, D)
        
        # Process all expert inputs in parallel
        # We need to call each expert separately, but can batch within each expert
        all_trajectories = []
        all_scores = []
        for expert_idx in range(self.num_experts):
            # Extract features for this expert: every num_experts-th element
            expert_features = mode_features_for_experts[expert_idx::self.num_experts]  # [B*M, D]
            expert_features = expert_features.unsqueeze(1)  # [B*M, 1, D] for expert input format
            
            # Get predictions from this expert for all B*M samples
            expert_traj, expert_score = self.experts[expert_idx](expert_features)  # [B*M, 1, T, 2], [B*M, 1]
            all_trajectories.append(expert_traj.squeeze(1))  # [B*M, T, 2]
            all_scores.append(expert_score.squeeze(1))  # [B*M]
        
        # Stack to get [B*M, num_experts, T, 2] and [B*M, num_experts]
        all_expert_traj = torch.stack(all_trajectories, dim=1)  # [B*M, num_experts, T, 2]
        all_expert_scores = torch.stack(all_scores, dim=1)  # [B*M, num_experts]
        
        # Reshape back to [B, M, num_experts, T, 2] and [B, M, num_experts]
        all_expert_traj = all_expert_traj.view(B, M, self.num_experts, self._future_len, 2)
        all_expert_scores = all_expert_scores.view(B, M, self.num_experts)
        
        # Step 2: Build sparse gating weights [B, M, num_experts]
        sparse_weights = torch.zeros(B, M, self.num_experts, device=mode_features.device)
        
        # Fill in top-k weights using advanced indexing
        batch_indices = torch.arange(B, device=mode_features.device).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        mode_indices = torch.arange(M, device=mode_features.device).unsqueeze(0).unsqueeze(2)  # [1, M, 1]
        sparse_weights[batch_indices, mode_indices, top_k_indices] = top_k_probs  # [B, M, top_k]
        
        # Step 3: Weighted aggregation using tensor operations
        # Expand sparse_weights for trajectory dimension: [B, M, num_experts, 1, 1]
        weights_for_traj = sparse_weights.unsqueeze(-1).unsqueeze(-1)
        
        # Weighted sum of trajectories: [B, M, T, 2]
        trajectories = (all_expert_traj * weights_for_traj).sum(dim=2)
        
        # Weighted sum of scores: [B, M]
        scores = (all_expert_scores * sparse_weights).sum(dim=2)
        
        # Compute load balancing loss for training
        # Encourage uniform expert utilization across all modes and batches
        aux_loss = self._compute_load_balance_loss(router_probs)
        
        return trajectories, scores, aux_loss, router_probs
    
    def _compute_load_balance_loss(self, router_probs):
        """
        Compute load balancing auxiliary loss
        Encourages uniform expert utilization during training
        
        Args:
            router_probs: [B, M, num_experts] - routing probabilities for each mode
        Returns:
            loss: scalar tensor (raw loss without weighting - weighting done in trainer)
        """
        # Compute average usage probability for each expert across all samples
        # Flatten to [B*M, num_experts] to treat all batch*mode samples equally
        # Then average over the B*M dimension to get per-expert utilization
        B, M, num_experts = router_probs.shape
        router_probs_flat = router_probs.view(B * M, num_experts)
        avg_probs = router_probs_flat.mean(dim=0)  # [num_experts] - average usage per expert
        
        # This is a widely used load balancing loss formulation for MoE
        # It's the product of the number of experts and the sum of squares of the expert probabilities
        # This encourages the expert probabilities to be close to uniform.
        aux_loss = self.num_experts * torch.sum(avg_probs * avg_probs)
        
        return aux_loss
    
    def get_expert_statistics(self, router_probs):
        """
        Get statistics about expert utilization for analysis
        Useful for debugging and understanding which experts are being used
        
        Args:
            router_probs: [B, M, num_experts] - routing probabilities for each mode
        Returns:
            dict with statistics about expert usage patterns
        """
        with torch.no_grad():
            # Average usage across all batch*mode samples
            B, M, num_experts = router_probs.shape
            router_probs_flat = router_probs.view(B * M, num_experts)
            avg_probs = router_probs_flat.mean(dim=0)  # [num_experts]
            max_expert = torch.argmax(avg_probs).item()
            min_expert = torch.argmin(avg_probs).item()
            
            stats = {
                'expert_probs': avg_probs.cpu().numpy(),
                'most_used_expert': max_expert,
                'least_used_expert': min_expert,
                'expert_utilization_std': avg_probs.std().item()
            }
        return stats


class TimeDecoder(nn.Module):
    """
    Time Decoder with Mixture of Experts
    Replaces standard MLP predictor with MoE for learning diverse driving patterns
    """
    def __init__(self, future_len=60, dim=128, num_experts=6, top_k=2):
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

        # MoE Predictor instead of MLP
        self.predictor = MoEPredictor(
            future_len=future_len,
            dim=dim,
            num_modes=6,
            num_experts=num_experts,
            top_k=top_k
        )

    def forward(self, mode, encoding, mask=None):
        """
        Args:
            mode: placeholder (not used, kept for compatibility)
            encoding: [B, N, C] - encoded scene features
            mask: [B, N] - attention mask
        Returns:
            y_hat: [B, M, T, 2] - predicted trajectories
            pi: [B, M] - mode probabilities
            aux_loss: load balancing loss
            expert_weights: [B, num_experts]
        """
        # Directional intention localization (Mode Localization Module)
        multi_modal_query = self.multi_modal_query_embedding(self.modal)  # [K, C] K=6
        mode_query = encoding[:, 0]  # [B, C]
        mode = mode_query[:, None] + multi_modal_query  # [B, K, C]

        for blk in self.cross_block_mode:
            mode = blk(mode, encoding, key_padding_mask=mask)
        for blk in self.self_block_mode:
            mode = blk(mode)

        # MoE prediction with auxiliary loss
        y_hat, pi, aux_loss, expert_weights = self.predictor(mode)

        return y_hat, pi, aux_loss, expert_weights
