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
        
        # Load balancing auxiliary loss weight
        self.load_balance_weight = 0.01
        
    def forward(self, mode_features):
        """
        Args:
            mode_features: [B, M, D] where M is num_modes (6)
        Returns:
            trajectories: [B, M, T, 2]
            scores: [B, M]
            aux_loss: load balancing loss
            expert_weights: [B, num_experts] for analysis
        """
        B, M, D = mode_features.shape
        assert M == self.num_modes, f"Expected {self.num_modes} modes, got {M}"
        
        # Use aggregated mode features as context for routing
        # Average pooling over all modes to capture global information
        context = mode_features.mean(dim=1)  # [B, D]
        
        # Get expert routing weights
        router_logits, router_probs = self.router(context)  # [B, num_experts]
        
        # Top-K sparse gating
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)  # [B, top_k]
        top_k_probs = F.softmax(top_k_logits, dim=-1)  # [B, top_k]
        
        # Initialize outputs
        trajectories = torch.zeros(B, M, self._future_len, 2, device=mode_features.device)
        scores = torch.zeros(B, M, device=mode_features.device)
        
        # Aggregate predictions from top-k experts
        for i in range(B):
            batch_traj = torch.zeros(M, self._future_len, 2, device=mode_features.device)
            batch_score = torch.zeros(M, device=mode_features.device)
            
            for k_idx in range(self.top_k):
                expert_idx = top_k_indices[i, k_idx]
                expert_weight = top_k_probs[i, k_idx]
                
                # Get prediction from expert
                expert = self.experts[expert_idx]
                expert_traj, expert_score = expert(mode_features[i:i+1])  # [1, M, T, 2], [1, M]
                
                # Weighted aggregation
                batch_traj += expert_weight * expert_traj.squeeze(0)
                batch_score += expert_weight * expert_score.squeeze(0)
            
            trajectories[i] = batch_traj
            scores[i] = batch_score
        
        # Compute load balancing loss for training
        # Encourage uniform expert utilization
        aux_loss = self._compute_load_balance_loss(router_probs)
        
        return trajectories, scores, aux_loss, router_probs
    
    def _compute_load_balance_loss(self, router_probs):
        """
        Compute load balancing auxiliary loss
        Encourages uniform expert utilization during training
        
        Args:
            router_probs: [B, num_experts]
        Returns:
            loss: scalar tensor
        """
        # Average probability per expert across batch
        avg_probs = router_probs.mean(dim=0)  # [num_experts]
        
        # Compute entropy to encourage uniform distribution
        # Higher entropy = more uniform utilization
        eps = 1e-8
        entropy = -(avg_probs * torch.log(avg_probs + eps)).sum()
        
        # We want to maximize entropy (uniform distribution)
        # So we minimize negative entropy
        load_balance_loss = -entropy * self.load_balance_weight
        
        # Alternative: L2 distance from uniform distribution
        uniform_dist = torch.ones_like(avg_probs) / self.num_experts
        l2_loss = F.mse_loss(avg_probs, uniform_dist)
        
        # Combine both losses
        total_aux_loss = load_balance_loss + 0.01 * l2_loss
        
        return total_aux_loss
    
    def get_expert_statistics(self, router_probs):
        """
        Get statistics about expert utilization for analysis
        
        Args:
            router_probs: [B, num_experts]
        Returns:
            dict with statistics
        """
        with torch.no_grad():
            avg_probs = router_probs.mean(dim=0)  # [num_experts]
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
