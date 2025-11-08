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



class GMMPredictor(nn.Module):
    """GMM (Gaussian Mixture Model) Predictor for trajectory prediction.
    
    Outputs trajectory predictions, confidence scores, and scale parameters
    for Laplace loss computation. This is the same predictor used in the
    original DeMo paper.
    """
    def __init__(self, future_len=60, dim=128):
        super(GMMPredictor, self).__init__()
        self._future_len = future_len
        
        # Trajectory prediction head (mean of Gaussian)
        self.gaussian = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self._future_len*2)
        )
        
        # Confidence score head
        self.score = nn.Sequential(
            nn.Linear(dim, 64), 
            nn.GELU(), 
            nn.Linear(64, 1),
        )
        
        # Scale parameter head (standard deviation of Gaussian)
        # Used for Laplace loss and uncertainty estimation
        self.scale = nn.Sequential(
            nn.Linear(dim, 256), 
            nn.GELU(), 
            nn.Linear(256, self._future_len*2)
        )
    
    def forward(self, input):
        """
        Args:
            input: [B, M, D] - batch, modes, dim
            
        Returns:
            res: [B, M, T, 2] - trajectory predictions
            score: [B, M] - confidence scores for each mode
            scal: [B, M, T, 2] - scale parameters for uncertainty
        """
        B, M, _ = input.shape
        
        # Predict trajectories [B, M, T, 2]
        res = self.gaussian(input).view(B, M, self._future_len, 2) 
        
        # Predict scale parameters [B, M, T, 2]
        # ELU activation ensures positivity: scale = elu(x) + 1.0 + eps
        scal = F.elu_(self.scale(input), alpha=1.0) + 1.0 + 0.0001
        scal = scal.view(B, M, self._future_len, 2) 
        
        # Predict confidence scores [B, M]
        score = self.score(input).squeeze(-1)

        return res, score, scal
    

class TimeDecoder(nn.Module):
    """Time Decoder with Mode Localization Module and GMM Predictor.
    
    This decoder only uses the Mode Localization Module from the original
    DeMo architecture, with a GMM predictor for trajectory prediction.
    The State Consistency Module and Hybrid Coupling Module are not included.
    """
    def __init__(self, future_len=60, dim=128):
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

        # mode query initialization (6 learnable mode queries)
        self.multi_modal_query_embedding = nn.Embedding(6, dim)
        self.register_buffer('modal', torch.arange(6).long())

        # GMM predictor for mode query
        self.predictor = GMMPredictor(future_len, dim)

    def forward(self, mode, encoding, mask=None):
        """Forward pass through Mode Localization Module with GMM predictor.
        
        Args:
            mode: Not used in this simplified version (for compatibility)
            encoding: [B, N+M, D] - encoded scene context (agents + lanes)
            mask: [B, N+M] - padding mask for scene elements
            
        Returns:
            y_hat: [B, K, T, 2] - predicted trajectories for K=6 modes
            pi: [B, K] - confidence scores for each mode
            scal: [B, K, T, 2] - scale parameters for Laplace loss
        """
        # Directional intention localization (Mode Localization Module only)
        multi_modal_query = self.multi_modal_query_embedding(self.modal) # [K, C]
        mode_query = encoding[:, 0] # [B, C] - use ego agent encoding
        mode = mode_query[:, None] + multi_modal_query # [B, K, C]

        # Mode cross attention: attend to scene context
        for blk in self.cross_block_mode:
            mode = blk(mode, encoding, key_padding_mask=mask)
        
        # Mode self attention: interaction between modes
        for blk in self.self_block_mode:
            mode = blk(mode)

        # GMM predictor: output trajectories, scores, and scales
        y_hat, pi, scal = self.predictor(mode)

        return y_hat, pi, scal
