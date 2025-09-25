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



class MLPPredictor(nn.Module):
    def __init__(self, future_len=60, dim=128):
        super(MLPPredictor, self).__init__()
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
        B, M, _ = input.shape
        # Predict trajectories [B, M, T, 2]
        trajectories = self.trajectory_head(input).view(B, M, self._future_len, 2) 
        # Predict confidence scores [B, M]
        scores = self.score_head(input).squeeze(-1)
        
        return trajectories, scores
    

class TimeDecoder(nn.Module):
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

        # mode query initialization
        self.multi_modal_query_embedding = nn.Embedding(6, dim)
        self.register_buffer('modal', torch.arange(6).long())

        # MLP for mode query
        self.predictor = MLPPredictor(future_len)

    def forward(self, mode, encoding, mask=None):
        # Directional intention localization (Mode Localization Module only)
        multi_modal_query = self.multi_modal_query_embedding(self.modal) # [K, C]
        mode_query = encoding[:, 0] # [B, C]
        mode = mode_query[:, None] + multi_modal_query # [B, K, C]

        for blk in self.cross_block_mode:
            mode = blk(mode, encoding, key_padding_mask=mask)
        for blk in self.self_block_mode:
            mode = blk(mode)

        y_hat, pi = self.predictor(mode)

        return y_hat, pi
