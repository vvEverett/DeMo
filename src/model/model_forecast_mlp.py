from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers.lane_embedding import LaneEmbeddingLayer
from .layers.transformer_blocks import Block, InteractionBlock
from .layers.time_decoder_mlp import TimeDecoder
from .layers.mamba.vim_mamba import init_weights, create_block
from functools import partial
from timm.models.layers import DropPath, to_2tuple
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


# only 'DeMo'
class ModelForecast(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_path=0.2,
        future_steps: int = 60,
    ) -> None:
        super().__init__()

        self.hist_embed_mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

        # Agent Encoding Mamba
        self.hist_embed_mamba = nn.ModuleList(  
            [
                create_block(  
                    d_model=embed_dim,
                    layer_idx=i,
                    drop_path=0.2,  
                    bimamba=False,  
                    rms_norm=True,  
                )
                for i in range(4)
            ]
        )
        self.norm_f = RMSNorm(embed_dim, eps=1e-5)
        self.drop_path = DropPath(drop_path)

        self.lane_embed = LaneEmbeddingLayer(3, embed_dim)

        self.pos_embed = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Scene Context Transformer
        self.blocks = nn.ModuleList(
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=0.2,
            )
            for i in range(5)
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.actor_type_embed = nn.Parameter(torch.Tensor(4, embed_dim))
        self.lane_type_embed = nn.Parameter(torch.Tensor(3, embed_dim))

        self.dense_predictor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Linear(256, future_steps * 2)
        )

        self.time_decoder = TimeDecoder()

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.actor_type_embed, std=0.02)
        nn.init.normal_(self.lane_type_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def load_from_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("net.") :]: v for k, v in ckpt.items() if k.startswith("net.")
        }
        
        # Filter out parameters from removed modules (State Consistency and Hybrid query)
        filtered_state_dict = {}
        removed_keys = []
        
        for k, v in state_dict.items():
            # Skip State Consistency Module parameters
            if any(x in k for x in ['time_embedding_mlp', 'timequery_embed_mamba', 'timequery_norm_f', 
                                   'timequery_drop_path', 'dense_predict', 'cross_block_time']):
                removed_keys.append(k)
                continue
            # Skip Hybrid query parameters  
            if any(x in k for x in ['self_block_dense', 'cross_block_dense', 'self_block_different_mode',
                                   'dense_embed_mamba', 'dense_norm_f', 'dense_drop_path', 'predictor_dense']):
                removed_keys.append(k)
                continue
            # Keep other parameters
            filtered_state_dict[k] = v
        
        print(f"Loading checkpoint from {ckpt_path}")
        print(f"Removed {len(removed_keys)} parameters from removed modules:")
        for key in removed_keys[:10]:  # Print first 10 removed keys
            print(f"  - {key}")
        if len(removed_keys) > 10:
            print(f"  ... and {len(removed_keys) - 10} more")
        
        missing_keys, unexpected_keys = self.load_state_dict(state_dict=filtered_state_dict, strict=False)
        
        if missing_keys:
            print(f"Missing keys (newly initialized): {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {unexpected_keys}")
            
        print('Pretrained weights have been loaded (Mode Localization Module only).')
        return self

    def forward(self, data):
        ###### Scene context encoding ###### 
        # agent encoding
        hist_valid_mask = data["x_valid_mask"]
        hist_key_valid_mask = data["x_key_valid_mask"]
        hist_feat = torch.cat(
            [
                data["x_positions_diff"], # (B, N, L, 2)  x_positions_diff: torch.Size([16, 48, 50, 2])
                data["x_velocity_diff"][..., None], # x_velocity_diff: torch.Size([16, 48, 50])
                hist_valid_mask[..., None], # x_valid_mask: torch.Size([16, 48, 50])
            ],
            dim=-1,
        ) # (B, N, L, D) = (batch_size, num_agents, hist_length, feature_dim) = [16, 48, 50, 4]

        B, N, L, D = hist_feat.shape
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_valid = hist_key_valid_mask.view(B * N)

        # unidirectional mamba (For agent history encoding)
        actor_feat = self.hist_embed_mlp(hist_feat[hist_feat_key_valid].contiguous())
        residual = None
        for blk_mamba in self.hist_embed_mamba:
            actor_feat, residual = blk_mamba(actor_feat, residual)
        fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
        actor_feat = fused_add_norm_fn(
            self.drop_path(actor_feat),
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=True  
        )

        actor_feat = actor_feat[:, -1]
        actor_feat_tmp = torch.zeros(
            B * N, actor_feat.shape[-1], device=actor_feat.device
        )
        actor_feat_tmp[hist_feat_key_valid] = actor_feat
        actor_feat = actor_feat_tmp.view(B, N, actor_feat.shape[-1])

        # map encoding (Using PointNet)
        lane_valid_mask = data["lane_valid_mask"]
        lane_normalized = data["lane_positions"] - data["lane_centers"].unsqueeze(-2)
        lane_normalized = torch.cat(
            [lane_normalized, lane_valid_mask[..., None]], dim=-1
        )
        B, M, L, D = lane_normalized.shape
        lane_feat = self.lane_embed(lane_normalized.view(-1, L, D).contiguous())
        lane_feat = lane_feat.view(B, M, -1)

        # type embedding and position embedding
        x_centers = torch.cat([data["x_centers"], data["lane_centers"]], dim=1)
        angles = torch.cat([data["x_angles"][:, :, -1], data["lane_angles"]], dim=1)
        x_angles = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        pos_feat = torch.cat([x_centers, x_angles], dim=-1)
        pos_embed = self.pos_embed(pos_feat)

        actor_type_embed = self.actor_type_embed[data["x_attr"][..., 2].long()]
        lane_type_embed = self.lane_type_embed[data["lane_attr"][..., 0].long()]
        actor_feat += actor_type_embed
        lane_feat += lane_type_embed

        # scene context features
        x_encoder = torch.cat([actor_feat, lane_feat], dim=1)
        key_valid_mask = torch.cat(
            [data["x_key_valid_mask"], data["lane_key_valid_mask"]], dim=1
        )

        x_encoder = x_encoder + pos_embed

        #  intra-interaction learning for scene context features (Transformer encoder)
        for blk in self.blocks:
            x_encoder = blk(x_encoder, key_padding_mask=~key_valid_mask)
        x_encoder = self.norm(x_encoder)

        ###### Trajectory decoding with Mode Localization Module only ###### 
        # outputs of other agents
        x_others = x_encoder[:, 1:N]
        y_hat_others = self.dense_predictor(x_others).view(B, x_others.size(1), -1, 2)

        # decoder module with Mode Localization Module only
        y_hat, pi = self.time_decoder(None, x_encoder, mask=~key_valid_mask)

        ret_dict = {
            "y_hat": y_hat,  # trajectory output from Mode Localization Module
            "pi": pi,  # probability output from Mode Localization Module

            "y_hat_others": y_hat_others,  # trajectory of other agents

            # Set unused outputs to None for compatibility
            "dense_predict": None,  
            "new_y_hat": None,  
            "new_pi": None,     
            "scal": None,  # No scale output for MLP predictor
            "scal_new": None,  
        }

        return ret_dict
        