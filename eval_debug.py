import os
import sys
import hydra
from hydra.core.hydra_config import HydraConfig
import pytorch_lightning as pl
from hydra.utils import instantiate, to_absolute_path
import torch

# Add current working directory to PYTHONPATH
sys.path.insert(0, os.getcwd())

from src.model.trainer_forecast import Trainer  # LightningModule wrapper
# ModelForecast / Av2DataModule are instantiated via Hydra now

@hydra.main(version_base=None, config_path="./conf/", config_name="config")
def debug_dimensions(conf):
    pl.seed_everything(conf.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[Warning] CUDA not available. Triton fused LayerNorm may fail on CPU.")

    checkpoint = to_absolute_path("ckpts/DeMo.ckpt")
    if not os.path.exists(checkpoint):
        print(f"Checkpoint not found: {checkpoint}")
        print("You need to download the pretrained model or specify another checkpoint path.")
        return
    print(f"Found checkpoint: {checkpoint}")

    print("Instantiating datamodule & model via Hydra config ...")
    datamodule: pl.LightningDataModule = instantiate(conf.datamodule.target, test=conf.test)
    model: pl.LightningModule = instantiate(conf.model.target)

    datamodule.setup(stage="validate")

    try:
        ckpt = torch.load(checkpoint, map_location=device)
        if "state_dict" in ckpt:
            missing, unexpected = model.load_state_dict(ckpt["state_dict"], strict=False)
            print("Loaded checkpoint state_dict into LightningModule (strict=False)")
            if missing:
                print(f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")
        else:
            print("No 'state_dict' key in checkpoint file.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        print("Proceeding with randomly initialized weights.")

    model.to(device).eval()
    net = getattr(model, 'net', model).to(device).eval()

    val_loader = datamodule.val_dataloader()
    loader_iter = iter(val_loader)
    batches = []
    for _ in range(2):  # collect two batches for two inference runs
        try:
            batches.append(next(loader_iter))
        except StopIteration:
            break

    stats_list = []
    for idx, batch in enumerate(batches):
        data = batch[0] if isinstance(batch, (list, tuple)) and isinstance(batch[0], dict) else batch
        run_name = f"Run-{idx+1}"
        stats = debug_model_dimensions(net, data, device, run_name)
        stats_list.append(stats)

    if len(stats_list) == 2:
        print("\n" + "#"*60)
        print("COMPARISON SUMMARY (Run-1 vs Run-2)")
        print("#"*60)
        def fmt(a, b):
            return f"{a} | {b} (Δ={b-a})"
        print(f"Total agents (N):        {fmt(stats_list[0]['N'], stats_list[1]['N'])}")
        print(f"Valid key agents count:  {fmt(stats_list[0]['valid_key_agents'], stats_list[1]['valid_key_agents'])}")
        print(f"Hist length (L):         {fmt(stats_list[0]['L'], stats_list[1]['L'])}")
    else:
        print("Only one batch available; cannot compare two runs.")

def move_batch_to_device(data, device):
    if isinstance(data, dict):
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
    return data

def debug_model_dimensions(net, data, device, run_name="Run-1"):
    print("\n" + "="*60)
    print(f"{run_name} - INPUT DATA DIMENSIONS")
    print("="*60)
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            print(f"{key:25}: {tuple(value.shape)}  dtype={value.dtype} device={value.device}")
        elif isinstance(value, list):
            print(f"{key:25}: list(len={len(value)})")
        else:
            print(f"{key:25}: {type(value)}")

    print("\n" + "="*60)
    print(f"{run_name} - STEP-BY-STEP DIMENSION TRACKING")
    print("="*60)

    stats = {}
    with torch.no_grad():
        # -------- Agent feature construction (mirrors internal logic) --------
        print("\nStep 1: Agent Encoding (feature assembly)")
        hist_valid_mask = data["x_valid_mask"]
        hist_key_valid_mask = data["x_key_valid_mask"]
        hist_feat = torch.cat([
            data["x_positions_diff"],
            data["x_velocity_diff"][..., None],
            hist_valid_mask[..., None],
        ], dim=-1)
        B, N, L, D = hist_feat.shape
        stats.update({"B": B, "N": N, "L": L})
        hist_feat = hist_feat.view(B * N, L, D)
        hist_feat_key_valid = hist_key_valid_mask.view(B * N)
        valid_key_agents = int(hist_feat_key_valid.sum().item())
        stats["valid_key_agents"] = valid_key_agents
        print(f"hist_feat (B,N,L,D): {hist_feat.shape} -> B={B} N={N} L={L} D={D}")
        print(f"Valid key agents: {valid_key_agents}")

        print("\nStep 2: MLP + Mamba Encoding (using net components)")
        # Access components from underlying net
        valid_hist_feat = hist_feat[hist_feat_key_valid]
        print(f"valid_hist_feat: {valid_hist_feat.shape}")
        valid_hist_feat_device = valid_hist_feat.to(device)
        try:
            hist_embed_mlp = getattr(net, 'hist_embed_mlp')
            hist_embed_mamba = getattr(net, 'hist_embed_mamba')
        except AttributeError:
            print("Net missing 'hist_embed_mlp' / 'hist_embed_mamba'.")
            hist_embed_mlp = None
            hist_embed_mamba = None
        if hist_embed_mlp is not None:
            actor_feat = hist_embed_mlp(valid_hist_feat_device)
            print(f"actor_feat after MLP: {actor_feat.shape}")
            if hist_embed_mamba is not None:
                print(f"Configured Mamba blocks: {len(hist_embed_mamba)}")
        # -------- Full forward pass through net --------
        print("\nFull Forward Pass (net)")
        try:
            data_device = move_batch_to_device(data, device)
            output = net(data_device)
            print("\n" + "="*60)
            print(f"{run_name} - OUTPUT DIMENSIONS")
            print("="*60)
            if isinstance(output, dict):
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        print(f"{key:25}: {tuple(value.shape)} dtype={value.dtype} device={value.device}")
                    else:
                        print(f"{key:25}: {type(value)}")
            elif isinstance(output, torch.Tensor):
                print(f"output: {tuple(output.shape)} dtype={output.dtype} device={output.device}")
            else:
                print(f"Output type: {type(output)}")
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
    return stats

if __name__ == "__main__":
    debug_dimensions()