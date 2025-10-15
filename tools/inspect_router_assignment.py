#!/usr/bin/env python3
"""Inspect the supervised MoE router assignments for a single AV2 scenario.

This utility loads one processed `.pt` file, runs it through the trained
Mixture-of-Experts predictor, and prints how the router distributes the modes
across the supervised experts together with their gating weights.

Usage:
    python tools/inspect_router_assignment.py \
        --pt data/DeMo_processed/val/XXXXX.pt \
        --checkpoint ckpts/.../checkpoints/epoch=XX.ckpt

By default the script relies on the global constants defined below. Override
any of them via command-line flags when needed.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

# Ensure repository root is on the import path when running as a script
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datamodule.av2_dataset import Av2Dataset, collate_fn
from src.model.trainer_forecast_moe_supervised import Trainer as MoETrainer


# === Global configuration ====================================================
# Update these defaults to match your environment when running without CLI args.
PT_FILE_PATH = Path("/data1/xiaowei/code/DeMo/data/DeMo_processed/train/scenario_00052803-b0e6-4d8a-8abe-d8870d7c4132.pt")
"""Path to the processed AV2 `.pt` file you want to inspect."""

CHECKPOINT_PATH: Optional[Path] = Path(
    "/data1/xiaowei/code/DeMo/ckpts/SupervisedMoEv1/checkpoints/last.ckpt"
)
"""Lightning checkpoint containing trained supervised MoE weights (optional)."""

CLASSIFICATION_CSV_PATH: Optional[Path] = Path(
    "data/DeMo_classified/heuristic_classifications_latest.csv"
)
"""CSV with heuristic expert labels; disable by setting to `None`."""

MODEL_CONFIG_PATH = Path("conf/config_moe_supervised.yaml")
"""Fallback Hydra config (top-level) used when no checkpoint override is supplied."""

DATASET_DEFAULTS: Dict[str, Any] = {
    "num_historical_steps": 50,
    "sequence_origins": [50],
    "radius": 150.0,
    "train_mode": "only_focal",
}

EXPERT_NAMES: Tuple[str, ...] = (
    "Lane Keeping",
    "Turn Left",
    "Turn Right",
    "Constraint-Driven Deceleration",
    "Others",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === Helper dataclasses ======================================================
@dataclass
class RouterSummary:
    mode_probs: torch.Tensor
    router_probs: torch.Tensor
    selected_experts: torch.Tensor
    topk_weights: torch.Tensor
    shared_weight: float
    unshared_weight: float
    expert_label: Optional[int]
    scenario_id: Optional[str]
    agent_track_id: Optional[int]
    timestamp_s: Optional[float]


# === Core logic ==============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect supervised MoE router assignment for a single sample",
    )
    parser.add_argument("--pt", type=str, default=None, help="Path to processed .pt file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained Lightning checkpoint (.ckpt). Optional but recommended.",
    )
    parser.add_argument(
        "--classification",
        type=str,
        default=None,
        help="Optional CSV with expert labels. Overrides global default.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Model config YAML (used only when checkpoint is absent).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (default: auto-detect).",
    )
    return parser.parse_args()


def _resolve_path(arg_value: Optional[str], default_path: Optional[Path]) -> Optional[Path]:
    if arg_value is None:
        return default_path
    if arg_value.lower() in {"none", ""}:
        return None
    return Path(arg_value)


def _validate_existing_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def _config_name_from_path(config_path: Path) -> str:
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    try:
        relative = config_path.relative_to((REPO_ROOT / "conf").resolve())
    except ValueError as exc:
        raise ValueError(
            f"Config path must reside under {REPO_ROOT / 'conf'} (got {config_path})."
        ) from exc
    return relative.with_suffix("").as_posix()


def _instantiate_module_from_config(config_path: Path) -> MoETrainer:
    config_name = _config_name_from_path(config_path)

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "conf")):
        cfg = compose(config_name=config_name)

    if "model" not in cfg or "target" not in cfg.model:
        raise KeyError(
            f"Config '{config_name}' does not contain 'model.target' definition required for instantiation."
        )

    module = instantiate(cfg.model.target)
    return module


def load_lightning_module(
    checkpoint_path: Optional[Path],
    config_path: Optional[Path],
    device: torch.device,
) -> MoETrainer:
    """Instantiate the Lightning module either from checkpoint or config."""
    if config_path is None:
        raise ValueError("A model config path is required to instantiate the module.")

    config_path = _validate_existing_path(config_path, "Model config").resolve()
    module = _instantiate_module_from_config(config_path)

    if checkpoint_path is not None:
        checkpoint_path = _validate_existing_path(checkpoint_path, "Checkpoint")
        # Load checkpoint (contains model state dict and training metadata)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        missing, unexpected = module.load_state_dict(state_dict, strict=False)
        print(f"Loaded weights from checkpoint: {checkpoint_path}")
        if missing:
            print(f"  Missing keys ({len(missing)}): {missing}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): {unexpected}")
    else:
        print(f"Instantiated model from config (no checkpoint): {config_path}")

    module.eval()
    module.freeze()
    module.to(device)
    return module


def _derive_dataset_root(pt_path: Path) -> Tuple[Path, str]:
    split = pt_path.parent.name
    if split not in {"train", "val", "test"}:
        raise ValueError(
            f"Could not infer data split from parent folder '{split}'. Expected train/val/test."
        )
    data_root = pt_path.parent.parent
    return data_root, split


def build_dataset_for_file(
    pt_path: Path,
    classification_csv: Optional[Path],
    dataset_defaults: Dict[str, Any],
) -> Tuple[Av2Dataset, int]:
    data_root, split = _derive_dataset_root(pt_path)
    dataset_kwargs = deepcopy(dataset_defaults)
    dataset_kwargs["classification_csv"] = classification_csv

    dataset = Av2Dataset(
        data_root=data_root,
        split=split,
        **dataset_kwargs,
    )

    try:
        file_index = dataset.file_list.index(pt_path)
    except ValueError as exc:
        raise ValueError(
            f"File {pt_path} was not found inside dataset root {data_root / split}."
        ) from exc

    return dataset, file_index


def load_single_batch(
    pt_path: Path,
    classification_csv: Optional[Path],
    dataset_defaults: Dict[str, Any],
) -> Dict[str, Any]:
    dataset, idx = build_dataset_for_file(pt_path, classification_csv, dataset_defaults)
    sequence_data = dataset[idx]  # List of dicts (one per sequence origin)
    batch_list = collate_fn([sequence_data])
    batch = batch_list[-1]  # Match training convention
    return batch


def _tensor_to_python(val: torch.Tensor) -> Any:
    if val.numel() == 1:
        return val.item()
    return val.cpu().numpy()


def extract_router_summary(
    module: MoETrainer,
    batch: Dict[str, Any],
    device: torch.device,
) -> RouterSummary:
    data = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            data[key] = value.to(device)
        else:
            data[key] = value

    with torch.no_grad():
        output = module(data)

    router_logits = output["router_logits"].detach().cpu()
    mode_logits = output["pi"].detach().cpu()
    selected_experts = output["selected_experts"].detach().cpu()

    mode_probs = torch.softmax(mode_logits, dim=-1)
    router_probs = torch.softmax(router_logits, dim=-1)

    gathered_logits = torch.gather(router_logits, dim=-1, index=selected_experts)
    topk_weights = torch.softmax(gathered_logits, dim=-1)

    predictor = module.net.time_decoder.predictor
    shared_weight = float(predictor.shared_weight)
    unshared_weight = float(predictor.unshared_weight)

    expert_label_tensor = output.get("expert_labels", None)
    expert_label: Optional[int]
    if isinstance(expert_label_tensor, torch.Tensor) and expert_label_tensor.numel() > 0:
        expert_label = int(expert_label_tensor[0].item())
    else:
        expert_label = None

    scenario_id = batch.get("scenario_id", [None])[0]
    track_id = batch.get("track_id", [None])[0]
    timestamp_tensor = batch.get("timestamp", None)
    timestamp_s: Optional[float] = None
    if isinstance(timestamp_tensor, torch.Tensor) and timestamp_tensor.numel() > 0:
        timestamp_s = float(timestamp_tensor[0].item())

    return RouterSummary(
        mode_probs=mode_probs,
        router_probs=router_probs,
        selected_experts=selected_experts,
        topk_weights=topk_weights,
        shared_weight=shared_weight,
        unshared_weight=unshared_weight,
        expert_label=expert_label,
        scenario_id=scenario_id,
        agent_track_id=track_id,
        timestamp_s=timestamp_s,
    )


def _format_percentage(value: float, precision: int = 2) -> str:
    return f"{value * 100:.{precision}f}%"


def print_router_report(pt_path: Path, summary: RouterSummary) -> None:
    print("\n" + "=" * 100)
    print("Supervised MoE Router Inspection")
    print("=" * 100)
    print(f"Sample file       : {pt_path}")
    if summary.scenario_id is not None:
        print(f"Scenario ID       : {summary.scenario_id}")
    if summary.agent_track_id is not None:
        print(f"Track ID          : {summary.agent_track_id}")
    if summary.timestamp_s is not None:
        print(f"Reference time    : {summary.timestamp_s:.2f}s")
    if summary.expert_label is not None:
        expert_name = EXPERT_NAMES[summary.expert_label - 1]
        print(f"Ground-truth label: {summary.expert_label} ({expert_name})")
    else:
        print("Ground-truth label: N/A (labels not provided)")

    print("\nModel weights:")
    print(f"  Shared expert contribution : {_format_percentage(summary.shared_weight)}")
    print(f"  Unshared experts contribution: {_format_percentage(summary.unshared_weight)}")

    mode_probs = summary.mode_probs.squeeze(0)
    router_probs = summary.router_probs.squeeze(0)
    selected_experts = summary.selected_experts.squeeze(0)
    topk_weights = summary.topk_weights.squeeze(0)

    best_mode = int(torch.argmax(mode_probs).item())
    print("\nMode probabilities (after softmax):")
    for mode_idx, prob in enumerate(mode_probs):
        marker = "<-- highest" if mode_idx == best_mode else ""
        print(f"  Mode {mode_idx}: {_format_percentage(float(prob))} {marker}")

    print("\nRouter distribution per mode (top-5 experts):")
    for mode_idx, mode_router in enumerate(router_probs):
        print(f"  Mode {mode_idx} (pi={_format_percentage(float(mode_probs[mode_idx]))}):")
        sorted_indices = torch.argsort(mode_router, descending=True)
        for rank, expert_idx in enumerate(sorted_indices.tolist()):
            expert_prob = float(mode_router[expert_idx])
            print(
                f"    #{rank + 1}: Expert {expert_idx} "
                f"({EXPERT_NAMES[expert_idx]}), prob={_format_percentage(expert_prob)}"
            )
        print("    Top-K selection used for synthesis:")
        for k_idx, expert_idx in enumerate(selected_experts[mode_idx].tolist()):
            gate_prob = float(topk_weights[mode_idx, k_idx])
            final_weight = gate_prob * summary.unshared_weight
            print(
                f"      • k={k_idx + 1}: Expert {expert_idx} ({EXPERT_NAMES[expert_idx]}) "
                f"-> gate={_format_percentage(gate_prob)}, "
                f"final contribution={_format_percentage(final_weight)}"
            )

    best_mode_expert = int(torch.argmax(router_probs[best_mode]).item())
    best_mode_gate = float(topk_weights[best_mode, 0]) * summary.unshared_weight
    print("\nMost confident routing (based on mode probability):")
    print(
        f"  Mode {best_mode} routes to Expert {best_mode_expert} "
        f"({EXPERT_NAMES[best_mode_expert]}) with top-k gate "
        f"weight {_format_percentage(best_mode_gate)} (within the unshared 70%)."
    )

    aggregated = torch.zeros(router_probs.size(-1))
    for mode_idx in range(router_probs.size(0)):
        for k_idx, expert_idx in enumerate(selected_experts[mode_idx].tolist()):
            weight = (
                float(mode_probs[mode_idx])
                * float(topk_weights[mode_idx, k_idx])
                * summary.unshared_weight
            )
            aggregated[expert_idx] += weight
    print("\nOverall expert contribution (mode-weighted, excluding shared expert):")
    for expert_idx, total_weight in enumerate(aggregated.tolist()):
        print(
            f"  Expert {expert_idx} ({EXPERT_NAMES[expert_idx]}): "
            f"{_format_percentage(total_weight)}"
        )
    print()


def main() -> None:
    args = parse_args()

    pt_path = _validate_existing_path(
        _resolve_path(args.pt, PT_FILE_PATH),
        "Processed .pt file",
    )

    classification_csv = _resolve_path(args.classification, CLASSIFICATION_CSV_PATH)
    if classification_csv is not None:
        _validate_existing_path(classification_csv, "Classification CSV")

    config_path = _resolve_path(args.config, MODEL_CONFIG_PATH)
    checkpoint_path = _resolve_path(args.checkpoint, CHECKPOINT_PATH)

    device = torch.device(args.device) if args.device else DEVICE
    print(f"Using device: {device}")

    module = load_lightning_module(checkpoint_path, config_path, device)

    batch = load_single_batch(pt_path, classification_csv, DATASET_DEFAULTS)
    summary = extract_router_summary(module, batch, device)
    print_router_report(pt_path, summary)


if __name__ == "__main__":
    main()
