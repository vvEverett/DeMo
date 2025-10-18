#!/usr/bin/env python3
"""
Inspect preprocessed .pt files from DeMo with detailed tensor information

Usage:
    # Random selection
    python tools/data_inspector.py

    # Select by index
    # Modify FILE_INDEX in configuration section below

    # Select by filename (with or without .pt extension)
    # Modify FILE_NAME in configuration section below
    # Example: FILE_NAME = 'scenario_0000b0f9-99f9-4a1f-a231-5be9e4c523f7'
"""
from pathlib import Path
import sys

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.data_visualization import DataLoader, DataSelector

# ============ Configuration ============
DATA_ROOT = 'data/DeMo_processed'
SPLIT = 'train'
FILE_INDEX = None  # None for random selection
FILE_NAME = None   # None for index/random selection, or specify filename (with or without .pt extension)
# ========================================


class DetailedInspector:
    """Inspector for detailed tensor information beyond visualization"""
    
    OBJECT_TYPES = {
        0: 'vehicle', 1: 'pedestrian', 2: 'motorcyclist', 3: 'cyclist',
        4: 'bus', 5: 'static', 6: 'background', 7: 'construction',
        8: 'riderless_bicycle', 9: 'unknown'
    }
    
    OBJECT_CATEGORIES = {
        0: 'track_fragment',   # Incomplete trajectory observations
        1: 'unscored_track',   # Background agents (not evaluated)
        2: 'scored_track',     # Important surrounding agents (evaluated)
        3: 'focal_track'       # Ego vehicle (prediction target)
    }
    
    COMBINED_TYPES = {0: 'vehicle/bus', 1: 'pedestrian', 2: 'cyclist/motorcyclist', 3: 'static/other'}
    LANE_TYPES = {0: 'vehicle', 1: 'bike', 2: 'bus'}
    
    def __init__(self, data_loader: DataLoader):
        self.loader = data_loader
        self.data = data_loader.current_data
    
    def print_tensor(self, name, tensor, sample_size=5):
        """Print tensor with optional samples"""
        if isinstance(tensor, (list, tuple)):
            print(f"  {name:20s}: length={len(tensor)}, type={type(tensor).__name__}")
            if len(tensor) <= 5:
                print(f"{'':22s}content={tensor}")
            return
        
        if not isinstance(tensor, torch.Tensor):
            print(f"  {name:20s}: {tensor}")
            return
            
        print(f"  {name:20s}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
        
        if tensor.numel() <= 10:
            print(f"{'':22s}value={tensor.tolist()}")
        elif len(tensor.shape) >= 2 and sample_size > 0:
            print(f"{'':22s}sample (first {sample_size}):")
            for i in range(min(sample_size, len(tensor))):
                print(f"{'':24s}[{i}] {tensor[i].tolist()}")
    
    def show_metadata(self):
        """Show metadata fields"""
        print("\n[Metadata]")
        for key in ['scenario_id', 'city', 'focal_idx']:
            if key in self.data:
                self.print_tensor(key, self.data[key])
        
        if 'agent_ids' in self.data:
            agent_ids = self.data['agent_ids']
            print(f"  agent_ids          : {len(agent_ids)} IDs")
            print(f"{'':22s}first 3: {agent_ids[:3]}")
        
        if 'scored_idx' in self.data:
            self.print_tensor('scored_idx', self.data['scored_idx'])
    
    def show_agent_data(self):
        """Show all agent-related tensors"""
        print("\n[Agent Data]")
        if 'x_positions' in self.data:
            N, T, _ = self.data['x_positions'].shape
            print(f"  Number of agents   : {N}")
            print(f"  Timesteps          : {T}")
        
        for key in ['x_positions', 'x_angles', 'x_velocity', 'x_valid_mask']:
            if key in self.data:
                self.print_tensor(key, self.data[key], sample_size=0)
    
    def show_agent_attributes(self):
        """Detailed x_attr analysis"""
        print("\n[Agent Attributes Details]")
        x_attr = self.data['x_attr']
        self.print_tensor('x_attr', x_attr, sample_size=5)
        
        print(f"{'':22s}Meaning: [object_type, object_category, combined_type]")
        print(f"{'':22s}object_type: 0=vehicle, 1=pedestrian, 2=motorcyclist,")
        print(f"{'':22s}             3=cyclist, 4=bus, 5=static, 6=background,")
        print(f"{'':22s}             7=construction, 8=riderless_bicycle, 9=unknown")
        print(f"{'':22s}object_category: 0=track_fragment, 1=unscored_track,")
        print(f"{'':22s}                 2=scored_track, 3=focal_track")
        print(f"{'':22s}combined_type: 0=vehicle/bus, 1=pedestrian,")
        print(f"{'':22s}               2=cyclist/motorcyclist, 3=static/other")
        
        # Distribution statistics
        print(f"\n{'':22s}Distribution:")
        for idx, (type_dict, name) in enumerate([
            (self.OBJECT_TYPES, 'object_type'),
            (self.OBJECT_CATEGORIES, 'object_category'),
            (self.COMBINED_TYPES, 'combined_type')
        ]):
            values = x_attr[:, idx]
            unique, counts = torch.unique(values, return_counts=True)
            print(f"{'':24s}{name}:")
            for val, cnt in zip(unique.tolist(), counts.tolist()):
                label = type_dict.get(val, f'unknown_{val}')
                print(f"{'':26s}{label}: {cnt}")
    
    def show_lane_data(self):
        """Show all lane-related tensors"""
        print("\n[Lane Data]")
        if 'lane_positions' in self.data:
            M, P, _ = self.data['lane_positions'].shape
            print(f"  Number of lanes    : {M}")
            print(f"  Points per lane    : {P}")
        
        for key in ['lane_positions', 'is_intersections']:
            if key in self.data:
                self.print_tensor(key, self.data[key], sample_size=0)
    
    def show_lane_attributes(self):
        """Detailed lane_attr analysis"""
        if 'lane_attr' not in self.data:
            return
            
        print("\n[Lane Attributes Details]")
        lane_attr = self.data['lane_attr']
        self.print_tensor('lane_attr', lane_attr, sample_size=3)
        print(f"{'':22s}Meaning: [lane_type, width, is_intersection]")
        print(f"{'':22s}lane_type: 0=vehicle, 1=bike, 2=bus")
        
        # Lane statistics
        lane_types = lane_attr[:, 0]
        unique_types, type_counts = torch.unique(lane_types, return_counts=True)
        print(f"{'':22s}Lane type distribution:")
        for val, cnt in zip(unique_types.tolist(), type_counts.tolist()):
            label = self.LANE_TYPES.get(val, f'type_{val}')
            print(f"{'':24s}{label}: {cnt}")
    
    def show_statistics(self):
        """Show statistical information for focal agent"""
        print("\n[Statistics - Focal Agent Only]")
        
        focal_idx = self.data.get('focal_idx', 0)
        
        # Focal agent valid timesteps
        if 'x_valid_mask' in self.data:
            focal_valid = self.data['x_valid_mask'][focal_idx]
            valid_count = focal_valid.sum().item()
            total_timesteps = len(focal_valid)
            print(f"  Valid timesteps: {valid_count}/{total_timesteps} ({valid_count/total_timesteps*100:.1f}%)")
        
        # Focal agent velocity statistics
        if 'x_velocity' in self.data and 'x_valid_mask' in self.data:
            focal_vel = self.data['x_velocity'][focal_idx]
            focal_valid_mask = self.data['x_valid_mask'][focal_idx]
            valid_vel = focal_vel[focal_valid_mask]
            
            if len(valid_vel) > 0:
                print(f"  Velocity:")
                print(f"    mean={valid_vel.mean():.2f} m/s, "
                    f"max={valid_vel.max():.2f} m/s, "
                    f"min={valid_vel.min():.2f} m/s, "
                    f"std={valid_vel.std():.2f} m/s")
        
        # Trajectory length
        if 'x_positions' in self.data and 'x_valid_mask' in self.data:
            positions = self.data['x_positions'][focal_idx]
            valid_mask = self.data['x_valid_mask'][focal_idx]
            valid_pos = positions[valid_mask]
            
            if len(valid_pos) > 1:
                diffs = torch.diff(valid_pos, dim=0)
                distances = torch.sqrt((diffs**2).sum(dim=1))
                total_distance = distances.sum().item()
                print(f"  Trajectory length: {total_distance:.2f} m")
        
        # Focal agent attributes (EGO x_attr)
        if 'x_attr' in self.data:
            focal_attr = self.data['x_attr'][focal_idx]
            print(f"  Attributes (x_attr):")
            print(f"    object_type: {focal_attr[0].item()} ({self.OBJECT_TYPES.get(focal_attr[0].item(), 'unknown')})")
            print(f"    object_category: {focal_attr[1].item()} ({self.OBJECT_CATEGORIES.get(focal_attr[1].item(), 'unknown')})")
            print(f"    combined_type: {focal_attr[2].item()} ({self.COMBINED_TYPES.get(focal_attr[2].item(), 'unknown')})")
    
    def show_data_samples(self):
        """Show actual coordinate values"""
        print("\n[Data Samples]")
        
        # Focal agent trajectory
        focal_idx = self.data.get('focal_idx', 0)
        positions = self.data['x_positions'][focal_idx]
        valid_mask = self.data['x_valid_mask'][focal_idx]
        valid_pos = positions[valid_mask]
        velocities = self.data['x_velocity'][focal_idx][valid_mask]
        
        print(f"  Focal agent (idx={focal_idx}) trajectory (first 3 valid timesteps):")
        for i in range(min(3, len(valid_pos))):
            print(f"    t={i}: pos=({valid_pos[i, 0]:7.2f}, {valid_pos[i, 1]:7.2f}), vel={velocities[i]:5.2f} m/s")
        
        # First lane
        if 'lane_positions' in self.data:
            lane0 = self.data['lane_positions'][0]
            print(f"  Lane 0 centerline (first 3 points):")
            for i in range(min(3, len(lane0))):
                print(f"    point {i}: ({lane0[i, 0]:7.2f}, {lane0[i, 1]:7.2f})")
    
    def inspect(self):
        """Full inspection routine"""
        print(f"\n{'='*60}")
        scenario_id = self.data.get('scenario_id', 'unknown')
        print(f"File: scenario_{scenario_id}.pt")
        print(f"{'='*60}")
        
        self.show_metadata()
        self.show_agent_data()
        self.show_agent_attributes()
        self.show_lane_data()
        self.show_lane_attributes()
        self.show_statistics()
        self.show_data_samples()


def build_file_dict(data_root: Path):
    """Build file dictionary for DataSelector"""
    all_files = {}
    for split in ['train', 'val', 'test']:
        split_path = data_root / split
        if split_path.exists():
            all_files[split] = sorted(list(split_path.glob('*.pt')))
    return all_files


def main():
    data_root = Path(DATA_ROOT)
    
    # Build file dictionary and create selector
    all_files = build_file_dict(data_root)
    selector = DataSelector(all_files)
    
    # Select file
    if FILE_NAME is not None:
        file_path = selector.select_file_by_name(FILE_NAME, SPLIT)
    elif FILE_INDEX is None:
        file_path = selector.select_random_file(SPLIT)
    else:
        file_path = selector.select_file_by_index(SPLIT, FILE_INDEX)
    
    if file_path is None:
        print("Error: Could not select file")
        return
    
    # Load with DataLoader
    loader = DataLoader()
    loader.load_scenario(file_path)
    
    # Detailed inspection
    inspector = DetailedInspector(loader)
    inspector.inspect()


if __name__ == '__main__':
    main()
