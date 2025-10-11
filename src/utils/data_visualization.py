"""
Data Visualization Utilities for DeMo Preprocessed Data

This module provides comprehensive visualization tools for exploring
DeMo autonomous vehicle trajectory data.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import pandas as pd
from pathlib import Path
import os
import random
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Default visualization configuration
DEFAULT_VIZ_CONFIG = {
    'figsize': (15, 10),
    'dpi': 100,
    'agent_colors': ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'],
    'focal_agent_color': 'red',
    'lane_color': 'blue',
    'lane_width': 2,
    'agent_size': 50,
    'trajectory_alpha': 0.7,
    'map_alpha': 0.5,
    'time_step_size': 0.1,  # Time between frames in seconds
}


class DataSelector:
    """Data selection interface for choosing scenarios to visualize."""
    
    def __init__(self, all_files: Dict[str, List[Path]]):
        self.all_files = all_files
        self.current_split = 'train'
        self.current_file_idx = 0
        
    def select_random_file(self, split: str = None) -> Optional[Path]:
        """Select a random file from the specified split."""
        if split is None:
            split = self.current_split
            
        if split not in self.all_files or len(self.all_files[split]) == 0:
            print(f"No files available for split: {split}")
            return None
            
        file_path = random.choice(self.all_files[split])
        print(f"Selected random file from {split}: {file_path.name}")
        return file_path
    
    def select_file_by_index(self, split: str, index: int) -> Optional[Path]:
        """Select a file by index from the specified split."""
        if split not in self.all_files:
            print(f"Split {split} not available!")
            return None
            
        files = self.all_files[split]
        if index < 0 or index >= len(files):
            print(f"Index {index} out of range. Available range: 0-{len(files)-1}")
            return None
            
        file_path = files[index]
        print(f"Selected file {index} from {split}: {file_path.name}")
        return file_path
    
    def select_file_by_name(self, filename: str, split: str = None) -> Optional[Path]:
        """Select a file by filename (with or without .pt extension)."""
        if split is None:
            split = self.current_split
            
        if split not in self.all_files:
            print(f"Split {split} not available!")
            return None
        
        # Add .pt extension if not present
        if not filename.endswith('.pt'):
            filename = filename + '.pt'
        
        files = self.all_files[split]
        for file_path in files:
            if file_path.name == filename:
                print(f"Selected file from {split}: {file_path.name}")
                return file_path
        
        print(f"File '{filename}' not found in {split} split!")
        print(f"Use list_files('{split}') to see available files.")
        return None
    
    def get_file_index(self, filename: str, split: str = None) -> Optional[int]:
        """Get the index of a file by its filename."""
        if split is None:
            split = self.current_split
            
        if split not in self.all_files:
            print(f"Split {split} not available!")
            return None
        
        # Add .pt extension if not present
        if not filename.endswith('.pt'):
            filename = filename + '.pt'
        
        files = self.all_files[split]
        for i, file_path in enumerate(files):
            if file_path.name == filename:
                return i
        
        return None
    
    def select_next_file(self, split: str = None) -> Optional[Path]:
        """Select the next file in sequence."""
        if split is None:
            split = self.current_split
            
        if split not in self.all_files or len(self.all_files[split]) == 0:
            print(f"No files available for split: {split}")
            return None
        
        files = self.all_files[split]
        self.current_file_idx = (self.current_file_idx + 1) % len(files)
        file_path = files[self.current_file_idx]
        print(f"Selected file {self.current_file_idx}/{len(files)-1} from {split}: {file_path.name}")
        return file_path
    
    def select_previous_file(self, split: str = None) -> Optional[Path]:
        """Select the previous file in sequence."""
        if split is None:
            split = self.current_split
            
        if split not in self.all_files or len(self.all_files[split]) == 0:
            print(f"No files available for split: {split}")
            return None
        
        files = self.all_files[split]
        self.current_file_idx = (self.current_file_idx - 1) % len(files)
        file_path = files[self.current_file_idx]
        print(f"Selected file {self.current_file_idx}/{len(files)-1} from {split}: {file_path.name}")
        return file_path
    
    def list_files(self, split: str, max_files: int = 10):
        """List available files for a split."""
        if split not in self.all_files:
            print(f"Split {split} not available!")
            return
            
        files = self.all_files[split]
        print(f"Available files in {split} (showing first {min(max_files, len(files))}):")
        for i, file_path in enumerate(files[:max_files]):
            print(f"  {i}: {file_path.name}")
        
        if len(files) > max_files:
            print(f"  ... and {len(files) - max_files} more files")


class DataLoader:
    """Load and parse DeMo preprocessed data files."""
    
    def __init__(self):
        self.current_data = None
        self.current_metadata = {}
        self.current_file_path = None
    
    def load_scenario(self, file_path: Path) -> Dict:
        """Load a scenario from a .pt file."""
        try:
            data = torch.load(file_path, map_location='cpu')
            self.current_data = data
            self.current_file_path = file_path  # Save the file path
            self._extract_metadata()
            print(f"Successfully loaded: {file_path.name}")
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _extract_metadata(self):
        """Extract metadata from loaded data."""
        if self.current_data is None:
            return
            
        data = self.current_data
        self.current_metadata = {
            'scenario_id': data.get('scenario_id', 'Unknown'),
            'city': data.get('city', 'Unknown'),
            'num_agents': data['x_positions'].shape[0] if 'x_positions' in data else 0,
            'num_timesteps': data['x_positions'].shape[1] if 'x_positions' in data else 0,
            'num_lanes': data['lane_positions'].shape[0] if 'lane_positions' in data else 0,
            'focal_agent_idx': data.get('focal_idx', 0),
            'num_scored_agents': len(data.get('scored_idx', [])),
        }
    
    def get_agent_trajectory(self, agent_idx: int) -> Dict:
        """Get trajectory data for a specific agent."""
        if self.current_data is None:
            return None
            
        data = self.current_data
        
        # Extract position, velocity, angle data
        positions = data['x_positions'][agent_idx].numpy()  # Shape: (timesteps, 2)
        valid_mask = data['x_valid_mask'][agent_idx].numpy()  # Shape: (timesteps,)
        velocities = data['x_velocity'][agent_idx].numpy()  # Shape: (timesteps,)
        angles = data['x_angles'][agent_idx].numpy()  # Shape: (timesteps,)
        
        # Filter valid timesteps
        valid_positions = positions[valid_mask.astype(bool)]
        valid_velocities = velocities[valid_mask.astype(bool)]
        valid_angles = angles[valid_mask.astype(bool)]
        valid_timesteps = np.where(valid_mask.astype(bool))[0]
        
        return {
            'positions': valid_positions,
            'velocities': valid_velocities,
            'angles': valid_angles,
            'timesteps': valid_timesteps,
            'agent_id': data['agent_ids'][agent_idx] if 'agent_ids' in data else f"agent_{agent_idx}",
            'agent_type': self._get_agent_type(agent_idx),
            'agent_category': self._get_agent_category(agent_idx),
            'agent_type_combined': self._get_agent_type_combined(agent_idx)
        }
    
    def _get_agent_type(self, agent_idx: int) -> str:
        """Get agent type from attributes."""
        if self.current_data is None or 'x_attr' not in self.current_data:
            return 'Unknown'
            
        # x_attr contains [object_type, object_category, object_type_combined]
        # object_type mapping from OBJECT_TYPE_MAP in av2_data_utils.py
        attr = self.current_data['x_attr'][agent_idx].numpy()
        
        # Complete mapping based on OBJECT_TYPE_MAP
        OBJECT_TYPE_NAMES = {
            0: 'Vehicle',
            1: 'Pedestrian',
            2: 'Motorcyclist',
            3: 'Cyclist',
            4: 'Bus',
            5: 'Static',
            6: 'Background',
            7: 'Construction',
            8: 'Riderless_Bicycle',
            9: 'Unknown'
        }
        
        if len(attr) > 0:
            object_type = int(attr[0])
            return OBJECT_TYPE_NAMES.get(object_type, f'Type_{object_type}')
        return 'Unknown'
    
    def _get_agent_category(self, agent_idx: int) -> str:
        """Get agent category (role in scenario) from attributes."""
        if self.current_data is None or 'x_attr' not in self.current_data:
            return 'Unknown'
            
        # x_attr contains [object_type, object_category, object_type_combined]
        # object_category indicates the role of the agent in the scenario
        attr = self.current_data['x_attr'][agent_idx].numpy()
        
        # Category mapping based on Argoverse 2 standards
        OBJECT_CATEGORY_NAMES = {
            0: 'Unscored',        # Regular unscored object
            1: 'Scored_Track',    # Scored track
            2: 'Scored_Agent',    # Scored agent (to be predicted)
            3: 'Focal_Agent'      # Focal agent (AV/main vehicle)
        }
        
        if len(attr) > 1:
            category = int(attr[1])
            return OBJECT_CATEGORY_NAMES.get(category, f'Category_{category}')
        return 'Unknown'
    
    def _get_agent_type_combined(self, agent_idx: int) -> str:
        """Get combined agent type from attributes."""
        if self.current_data is None or 'x_attr' not in self.current_data:
            return 'Unknown'
            
        # x_attr contains [object_type, object_category, object_type_combined]
        # object_type_combined is a simplified grouping from OBJECT_TYPE_MAP_COMBINED
        attr = self.current_data['x_attr'][agent_idx].numpy()
        
        # Combined type mapping based on OBJECT_TYPE_MAP_COMBINED in av2_data_utils.py
        # Groups similar types together for simplified classification
        COMBINED_TYPE_NAMES = {
            0: 'Vehicle',         # Vehicle, Bus
            1: 'Pedestrian',      # Pedestrian
            2: 'Cyclist',         # Motorcyclist, Cyclist
            3: 'Other'            # Static, Background, Construction, Riderless_Bicycle, Unknown
        }
        
        if len(attr) > 2:
            combined_type = int(attr[2])
            return COMBINED_TYPE_NAMES.get(combined_type, f'Combined_{combined_type}')
        return 'Unknown'
    
    def get_agent_attributes(self, agent_idx: int) -> Dict:
        """Get all attributes for a specific agent."""
        if self.current_data is None:
            return None
            
        return {
            'agent_idx': agent_idx,
            'agent_id': self.current_data['agent_ids'][agent_idx] if 'agent_ids' in self.current_data else f"agent_{agent_idx}",
            'object_type': self._get_agent_type(agent_idx),
            'object_category': self._get_agent_category(agent_idx),
            'object_type_combined': self._get_agent_type_combined(agent_idx),
            'is_focal': agent_idx == self.current_data.get('focal_idx', -1),
            'is_scored': agent_idx in self.current_data.get('scored_idx', [])
        }
    
    def get_lane_data(self) -> Dict:
        """Get lane/map data."""
        if self.current_data is None:
            return None
            
        data = self.current_data
        
        lane_positions = data['lane_positions'].numpy()  # Shape: (num_lanes, points_per_lane, 2)
        lane_attr = data['lane_attr'].numpy()  # Shape: (num_lanes, attr_dim)
        is_intersections = data['is_intersections'].numpy()  # Shape: (num_lanes,)
        
        return {
            'lane_positions': lane_positions,
            'lane_attributes': lane_attr,
            'is_intersections': is_intersections
        }
    
    def print_summary(self):
        """Print a summary of the loaded data."""
        if self.current_data is None:
            print("No data loaded.")
            return
            
        print("=== Data Summary ===")
        for key, value in self.current_metadata.items():
            print(f"{key}: {value}")
        
        focal_idx = self.current_metadata.get('focal_agent_idx', 0)
        focal_combined_type = self._get_agent_type_combined(focal_idx)
        print(f"ego_combined_type: {focal_combined_type}")


class TrajectoryVisualizer:
    """Visualize agent trajectories and related data."""
    
    def __init__(self, data_loader: DataLoader, config: Dict = None):
        self.data_loader = data_loader
        self.config = config or DEFAULT_VIZ_CONFIG
    
    def plot_trajectories(self, 
                         show_lanes: bool = True,
                         show_velocity: bool = False,
                         show_agent_ids: bool = True,
                         time_range: Optional[Tuple[int, int]] = None,
                         specific_agents: Optional[List[int]] = None):
        """Plot agent trajectories with various options."""
        
        if self.data_loader.current_data is None:
            print("No data loaded. Please load a scenario first.")
            return
        
        fig, ax = plt.subplots(figsize=self.config['figsize'])
        
        # Plot lanes first (background)
        if show_lanes:
            self._plot_lanes(ax)
        
        # Get agent data
        num_agents = self.data_loader.current_metadata['num_agents']
        focal_idx = self.data_loader.current_metadata['focal_agent_idx']
        
        # Determine which agents to plot
        if specific_agents is None:
            agents_to_plot = range(num_agents)
        else:
            agents_to_plot = specific_agents
        
        # Plot each agent's trajectory
        for i, agent_idx in enumerate(agents_to_plot):
            if agent_idx >= num_agents:
                continue
                
            agent_data = self.data_loader.get_agent_trajectory(agent_idx)
            if agent_data is None or len(agent_data['positions']) == 0:
                continue
            
            positions = agent_data['positions']
            
            # Apply time range filter if specified
            if time_range is not None:
                start_t, end_t = time_range
                time_mask = (agent_data['timesteps'] >= start_t) & (agent_data['timesteps'] <= end_t)
                if np.any(time_mask):
                    positions = positions[time_mask]
                    timesteps = agent_data['timesteps'][time_mask]
                else:
                    continue
            
            # Choose color
            if agent_idx == focal_idx:
                color = self.config['focal_agent_color']
                linewidth = 3
                alpha = 1.0
                marker_size = self.config['agent_size'] * 1.5
            else:
                color = self.config['agent_colors'][i % len(self.config['agent_colors'])]
                linewidth = 2
                alpha = self.config['trajectory_alpha']
                marker_size = self.config['agent_size']
            
            # Plot trajectory line
            ax.plot(positions[:, 0], positions[:, 1], 
                   color=color, linewidth=linewidth, alpha=alpha,
                   label=f"Agent {agent_idx}" + (" (Focal)" if agent_idx == focal_idx else ""))
            
            # Plot start and end points
            if len(positions) > 0:
                # Start point (circle)
                ax.scatter(positions[0, 0], positions[0, 1], 
                          c=color, s=marker_size, marker='o', alpha=alpha,
                          edgecolors='black', linewidths=1)
                
                # End point (triangle)
                ax.scatter(positions[-1, 0], positions[-1, 1], 
                          c=color, s=marker_size, marker='^', alpha=alpha,
                          edgecolors='black', linewidths=1)
            
            # Add agent ID text
            if show_agent_ids and len(positions) > 0:
                mid_idx = len(positions) // 2
                ax.annotate(f'{agent_idx}', 
                           (positions[mid_idx, 0], positions[mid_idx, 1]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color=color, fontweight='bold')
        
        # Customize plot
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        ax.set_title(f"Scenario: {self.data_loader.current_metadata['scenario_id']}\n"
                    f"City: {self.data_loader.current_metadata['city']}")
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Add legend (limit to reasonable number of entries)
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # Show only focal agent and first few others
            focal_handles = [h for h, l in zip(handles, labels) if 'Focal' in l]
            other_handles = [h for h, l in zip(handles, labels) if 'Focal' not in l][:5]
            selected_labels = [l for h, l in zip(handles, labels) if 'Focal' in l]
            selected_labels += [l for h, l in zip(handles, labels) if 'Focal' not in l][:5]
            if len(other_handles) < len(handles) - len(focal_handles):
                selected_labels.append(f"... and {len(handles) - len(focal_handles) - 5} more agents")
            
            ax.legend(focal_handles + other_handles, selected_labels, 
                     bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_lanes(self, ax):
        """Plot lane information on the given axes."""
        lane_data = self.data_loader.get_lane_data()
        if lane_data is None:
            return
        
        lane_positions = lane_data['lane_positions']
        is_intersections = lane_data['is_intersections']
        
        for i, lane_points in enumerate(lane_positions):
            # Filter out invalid points (typically zeros)
            valid_points = lane_points[~np.all(lane_points == 0, axis=1)]
            
            if len(valid_points) < 2:
                continue
            
            # Choose color based on intersection status
            color = 'orange' if is_intersections[i] else self.config['lane_color']
            alpha = self.config['map_alpha']
            
            ax.plot(valid_points[:, 0], valid_points[:, 1], 
                   color=color, linewidth=self.config['lane_width'], 
                   alpha=alpha, linestyle='-')
    
    def plot_velocity_profile(self, agent_idx: int = None):
        """Plot velocity profile over time for an agent."""
        if self.data_loader.current_data is None:
            print("No data loaded.")
            return
        
        if agent_idx is None:
            agent_idx = self.data_loader.current_metadata['focal_agent_idx']
        
        agent_data = self.data_loader.get_agent_trajectory(agent_idx)
        if agent_data is None:
            print(f"No data for agent {agent_idx}")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        timesteps = agent_data['timesteps']
        velocities = agent_data['velocities']
        angles = agent_data['angles']
        
        # Velocity plot
        ax1.plot(timesteps, velocities, 'b-', linewidth=2)
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_title(f'Agent {agent_idx} Velocity Profile')
        ax1.grid(True, alpha=0.3)
        
        # Heading angle plot
        ax2.plot(timesteps, np.degrees(angles), 'r-', linewidth=2)
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Heading Angle (degrees)')
        ax2.set_title(f'Agent {agent_idx} Heading Profile')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class EgoOtherVisualizer:
    """Specialized visualizer for ego vs. other vehicles with separate plots."""
    
    def __init__(self, data_loader: DataLoader, config: Dict = None):
        self.data_loader = data_loader
        self.config = config or DEFAULT_VIZ_CONFIG
    
    def plot_ego_vs_others(self, 
                          show_lanes: bool = True,
                          show_agent_ids: bool = True,
                          figure_size: Tuple[int, int] = (16, 8)):
        """Create two separate plots: ego vehicle and other vehicles."""
        
        if self.data_loader.current_data is None:
            print("No data loaded. Please load a scenario first.")
            return
        
        metadata = self.data_loader.current_metadata
        focal_idx = metadata['focal_agent_idx']
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figure_size)
        
        # Plot 1: Ego Vehicle Only
        self._plot_ego_trajectory(ax1, focal_idx, show_lanes, show_agent_ids)
        
        # Plot 2: Other Vehicles Only
        self._plot_other_trajectories(ax2, focal_idx, show_lanes, show_agent_ids)
        
        # Add overall title
        scenario_id = metadata['scenario_id']
        city = metadata['city']
        fig.suptitle(f"Ego vs. Other Vehicles - Scenario: {scenario_id} (City: {city})", 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_ego_trajectory(self, ax, focal_idx: int, show_lanes: bool, show_agent_ids: bool):
        """Plot only the ego vehicle trajectory."""
        
        # Plot lanes first (background)
        if show_lanes:
            self._plot_lanes(ax, alpha=0.3)
        
        # Get ego agent data
        ego_data = self.data_loader.get_agent_trajectory(focal_idx)
        
        if ego_data is None or len(ego_data['positions']) == 0:
            ax.text(0.5, 0.5, 'No ego vehicle data available', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, color='red')
            ax.set_title("Ego Vehicle Trajectory")
            return
        
        positions = ego_data['positions']
        timesteps = ego_data['timesteps']
        
        # Plot ego trajectory with special styling
        ax.plot(positions[:, 0], positions[:, 1], 
               color=self.config['focal_agent_color'], 
               linewidth=2, alpha=1.0, label=f"Ego (Agent {focal_idx})")
        
        # Plot start and end points
        if len(positions) > 0:
            # Start point (medium green circle)
            ax.scatter(positions[0, 0], positions[0, 1], 
                      c='green', s=50, marker='o', alpha=1.0,
                      edgecolors='black', linewidths=2, label='Start')
            
            # End point (medium red triangle)
            ax.scatter(positions[-1, 0], positions[-1, 1], 
                      c='red', s=50, marker='^', alpha=1.0,
                      edgecolors='black', linewidths=2, label='End')
            
            # Mark position at t=5s (timestep 50, assuming 0.1s per step)
            time_seconds = timesteps * 0.1
            if len(time_seconds) > 0 and np.max(time_seconds) >= 5.0:
                idx_5s = np.argmin(np.abs(time_seconds - 5.0))
                if idx_5s < len(positions):
                    ax.scatter(positions[idx_5s, 0], positions[idx_5s, 1], c='orange', s=80, marker='*', 
                               edgecolors='black', linewidths=2, label='t=5s', zorder=6)
        
        # # Add ego agent ID
        # if show_agent_ids and len(positions) > 0:
        #     mid_idx = len(positions) // 2
        #     ax.annotate(f'EGO-{focal_idx}', 
        #                (positions[mid_idx, 0], positions[mid_idx, 1]),
        #                xytext=(10, 10), textcoords='offset points',
        #                fontsize=12, color=self.config['focal_agent_color'], 
        #                fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
        #                                            facecolor='white', alpha=0.8))
        
        # Customize ego plot
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        ax.set_title('Ego Vehicle Trajectory', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
        
        # # Add ego statistics
        # avg_velocity = np.mean(ego_data['velocities']) if len(ego_data['velocities']) > 0 else 0
        # max_velocity = np.max(ego_data['velocities']) if len(ego_data['velocities']) > 0 else 0
        # trajectory_length = self._calculate_trajectory_length(positions)
        
        # stats_text = f"Avg Speed: {avg_velocity:.1f} m/s\nMax Speed: {max_velocity:.1f} m/s\nDistance: {trajectory_length:.1f} m"
        # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        #        verticalalignment='top', fontsize=10,
        #        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
    
    def _plot_other_trajectories(self, ax, focal_idx: int, show_lanes: bool, show_agent_ids: bool):
        """Plot all other vehicles' trajectories."""
        
        # Plot lanes first (background)
        if show_lanes:
            self._plot_lanes(ax, alpha=0.3)
        
        metadata = self.data_loader.current_metadata
        num_agents = metadata['num_agents']
        
        # Plot all other agents
        other_agents_count = 0
        for agent_idx in range(num_agents):
            if agent_idx == focal_idx:  # Skip ego vehicle
                continue
                
            agent_data = self.data_loader.get_agent_trajectory(agent_idx)
            if agent_data is None or len(agent_data['positions']) == 0:
                continue
            
            positions = agent_data['positions']
            
            # Choose color for other agents
            color_idx = other_agents_count % len(self.config['agent_colors'])
            color = self.config['agent_colors'][color_idx]
            
            # Plot trajectory
            ax.plot(positions[:, 0], positions[:, 1], 
                   color=color, linewidth=2, alpha=self.config['trajectory_alpha'])
            
            # Plot start and end points (smaller)
            if len(positions) > 0:
                # Start point (small circle)
                ax.scatter(positions[0, 0], positions[0, 1], 
                          c=color, s=30, marker='o', alpha=0.8,
                          edgecolors='black', linewidths=1)
                
                # End point (small triangle)
                ax.scatter(positions[-1, 0], positions[-1, 1], 
                          c=color, s=30, marker='^', alpha=0.8,
                          edgecolors='black', linewidths=1)
            
            # Add agent ID (only for first few agents to avoid clutter)
            if show_agent_ids and other_agents_count < 10 and len(positions) > 0:
                mid_idx = len(positions) // 2
                ax.annotate(f'{agent_idx}', 
                           (positions[mid_idx, 0], positions[mid_idx, 1]),
                           xytext=(3, 3), textcoords='offset points',
                           fontsize=8, color=color, fontweight='bold')
            
            other_agents_count += 1
        
        # Customize other agents plot
        ax.set_xlabel('X Position (meters)')
        ax.set_ylabel('Y Position (meters)')
        ax.set_title(f'Other Vehicles Trajectories ({other_agents_count} agents)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Add other agents statistics
        if other_agents_count > 0:
            stats_text = f"Total Other Agents: {other_agents_count}\nShowing first 10 IDs"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
    
    def _plot_lanes(self, ax, alpha: float = 0.5):
        """Plot lane information on the given axes."""
        lane_data = self.data_loader.get_lane_data()
        if lane_data is None:
            return
        
        lane_positions = lane_data['lane_positions']
        is_intersections = lane_data['is_intersections']
        
        for i, lane_points in enumerate(lane_positions):
            # Filter out invalid points
            valid_points = lane_points[~np.all(lane_points == 0, axis=1)]
            
            if len(valid_points) < 2:
                continue
            
            # Choose color based on intersection status
            color = 'orange' if is_intersections[i] else self.config['lane_color']
            
            ax.plot(valid_points[:, 0], valid_points[:, 1], 
                   color=color, linewidth=self.config['lane_width'], 
                   alpha=alpha, linestyle='-')
    
    def _calculate_trajectory_length(self, positions: np.ndarray) -> float:
        """Calculate the total length of a trajectory."""
        if len(positions) < 2:
            return 0.0
        
        diffs = np.diff(positions, axis=0)
        distances = np.sqrt(np.sum(diffs**2, axis=1))
        return np.sum(distances)


# Utility functions for data information display
def display_agent_information(data_loader: DataLoader):
    """Display detailed information about all agents in the current scenario."""
    if data_loader.current_data is None:
        print("No data loaded. Please load a scenario first.")
        return
    
    print("=== Agent Information ===")
    metadata = data_loader.current_metadata
    
    print(f"Scenario ID: {metadata['scenario_id']}")
    print(f"City: {metadata['city']}")
    print(f"Total Agents: {metadata['num_agents']}")
    print(f"Focal Agent Index: {metadata['focal_agent_idx']}")
    print(f"Number of Scored Agents: {metadata['num_scored_agents']}")
    print(f"Total Timesteps: {metadata['num_timesteps']}")
    print(f"Number of Lanes: {metadata['num_lanes']}")
    
    print("\n=== Individual Agent Details ===")
    
    # Create a summary table
    agent_summary = []
    
    for agent_idx in range(min(metadata['num_agents'], 20)):  # Limit to first 20 agents
        agent_data = data_loader.get_agent_trajectory(agent_idx)
        
        if agent_data is None:
            continue
        
        # Calculate statistics
        positions = agent_data['positions']
        velocities = agent_data['velocities']
        
        if len(positions) == 0:
            continue
        
        # Calculate trajectory length
        if len(positions) > 1:
            diffs = np.diff(positions, axis=0)
            trajectory_length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
        else:
            trajectory_length = 0
        
        # Calculate statistics
        avg_velocity = np.mean(velocities) if len(velocities) > 0 else 0
        max_velocity = np.max(velocities) if len(velocities) > 0 else 0
        
        agent_summary.append({
            'Agent Index': agent_idx,
            'Agent ID': agent_data['agent_id'],
            'Type': agent_data['agent_type'],
            'Is Focal': '✓' if agent_idx == metadata['focal_agent_idx'] else '',
            'Valid Timesteps': len(agent_data['timesteps']),
            'Trajectory Length (m)': f"{trajectory_length:.1f}",
            'Avg Velocity (m/s)': f"{avg_velocity:.2f}",
            'Max Velocity (m/s)': f"{max_velocity:.2f}",
        })
    
    # Create and display DataFrame
    if agent_summary:
        df = pd.DataFrame(agent_summary)
        print(df.to_string(index=False))
        
        if metadata['num_agents'] > 20:
            print(f"\n... and {metadata['num_agents'] - 20} more agents (showing first 20)")
    else:
        print("No valid agent data found.")


def display_scenario_statistics(data_loader: DataLoader):
    """Display statistical summary of the current scenario."""
    if data_loader.current_data is None:
        print("No data loaded.")
        return
    
    print("=== Scenario Statistics ===")
    
    # Collect all agent trajectories
    all_velocities = []
    all_trajectory_lengths = []
    agent_types = {}
    
    metadata = data_loader.current_metadata
    
    for agent_idx in range(metadata['num_agents']):
        agent_data = data_loader.get_agent_trajectory(agent_idx)
        
        if agent_data is None or len(agent_data['positions']) == 0:
            continue
        
        # Collect velocities
        all_velocities.extend(agent_data['velocities'])
        
        # Calculate trajectory length
        positions = agent_data['positions']
        if len(positions) > 1:
            diffs = np.diff(positions, axis=0)
            trajectory_length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
            all_trajectory_lengths.append(trajectory_length)
        
        # Count agent types
        agent_type = agent_data['agent_type']
        agent_types[agent_type] = agent_types.get(agent_type, 0) + 1
    
    # Display statistics
    if all_velocities:
        all_velocities = np.array(all_velocities)
        print(f"Velocity Statistics:")
        print(f"  Mean: {np.mean(all_velocities):.2f} m/s")
        print(f"  Std:  {np.std(all_velocities):.2f} m/s")
        print(f"  Min:  {np.min(all_velocities):.2f} m/s")
        print(f"  Max:  {np.max(all_velocities):.2f} m/s")
    
    if all_trajectory_lengths:
        all_trajectory_lengths = np.array(all_trajectory_lengths)
        print(f"\nTrajectory Length Statistics:")
        print(f"  Mean: {np.mean(all_trajectory_lengths):.1f} m")
        print(f"  Std:  {np.std(all_trajectory_lengths):.1f} m")
        print(f"  Min:  {np.min(all_trajectory_lengths):.1f} m")
        print(f"  Max:  {np.max(all_trajectory_lengths):.1f} m")
    
    if agent_types:
        print(f"\nAgent Type Distribution:")
        for agent_type, count in agent_types.items():
            print(f"  {agent_type}: {count}")


def get_available_files(base_data_path: Path, split: str = 'train') -> List[Path]:
    """Get all available preprocessed data files for a given split."""
    split_path = base_data_path / split
    if not split_path.exists():
        print(f"Split {split} not found!")
        return []
    
    files = list(split_path.glob("*.pt"))
    files.sort()  # Sort files for consistent ordering
    return files


def display_file_info(files: List[Path]):
    """Display information about available files."""
    print(f"Found {len(files)} files:")
    if len(files) > 0:
        print(f"Sample files:")
        for i, file in enumerate(files[:5]):  # Show first 5 files
            print(f"  {i+1}. {file.name}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more files")


# High-level exploration functions
def explore_scenario(data_selector: DataSelector, 
                    data_loader: DataLoader,
                    visualizer: TrajectoryVisualizer,
                    split: str = 'train', 
                    scenario_index: Optional[int] = None,
                    show_lanes: bool = True,
                    show_agent_info: bool = True,
                    show_statistics: bool = False,
                    specific_agents: Optional[List[int]] = None):
    """
    Complete function to explore a scenario with all visualization options.
    
    Parameters:
    - data_selector: DataSelector instance
    - data_loader: DataLoader instance  
    - visualizer: TrajectoryVisualizer instance
    - split: Dataset split ('train', 'val', 'test')
    - scenario_index: Specific scenario index (None for random)
    - show_lanes: Whether to show lane information
    - show_agent_info: Whether to display agent information table
    - show_statistics: Whether to show scenario statistics
    - specific_agents: List of specific agent indices to visualize (None for all)
    """
    
    print(f"=== Exploring {split.upper()} Split ===")
    
    # Select file
    if scenario_index is None:
        file_path = data_selector.select_random_file(split)
    else:
        file_path = data_selector.select_file_by_index(split, scenario_index)
    
    if file_path is None:
        return
    
    # Load data
    data = data_loader.load_scenario(file_path)
    if data is None:
        return
    
    # Print basic info
    data_loader.print_summary()
    print()
    
    # Show agent information
    if show_agent_info:
        display_agent_information(data_loader)
        print()
    
    # Show statistics
    if show_statistics:
        display_scenario_statistics(data_loader)
        print()
    
    # Plot trajectories
    print("Generating trajectory visualization...")
    visualizer.plot_trajectories(
        show_lanes=show_lanes,
        show_agent_ids=True,
        specific_agents=specific_agents
    )

# === NEW VELOCITY ANALYSIS FUNCTIONS ===

def plot_ego_velocity_analysis(data_loader: DataLoader, 
                               show_acceleration: bool = True, 
                               time_window: Optional[Tuple[float, float]] = None) -> Dict:
    """
    Analyze ego agent velocity and acceleration profiles over time.
    
    Parameters:
    - data_loader: DataLoader instance with loaded scenario data
    - show_acceleration: Whether to calculate acceleration data
    - time_window: Tuple (start_time, end_time) to focus on specific time range
    
    Returns:
    - Dictionary containing velocity analysis data and statistics
    """
    
    if data_loader.current_data is None:
        print("No data loaded. Please load a scenario first.")
        return None
    
    metadata = data_loader.current_metadata
    focal_idx = metadata['focal_agent_idx']
    
    # Get ego agent trajectory data
    ego_data = data_loader.get_agent_trajectory(focal_idx)
    
    if ego_data is None or len(ego_data['positions']) == 0:
        print("No ego vehicle data available.")
        return None
    
    # Extract time-series data
    timesteps = ego_data['timesteps']
    velocities = ego_data['velocities']
    positions = ego_data['positions']
    angles = ego_data['angles']
    agent_type_combined = ego_data['agent_type_combined']
    
    # Convert timesteps to time in seconds (assuming 0.1s intervals)
    time_seconds = timesteps * 0.1
    
    # Calculate acceleration (numerical derivative of velocity)
    accelerations = None
    if show_acceleration and len(velocities) > 1:
        accelerations = np.gradient(velocities) / 0.1  # acceleration = dv/dt
    
    # Apply time window filter if specified
    if time_window is not None and len(time_window) == 2:
        start_time, end_time = time_window
        mask = (time_seconds >= start_time) & (time_seconds <= end_time)
        time_seconds = time_seconds[mask]
        velocities = velocities[mask]
        angles = angles[mask]  # Filter angles as well
        if accelerations is not None:
            accelerations = accelerations[mask]
        timesteps = timesteps[mask]
        positions = positions[mask]
    
    # Calculate statistics
    velocity_stats = {}
    if len(velocities) > 0:
        velocity_stats = {
            'mean': np.mean(velocities),
            'max': np.max(velocities),
            'min': np.min(velocities),
            'std': np.std(velocities)
        }
    
    acceleration_stats = {}
    if accelerations is not None and len(accelerations) > 0:
        acceleration_stats = {
            'mean': np.mean(accelerations),
            'max': np.max(accelerations),
            'min': np.min(accelerations),
            'std': np.std(accelerations)
        }
    
    # Calculate angle statistics
    angle_stats = {}
    if len(angles) > 0:
        angle_stats = {
            'mean': np.mean(angles),
            'max': np.max(angles),
            'min': np.min(angles),
            'std': np.std(angles)
        }
    
    # Calculate trajectory distance
    total_distance = 0.0
    if len(positions) > 1:
        diffs = np.diff(positions, axis=0)
        total_distance = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
    
    return {
        'time_seconds': time_seconds,
        'velocities': velocities,
        'accelerations': accelerations,
        'angles': angles,
        'positions': positions,
        'timesteps': timesteps,
        'velocity_stats': velocity_stats,
        'acceleration_stats': acceleration_stats,
        'angle_stats': angle_stats,
        'total_distance': total_distance,
        'agent_type_combined': agent_type_combined,
        'metadata': metadata
    }


def create_velocity_plots(analysis_data: Dict, figure_size: Tuple[int, int] = (14, 12)) -> None:
    """
    Create velocity and acceleration plots from analysis data.
    
    Parameters:
    - analysis_data: Dictionary returned by plot_ego_velocity_analysis
    - figure_size: Size of the figure
    """
    
    if analysis_data is None:
        return
    
    time_seconds = analysis_data['time_seconds']
    velocities = analysis_data['velocities']
    accelerations = analysis_data['accelerations']
    positions = analysis_data['positions']
    velocity_stats = analysis_data['velocity_stats']
    acceleration_stats = analysis_data['acceleration_stats']
    metadata = analysis_data['metadata']
    
    # Create subplots
    if accelerations is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figure_size)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(figure_size[0], figure_size[1]*2/3))
    
    # Plot 1: Velocity over time
    ax1.plot(time_seconds, velocities, 'b-', linewidth=2.5, label='Velocity', alpha=0.8)
    ax1.fill_between(time_seconds, velocities, alpha=0.3, color='blue')
    
    # Add vertical line at t=5s
    if len(time_seconds) > 0 and np.max(time_seconds) >= 5.0:
        ax1.axvline(x=5.0, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='t=5s')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.set_title(f'Ego Agent Velocity Profile - Scenario: {metadata["scenario_id"]}', 
                  fontweight='bold', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add velocity statistics
    if velocity_stats:
        v_mean = velocity_stats['mean']
        v_max = velocity_stats['max']
        v_min = velocity_stats['min']
        v_std = velocity_stats['std']
        
        stats_text = f'Mean: {v_mean:.2f} m/s\nMax: {v_max:.2f} m/s\nMin: {v_min:.2f} m/s\nStd: {v_std:.2f} m/s'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.8))
        
        # Add horizontal lines for mean and max velocity
        ax1.axhline(y=v_mean, color='red', linestyle='--', alpha=0.7, label=f'Mean: {v_mean:.2f} m/s')
        ax1.axhline(y=v_max, color='orange', linestyle='--', alpha=0.7, label=f'Max: {v_max:.2f} m/s')
        ax1.legend()
    
    # Plot 2: Trajectory in 2D space
    ax2.plot(positions[:, 0], positions[:, 1], 'g-', linewidth=3, alpha=0.8, label='Ego Trajectory')
    ax2.scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='o', 
               edgecolors='black', linewidths=2, label='Start', zorder=5)
    ax2.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, marker='^', 
               edgecolors='black', linewidths=2, label='End', zorder=5)
    
    # Mark position at t=5s
    if len(time_seconds) > 0 and np.max(time_seconds) >= 5.0:
        # Find the index closest to 5s
        idx_5s = np.argmin(np.abs(time_seconds - 5.0))
        if idx_5s < len(positions):
            ax2.scatter(positions[idx_5s, 0], positions[idx_5s, 1], c='red', s=150, marker='*', 
                       edgecolors='black', linewidths=2, label='t=5s', zorder=6)
    
    ax2.set_xlabel('X Position (meters)')
    ax2.set_ylabel('Y Position (meters)')
    ax2.set_title('Ego Agent Trajectory Path', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    ax2.legend()
    
    # Add distance information
    total_distance = analysis_data['total_distance']
    ax2.text(0.02, 0.98, f'Total Distance: {total_distance:.1f} m', 
            transform=ax2.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor='lightgreen', alpha=0.8))
    
    # Plot 3: Acceleration over time (if available)
    if accelerations is not None:
        ax3.plot(time_seconds, accelerations, 'r-', linewidth=2.5, label='Acceleration', alpha=0.8)
        ax3.fill_between(time_seconds, accelerations, alpha=0.3, color='red')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)  # Zero acceleration line
        
        # Add vertical line at t=5s
        if len(time_seconds) > 0 and np.max(time_seconds) >= 5.0:
            ax3.axvline(x=5.0, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='t=5s')
        
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Acceleration (m/s²)')
        ax3.set_title('Ego Agent Acceleration Profile', fontweight='bold', fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add acceleration statistics
        if acceleration_stats:
            a_mean = acceleration_stats['mean']
            a_max = acceleration_stats['max']
            a_min = acceleration_stats['min']
            a_std = acceleration_stats['std']
            
            acc_stats_text = f'Mean: {a_mean:.2f} m/s²\nMax: {a_max:.2f} m/s²\nMin: {a_min:.2f} m/s²\nStd: {a_std:.2f} m/s²'
            ax3.text(0.02, 0.98, acc_stats_text, transform=ax3.transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def create_integrated_ego_analysis(data_loader: DataLoader, 
                                  ego_visualizer: EgoOtherVisualizer,
                                  show_lanes: bool = True,
                                  show_agent_ids: bool = True,
                                  show_velocity: bool = True) -> None:
    """
    Create an integrated visualization showing both ego trajectory and velocity analysis.
    
    Parameters:
    - data_loader: DataLoader instance with loaded data
    - ego_visualizer: EgoOtherVisualizer instance
    - show_lanes: Whether to show lane information
    - show_agent_ids: Whether to show agent ID labels
    - show_velocity: Whether to include velocity analysis
    """
    
    if data_loader.current_data is None:
        print("No data loaded. Please load a scenario first.")
        return
    
    metadata = data_loader.current_metadata
    focal_idx = metadata['focal_agent_idx']
    
    # Get velocity analysis data if requested
    velocity_data = None
    if show_velocity:
        velocity_data = plot_ego_velocity_analysis(data_loader, show_acceleration=True)
    
    # Create the combined figure layout
    if show_velocity and velocity_data is not None:
        fig = plt.figure(figsize=(20, 12))
        
        # Create a more flexible grid layout
        # Top row: 2 columns for trajectories
        # Bottom row: 3 columns for analysis
        gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1, 1, 1])
        
        # Top row: Trajectories spanning full width
        # Ego trajectory (spanning 3 columns)
        ax_ego = fig.add_subplot(gs[0, :3])
        # Other agents trajectory (spanning 3 columns)
        ax_others = fig.add_subplot(gs[0, 3:])
        
        # Bottom row: Analysis plots (each spanning 2 columns for better proportions)
        # Velocity profile
        ax_velocity = fig.add_subplot(gs[1, :2])
        # Trajectory path
        ax_path = fig.add_subplot(gs[1, 2:4])
        # Acceleration profile
        ax_accel = fig.add_subplot(gs[1, 4:])
        
        # Plot ego and other trajectories (top row)
        ego_visualizer._plot_ego_trajectory(ax_ego, focal_idx, show_lanes, show_agent_ids)
        ax_ego.set_title('Ego Vehicle Trajectory', fontweight='bold', fontsize=12)
        
        ego_visualizer._plot_other_trajectories(ax_others, focal_idx, show_lanes, show_agent_ids)
        ax_others.set_title('Other Vehicles Trajectories', fontweight='bold', fontsize=12)
        
        # Plot velocity analysis (bottom row)
        time_seconds = velocity_data['time_seconds']
        velocities = velocity_data['velocities']
        accelerations = velocity_data['accelerations']
        positions = velocity_data['positions']
        
        # Velocity plot
        ax_velocity.plot(time_seconds, velocities, 'b-', linewidth=2, alpha=0.8)
        ax_velocity.fill_between(time_seconds, velocities, alpha=0.3, color='blue')
        
        # Add vertical line at t=5s
        if len(time_seconds) > 0 and np.max(time_seconds) >= 5.0:
            ax_velocity.axvline(x=5.0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='t=5s')
        
        ax_velocity.set_xlabel('Time (s)', fontsize=10)
        ax_velocity.set_ylabel('Velocity (m/s)', fontsize=10)
        ax_velocity.set_title('Ego Velocity Profile', fontweight='bold', fontsize=11)
        ax_velocity.grid(True, alpha=0.3)
        ax_velocity.legend(fontsize=9)
        
        # Add velocity statistics
        velocity_stats = velocity_data['velocity_stats']
        if velocity_stats:
            v_mean = velocity_stats['mean']
            v_max = velocity_stats['max']
            ax_velocity.axhline(y=v_mean, color='red', linestyle='--', alpha=0.7)
            ax_velocity.text(0.02, 0.98, f'Avg: {v_mean:.1f} m/s\nMax: {v_max:.1f} m/s', 
                           transform=ax_velocity.transAxes, verticalalignment='top', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        
        # Trajectory path colored by velocity
        ax_path.plot(positions[:, 0], positions[:, 1], 'g-', linewidth=4, alpha=0.8)
        ax_path.scatter(positions[0, 0], positions[0, 1], c='green', s=60, marker='o', 
                       edgecolors='black', linewidths=2, label='Start', zorder=5)
        ax_path.scatter(positions[-1, 0], positions[-1, 1], c='red', s=60, marker='^', 
                       edgecolors='black', linewidths=2, label='End', zorder=5)
        
        # Mark position at t=5s
        if len(time_seconds) > 0 and np.max(time_seconds) >= 5.0:
            idx_5s = np.argmin(np.abs(time_seconds - 5.0))
            if idx_5s < len(positions):
                ax_path.scatter(positions[idx_5s, 0], positions[idx_5s, 1], c='red', s=100, marker='*', 
                               edgecolors='black', linewidths=2, label='t=5s', zorder=6)
        
        # Color trajectory by velocity
        if len(positions) > 1 and len(velocities) > 0:
            max_vel = np.max(velocities) if np.max(velocities) > 0 else 1
            for i in range(len(positions) - 1):
                vel = velocities[i] if i < len(velocities) else 0
                color_intensity = min(vel / max_vel, 1)
                ax_path.plot(positions[i:i+2, 0], positions[i:i+2, 1], 
                           color=plt.cm.plasma(color_intensity), linewidth=3, alpha=0.8)
        
        ax_path.set_xlabel('X Position (m)', fontsize=10)
        ax_path.set_ylabel('Y Position (m)', fontsize=10)
        ax_path.set_title('Ego Trajectory Path (colored by velocity)', fontweight='bold', fontsize=11)
        ax_path.grid(True, alpha=0.3)
        ax_path.axis('equal')
        ax_path.legend(fontsize=9)
        
        # Add colorbar for velocity
        if len(velocities) > 0:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, 
                                     norm=plt.Normalize(vmin=0, vmax=np.max(velocities)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax_path, shrink=0.8)
            cbar.set_label('Velocity (m/s)', fontsize=9)
        
        # Acceleration plot
        if accelerations is not None:
            ax_accel.plot(time_seconds, accelerations, 'r-', linewidth=2, alpha=0.8)
            ax_accel.fill_between(time_seconds, accelerations, alpha=0.3, color='red')
            ax_accel.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add vertical line at t=5s
            if len(time_seconds) > 0 and np.max(time_seconds) >= 5.0:
                ax_accel.axvline(x=5.0, color='red', linestyle='--', linewidth=2, alpha=0.8, label='t=5s')
            
            ax_accel.set_xlabel('Time (s)', fontsize=10)
            ax_accel.set_ylabel('Acceleration (m/s²)', fontsize=10)
            ax_accel.set_title('Ego Acceleration Profile', fontweight='bold', fontsize=11)
            ax_accel.grid(True, alpha=0.3)
            ax_accel.legend(fontsize=9)
            
            # Add acceleration statistics
            acceleration_stats = velocity_data['acceleration_stats']
            if acceleration_stats:
                a_mean = acceleration_stats['mean']
                a_max = acceleration_stats['max']
                a_min = acceleration_stats['min']
                ax_accel.text(0.02, 0.98, f'Avg: {a_mean:.1f} m/s²\nMax: {a_max:.1f} m/s²\nMin: {a_min:.1f} m/s²', 
                             transform=ax_accel.transAxes, verticalalignment='top', fontsize=9,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
        
        # Add overall title
        scenario_id = metadata['scenario_id']
        city = metadata['city']
        duration = len(velocity_data['timesteps']) * 0.1
        title = f"Integrated Ego Analysis - Scenario: {scenario_id} (City: {city}) | Duration: {duration:.1f}s"
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Print motion summary
        print(f"\n=== Motion Summary ===")
        print(f"Ego Agent: {focal_idx} | Total Agents: {metadata['num_agents']}")
        if velocity_stats:
            print(f"Avg Speed: {velocity_stats['mean']:.1f} m/s ({velocity_stats['mean']*3.6:.1f} km/h)")
            print(f"Max Speed: {velocity_stats['max']:.1f} m/s ({velocity_stats['max']*3.6:.1f} km/h)")
            print(f"Distance: {velocity_data['total_distance']:.1f} m")
        
    else:
        # Just ego vs others layout
        fig, (ax_ego, ax_others) = plt.subplots(1, 2, figsize=(16, 8))
        ego_visualizer._plot_ego_trajectory(ax_ego, focal_idx, show_lanes, show_agent_ids)
        ax_ego.set_title('Ego Vehicle Trajectory', fontweight='bold', fontsize=11)
        
        ego_visualizer._plot_other_trajectories(ax_others, focal_idx, show_lanes, show_agent_ids)
        ax_others.set_title('Other Vehicles Trajectories', fontweight='bold', fontsize=11)
        
        scenario_id = metadata['scenario_id']
        city = metadata['city']
        fig.suptitle(f"Ego vs. Other Vehicles - Scenario: {scenario_id} (City: {city})", 
                    fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
