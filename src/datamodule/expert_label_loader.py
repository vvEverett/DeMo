"""
Utility to load expert labels for supervised MoE training
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


class ExpertLabelLoader:
    """
    Loads expert labels from CSV classification file
    Maps scenario filenames to expert IDs (1-5)
    
    Expert definitions:
    - Expert 1: Lane Keeping (Straight/Lane Change)
    - Expert 2: Turn Left
    - Expert 3: Turn Right
    - Expert 4: Constraint-Driven Deceleration (Stop/Yield/Junction)
    - Expert 5: Others (Long-tail behaviors)
    """
    
    def __init__(self, classification_csv: Optional[Path] = None):
        """
        Args:
            classification_csv: Path to CSV file with columns [file_index, filename, expert_id]
        """
        self.label_map: Dict[str, int] = {}
        
        if classification_csv is not None and Path(classification_csv).exists():
            self._load_labels(classification_csv)
            print(f"Loaded expert labels for {len(self.label_map)} scenarios from {classification_csv}")
        else:
            print("No classification CSV provided. Expert labels will be None (inference mode).")
    
    def _load_labels(self, csv_path: Path):
        """Load expert labels from CSV file"""
        df = pd.read_csv(csv_path)
        
        # Create mapping from filename to expert_id
        for _, row in df.iterrows():
            filename = row['filename']
            expert_id = int(row['expert_id'])
            self.label_map[filename] = expert_id
    
    def get_label(self, filename: str) -> Optional[int]:
        """
        Get expert label for a given scenario filename
        
        Args:
            filename: Scenario filename (e.g., 'scenario_xxxxx.pt')
        
        Returns:
            expert_id (1-5) if available, None otherwise
        """
        return self.label_map.get(filename, None)
    
    def has_labels(self) -> bool:
        """Check if any labels are loaded"""
        return len(self.label_map) > 0


# Default classification file path
DEFAULT_CLASSIFICATION_CSV = Path("data/DeMo_classified/heuristic_classifications_latest.csv")
