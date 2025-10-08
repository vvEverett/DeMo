"""
Scenario Classification Utilities for DeMo Data

This module provides tools for classifying and organizing autonomous vehicle scenarios.
"""

import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd


class ScenarioClassifier:
    """
    Manages scenario classification and file organization.
    
    Features:
    - Load/save classification records
    - Classify scenarios and organize files
    - Track statistics and query status
    - Browse and export results
    """
    
    def __init__(self, output_base_path: Path, classification_log_path: Path, 
                 expert_categories: Dict[int, str]):
        """
        Initialize the scenario classifier.
        
        Args:
            output_base_path: Base directory for organized classified files
            classification_log_path: Path to the JSON log file
            expert_categories: Dictionary mapping expert IDs to category names
        """
        self.output_base_path = Path(output_base_path)
        self.log_path = Path(classification_log_path)
        self.expert_categories = expert_categories
        self.classifications = self._load_classifications()
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        self.output_base_path.mkdir(parents=True, exist_ok=True)
        for expert_id, category_name in self.expert_categories.items():
            category_dir = self.output_base_path / f"Expert_{expert_id}_{category_name}"
            category_dir.mkdir(exist_ok=True)
    
    def _load_classifications(self) -> Dict:
        """Load existing classifications from log file."""
        if self.log_path.exists():
            try:
                with open(self.log_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load classifications: {e}")
        return {
            "metadata": {"created": datetime.now().isoformat()}, 
            "classifications": {}
        }
    
    def _save_classifications(self):
        """Save classifications to log file."""
        self.classifications["metadata"]["last_updated"] = datetime.now().isoformat()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, 'w') as f:
            json.dump(self.classifications, f, indent=2)
    
    def classify_scenario(self, file_path: Path, expert_id: int, notes: str = "") -> bool:
        """
        Classify a scenario and copy file to appropriate directory.
        
        Args:
            file_path: Path to the scenario file
            expert_id: Expert category ID
            notes: Optional classification notes
        
        Returns:
            True if successful, False otherwise
        """
        if expert_id not in self.expert_categories:
            print(f"❌ Invalid expert ID: {expert_id}")
            return False
        
        category_name = self.expert_categories[expert_id]
        target_dir = self.output_base_path / f"Expert_{expert_id}_{category_name}"
        target_file = target_dir / file_path.name
        
        try:
            shutil.copy2(file_path, target_file)
            
            self.classifications["classifications"][file_path.name] = {
                "expert_id": expert_id,
                "category": category_name,
                "original_path": str(file_path),
                "target_path": str(target_file),
                "classified_at": datetime.now().isoformat(),
                "notes": notes
            }
            
            self._save_classifications()
            print(f"✅ Classified {file_path.name} as Expert {expert_id} ({category_name})")
            return True
            
        except Exception as e:
            print(f"❌ Error classifying {file_path.name}: {e}")
            return False
    
    def is_classified(self, file_path: Path) -> bool:
        """Check if a file has been classified."""
        return file_path.name in self.classifications["classifications"]
    
    def get_classification_info(self, file_path: Path) -> Optional[Dict]:
        """Get classification information for a specific file."""
        return self.classifications["classifications"].get(file_path.name)
    
    def get_unclassified_files(self, file_list: List[Path]) -> List[Path]:
        """Get list of unclassified files."""
        return [f for f in file_list if not self.is_classified(f)]
    
    def get_classification_stats(self) -> Dict[int, int]:
        """Get classification statistics."""
        stats = {expert_id: 0 for expert_id in self.expert_categories.keys()}
        total = 0
        
        for info in self.classifications["classifications"].values():
            stats[info["expert_id"]] += 1
            total += 1
        
        print("\n📊 Classification Statistics:")
        print(f"Total: {total} scenarios")
        for expert_id, count in stats.items():
            category = self.expert_categories[expert_id]
            print(f"  Expert {expert_id} ({category}): {count}")
        
        return stats
    
    def browse_classified_files(self, expert_id: Optional[int] = None) -> List[Tuple[str, Dict]]:
        """
        Browse classified files.
        
        Args:
            expert_id: Filter by specific expert ID (None for all)
        
        Returns:
            List of (filename, classification_info) tuples
        """
        classified_files = [
            (name, info) for name, info in self.classifications["classifications"].items()
            if expert_id is None or info["expert_id"] == expert_id
        ]
        
        title = (f"Expert {expert_id} ({self.expert_categories[expert_id]})" 
                if expert_id else "All classified files")
        print(f"\n📁 {title}:")
        
        for i, (name, info) in enumerate(classified_files, 1):
            timestamp = info["classified_at"][:19]
            notes = info["notes"] or "No notes"
            print(f"  {i:2d}. {name}")
            print(f"      Expert {info['expert_id']} ({info['category']}) | {timestamp}")
            print(f"      Notes: {notes}")
        
        return classified_files
    
    def export_classification_summary(self, output_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """
        Export classification summary to CSV.
        
        Args:
            output_path: Custom output path (default: output_base_path/classification_summary.csv)
        
        Returns:
            DataFrame with classification summary
        """
        if not self.classifications["classifications"]:
            print("No classified scenarios to export.")
            return None
        
        summary_data = [
            {
                'file_name': name,
                'expert_id': info['expert_id'],
                'category': info['category'],
                'classified_at': info['classified_at'],
                'notes': info['notes'],
                'original_path': info['original_path'],
                'target_path': info['target_path']
            }
            for name, info in self.classifications["classifications"].items()
        ]
        
        df = pd.DataFrame(summary_data)
        output_path = output_path or (self.output_base_path / "classification_summary.csv")
        df.to_csv(output_path, index=False)
        print(f"📊 Summary exported to: {output_path}")
        print(f"Total: {len(summary_data)} scenarios")
        return df


def list_files_with_status(file_list: List[Path], 
                          classifier: Optional[ScenarioClassifier] = None,
                          max_files: int = 20) -> None:
    """
    List files with their classification status.
    
    Args:
        file_list: List of file paths
        classifier: Optional ScenarioClassifier to show status
        max_files: Maximum number of files to display
    """
    print(f"\n📁 Available Files (Total: {len(file_list)}):")
    print(f"Showing first {min(max_files, len(file_list))} files:\n")
    
    for i, file_path in enumerate(file_list[:max_files]):
        status_info = ""
        if classifier:
            if classifier.is_classified(file_path):
                info = classifier.get_classification_info(file_path)
                status_info = f" - ✅ Expert {info['expert_id']}: {info['category']}"
            else:
                status_info = " - ⭕ Unclassified"
        
        print(f"  [{i:3d}] {file_path.name}{status_info}")
    
    if len(file_list) > max_files:
        print(f"\n  ... and {len(file_list) - max_files} more")
    
    if classifier:
        unclassified = len(classifier.get_unclassified_files(file_list))
        classified = len(file_list) - unclassified
        print(f"\n📊 Summary: {classified} classified, {unclassified} unclassified")


def print_expert_categories(expert_categories: Dict[int, str]) -> None:
    """
    Print all available expert categories.
    
    Args:
        expert_categories: Dictionary mapping expert IDs to category names
    """
    print("\n🏷️  Available Expert Categories:")
    for expert_id, category in expert_categories.items():
        print(f"  {expert_id}: {category}")
    print()


def manual_classify_current(data_loader, classifier: ScenarioClassifier,
                           expert_id: int, notes: str = "") -> bool:
    """
    Manually classify the currently loaded scenario.
    
    Args:
        data_loader: DataLoader instance with current_file_path attribute
        classifier: ScenarioClassifier instance
        expert_id: Expert category ID (1-9)
        notes: Optional notes about the classification
        
    Returns:
        True if successful, False otherwise
    """
    if not hasattr(data_loader, 'current_file_path') or data_loader.current_file_path is None:
        print("❌ No scenario currently loaded. Please visualize a scenario first.")
        return False
    
    print(f"🔄 Classifying: {data_loader.current_file_path.name}")
    return classifier.classify_scenario(data_loader.current_file_path, expert_id, notes)
