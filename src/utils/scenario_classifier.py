"""
Scenario Classification Utilities for DeMo Data

This module provides tools for classifying and organizing autonomous vehicle scenarios.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pandas as pd


class ScenarioClassifier:
    """
    Manages scenario classification using CSV format.
    
    Features:
    - Load/save classification records in CSV format
    - Classify scenarios without moving files
    - Track statistics and query status
    - Browse and export results
    """
    
    def __init__(self, output_base_path: Path, classification_log_path: Path, 
                 expert_categories: Dict[int, str]):
        """
        Initialize the scenario classifier.
        
        Args:
            output_base_path: Base directory for classification CSV file
            classification_log_path: Path to the CSV log file (should end with .csv)
            expert_categories: Dictionary mapping expert IDs to category names
        """
        self.output_base_path = Path(output_base_path)
        # Change to CSV format
        if classification_log_path.suffix == '.json':
            self.csv_path = Path(str(classification_log_path).replace('.json', '.csv'))
        else:
            self.csv_path = Path(classification_log_path)
        
        self.expert_categories = expert_categories
        self.classifications_df = self._load_classifications()
    
    def _load_classifications(self) -> pd.DataFrame:
        """Load existing classifications from CSV file."""
        if self.csv_path.exists():
            try:
                df = pd.read_csv(self.csv_path)
                print(f"📂 Loaded {len(df)} existing classifications from {self.csv_path}")
                return df
            except Exception as e:
                print(f"Warning: Could not load classifications: {e}")
        
        # Create empty DataFrame with minimal schema: [file_index, filename, expert_id]
        return pd.DataFrame(columns=['file_index', 'filename', 'expert_id'])
    
    def _save_classifications(self):
        """Save classifications to CSV file."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.classifications_df.to_csv(self.csv_path, index=False)
        print(f"💾 Saved classifications to {self.csv_path}")
    
    def classify_scenario(self, file_path: Path, expert_id: int, notes: str = "", file_index: int = None) -> bool:
        """
        Classify a scenario by recording it in the CSV file (does not move files).
        
        Args:
            file_path: Path to the scenario file
            expert_id: Expert category ID
            notes: Optional classification notes (ignored in minimal CSV format)
            file_index: Optional file index. If None, will be auto-assigned
        
        Returns:
            True if successful, False otherwise
        """
        if expert_id not in self.expert_categories:
            print(f"❌ Invalid expert ID: {expert_id}")
            return False
        
        try:
            # Check if file already classified
            filename = file_path.name
            if filename in self.classifications_df['filename'].values:
                # Update existing classification
                mask = self.classifications_df['filename'] == filename
                self.classifications_df.loc[mask, 'expert_id'] = expert_id
                print(f"🔄 Updated classification for {filename}")
            else:
                # Add new classification
                # Use provided file_index or auto-assign
                if file_index is None:
                    file_index = len(self.classifications_df)
                
                new_row = pd.DataFrame([{
                    'file_index': file_index,
                    'filename': filename,
                    'expert_id': expert_id
                }])
                self.classifications_df = pd.concat([self.classifications_df, new_row], 
                                                     ignore_index=True)
                print(f"✅ Classified {filename} as Expert {expert_id}")
            
            self._save_classifications()
            return True
            
        except Exception as e:
            print(f"❌ Error classifying {file_path.name}: {e}")
            return False
    
    def is_classified(self, file_path: Path) -> bool:
        """Check if a file has been classified."""
        return file_path.name in self.classifications_df['filename'].values
    
    def get_classification_info(self, file_path: Path) -> Optional[Dict]:
        """Get classification information for a specific file."""
        filename = file_path.name
        if filename not in self.classifications_df['filename'].values:
            return None
        
        row = self.classifications_df[self.classifications_df['filename'] == filename].iloc[0]
        return {
            'file_index': int(row['file_index']),
            'expert_id': int(row['expert_id']),
            'category': self.expert_categories[int(row['expert_id'])]
        }
    
    def get_unclassified_files(self, file_list: List[Path]) -> List[Path]:
        """Get list of unclassified files."""
        return [f for f in file_list if not self.is_classified(f)]
    
    def get_classification_stats(self) -> Dict[int, int]:
        """Get classification statistics."""
        stats = {expert_id: 0 for expert_id in self.expert_categories.keys()}
        
        if len(self.classifications_df) == 0:
            print("\n📊 Classification Statistics:")
            print(f"Total: 0 scenarios")
            return stats
        
        # Count by expert_id
        for expert_id in self.expert_categories.keys():
            stats[expert_id] = len(self.classifications_df[
                self.classifications_df['expert_id'] == expert_id
            ])
        
        total = len(self.classifications_df)
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
        if len(self.classifications_df) == 0:
            print("📁 No classified files yet.")
            return []
        
        # Filter by expert_id if specified
        if expert_id is not None:
            filtered_df = self.classifications_df[
                self.classifications_df['expert_id'] == expert_id
            ]
        else:
            filtered_df = self.classifications_df
        
        title = (f"Expert {expert_id} ({self.expert_categories[expert_id]})" 
                if expert_id else "All classified files")
        print(f"\n📁 {title}:")
        
        classified_files = []
        for i, (_, row) in enumerate(filtered_df.iterrows(), 1):
            filename = row['filename']
            file_index = int(row['file_index'])
            expert_id_val = int(row['expert_id'])
            category = self.expert_categories[expert_id_val]
            
            print(f"  {i:2d}. [{file_index:4d}] {filename} -> Expert {expert_id_val} ({category})")
            
            info_dict = {
                'file_index': file_index,
                'expert_id': expert_id_val,
                'category': category
            }
            classified_files.append((filename, info_dict))
        
        return classified_files
    
    def export_classification_summary(self, output_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
        """
        Export classification summary to CSV.
        
        Args:
            output_path: Custom output path (default: output_base_path/classification_summary.csv)
        
        Returns:
            DataFrame with classification summary
        """
        if len(self.classifications_df) == 0:
            print("No classified scenarios to export.")
            return None
        
        output_path = output_path or (self.output_base_path / "classification_summary.csv")
        self.classifications_df.to_csv(output_path, index=False)
        print(f"📊 Summary exported to: {output_path}")
        print(f"Total: {len(self.classifications_df)} scenarios")
        return self.classifications_df.copy()


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
                           expert_id: int, notes: str = "", file_index: int = None) -> bool:
    """
    Manually classify the currently loaded scenario.
    
    Args:
        data_loader: DataLoader instance with current_file_path attribute
        classifier: ScenarioClassifier instance
        expert_id: Expert category ID (1-9)
        notes: Optional notes about the classification (ignored in minimal format)
        file_index: Optional file index (will be auto-assigned if None)
        
    Returns:
        True if successful, False otherwise
    """
    if not hasattr(data_loader, 'current_file_path') or data_loader.current_file_path is None:
        print("❌ No scenario currently loaded. Please visualize a scenario first.")
        return False
    
    print(f"🔄 Classifying: {data_loader.current_file_path.name}")
    return classifier.classify_scenario(data_loader.current_file_path, expert_id, notes, file_index)
