import argparse
import os
import pickle
from pathlib import Path

import torch
from tqdm import tqdm


def check_corrupted_files(data_dir, delete_corrupted=False, move_to_quarantine=False):
    """
    Detect corrupted .pt files
    
    Args:
        data_dir: Data directory path
        delete_corrupted: Whether to delete corrupted files
        move_to_quarantine: Whether to move corrupted files to quarantine directory
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # Get all .pt files
    pt_files = list(data_path.glob('*.pt'))
    print(f"Found {len(pt_files)} .pt files")
    
    corrupted_files = []
    quarantine_dir = None
    
    if move_to_quarantine:
        quarantine_dir = data_path / "corrupted_files"
        quarantine_dir.mkdir(exist_ok=True)
        print(f"Quarantine directory: {quarantine_dir}")
    
    # Check each file
    print("Checking file integrity...")
    for file_path in tqdm(pt_files):
        try:
            # Try to load the file
            data = torch.load(file_path, map_location='cpu')
            # Simple data structure validation
            if not isinstance(data, dict):
                raise ValueError("Incorrect data format")
                
        except (EOFError, RuntimeError, pickle.UnpicklingError, 
                ValueError, OSError, Exception) as e:
            corrupted_files.append(file_path)
            print(f"\nCorrupted file: {file_path}")
            print(f"Error type: {type(e).__name__}: {e}")
            
            # Handle corrupted files
            if delete_corrupted:
                try:
                    file_path.unlink()
                    print(f"Deleted: {file_path}")
                except Exception as del_e:
                    print(f"Failed to delete: {del_e}")
                    
            elif move_to_quarantine and quarantine_dir:
                try:
                    quarantine_path = quarantine_dir / file_path.name
                    file_path.rename(quarantine_path)
                    print(f"Moved to: {quarantine_path}")
                except Exception as move_e:
                    print(f"Failed to move: {move_e}")
    
    # Output results
    print(f"\n{'='*50}")
    print(f"Check completed!")
    print(f"Total files: {len(pt_files)}")
    print(f"Corrupted files: {len(corrupted_files)}")
    print(f"Intact files: {len(pt_files) - len(corrupted_files)}")
    
    if corrupted_files:
        print(f"\nCorrupted files list:")
        for file_path in corrupted_files:
            print(f"  - {file_path}")
        
        if not delete_corrupted and not move_to_quarantine:
            print(f"\nYou can use the following commands to delete corrupted files:")
            for file_path in corrupted_files:
                print(f"rm '{file_path}'")
    else:
        print("No corrupted files found!")
    
    return len(corrupted_files)


def main():
    parser = argparse.ArgumentParser(description='Detect corrupted PyTorch data files')
    parser.add_argument('--data_dir', '-d', type=str, 
                       default='data/DeMo_processed/val',
                       help='Data directory path')
    parser.add_argument('--delete', action='store_true',
                       help='Automatically delete corrupted files')
    parser.add_argument('--quarantine', action='store_true',
                       help='Move corrupted files to quarantine directory instead of deleting')
    parser.add_argument('--check_all', action='store_true',
                       help='Check all directories: train, val, test')
    
    args = parser.parse_args()
    
    if args.check_all:
        # Check all datasets
        base_dir = Path('data/DeMo_processed')
        total_corrupted = 0
        for split in ['train', 'val', 'test']:
            split_dir = base_dir / split
            if split_dir.exists():
                print(f"\n{'='*20} Checking {split} dataset {'='*20}")
                corrupted_count = check_corrupted_files(split_dir, args.delete, args.quarantine)
                total_corrupted += corrupted_count
            else:
                print(f"Skipping non-existent directory: {split_dir}")
        
        print(f"\n{'='*60}")
        print(f"All dataset checks completed! Total corrupted files found: {total_corrupted}")
    else:
        check_corrupted_files(args.data_dir, args.delete, args.quarantine)


if __name__ == "__main__":
    main()
