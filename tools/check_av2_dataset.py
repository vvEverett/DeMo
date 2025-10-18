from pathlib import Path
import os
import time

# Checking the original Argoverse 2 Motion Forecasting Dataset
# This script counts the number of scenarios in each split (train, val, test)

data_dir = Path("/data/av2")

print("=== Checking original Argoverse 2 Motion Forecasting Dataset ===\n")

start_time = time.time()

for split in ["train", "val", "test"]:
    split_dir = data_dir / split
    if split_dir.exists():
        split_start_time = time.time()
        
        # Using os.scandir() for faster directory counting
        with os.scandir(split_dir) as entries:
            scenario_count = sum(1 for entry in entries if entry.is_dir())
        
        print(f"{split}: {scenario_count} scenarios")
        
        # Check for parquet and map files
        sample_dirs = []
        with os.scandir(split_dir) as entries:
            for i, entry in enumerate(entries):
                if entry.is_dir() and i < 3:  # Only check the first 3 directories as samples
                    sample_dirs.append(entry.path)
                elif i >= 3:
                    break
        
        parquet_count = 0
        map_count = 0
        for sample_dir in sample_dirs:
            sample_path = Path(sample_dir)
            parquet_files = list(sample_path.glob("scenario_*.parquet"))
            map_files = list(sample_path.glob("log_map_archive_*.json"))
            parquet_count += len(parquet_files)
            map_count += len(map_files)
        
        if sample_dirs:
            avg_parquet = parquet_count / len(sample_dirs)
            avg_map = map_count / len(sample_dirs)
            estimated_parquet = int(avg_parquet * scenario_count)
            estimated_map = int(avg_map * scenario_count)
            print(f"  - estimated parquet files: ~{estimated_parquet} (based on {len(sample_dirs)} samples)")
            print(f"  - estimated map files: ~{estimated_map} (based on {len(sample_dirs)} samples)")
        
        split_time = time.time() - split_start_time
        print(f"  - processed in {split_time:.2f}s")
        
    else:
        print(f"{split}: directory not found")
    print()

total_time = time.time() - start_time
print(f"Total processing time: {total_time:.2f}s")
print("\nExpected counts:")
print("train: ~199,908")
print("val: ~24,988")
print("test: ~24,984")
