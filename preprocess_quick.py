"""
Usage:
1) Generate file lists with s5cmd (no AWS signing required):
   s5cmd --no-sign-request ls "s3://argoverse/datasets/av2/motion-forecasting/train/*/*.parquet" > train_files.txt
   s5cmd --no-sign-request ls "s3://argoverse/datasets/av2/motion-forecasting/val/*/*.parquet" > val_files.txt
   s5cmd --no-sign-request ls "s3://argoverse/datasets/av2/motion-forecasting/test/*/*.parquet" > test_files.txt

2) Run preprocessing:
   python preprocess_quick.py --data_root=/data/av2 -p
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import List
import multiprocessing
import time
from tqdm import tqdm
from src.datamodule.av2_extractor import Av2Extractor


def glob_files(data_root: Path, mode: str):
    # Read the s5cmd-generated file list to speed up discovery
    list_file = Path(f"{mode}_files.txt")
    
    if list_file.exists():
        print(f"Using pre-generated file list: {list_file}")
        scenario_files = []
        
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Method 1: handle recursive s5cmd output
                if line.endswith('.parquet') and not line.startswith('DIR'):
                    # Extract file path from s5cmd recursive output
                    parts = line.split()
                    if len(parts) >= 3:
                        # Use the last token as relative path
                        relative_path = parts[-1]
                        if relative_path.endswith('.parquet'):
                            # Build full path: data_root/mode/<scenario_id>/<file>.parquet
                            full_path = data_root / mode / relative_path
                            scenario_files.append(full_path)
                
                # Method 2: handle directory listing format
                elif "DIR" in line and "/" in line:
                    # Extract scenario ID from "DIR  <uuid>/" format
                    scenario_id = line.split()[-1].rstrip('/')
                    if scenario_id and len(scenario_id) == 36:
                        scenario_dir = data_root / mode / scenario_id
                        if scenario_dir.exists():
                            parquet_files = list(scenario_dir.glob("*.parquet"))
                            scenario_files.extend(parquet_files)
        
        print(f"Found {len(scenario_files)} scenario files from list")
        return scenario_files
    else:
        print(f"File list {list_file} not found, falling back to original glob")
        # Fallback to the original method
        file_root = data_root / mode
        scenario_files = list(file_root.rglob("*.parquet"))
        return scenario_files


def preprocess(args):
    batch = args.batch
    data_root = Path(args.data_root)

    for mode in ["train", "val", "test"]:
        print(f"\nProcessing {mode} dataset...")
        save_dir = Path("data/DeMo_processed") / mode
        extractor = Av2Extractor(save_path=save_dir, mode=mode)

        save_dir.mkdir(exist_ok=True, parents=True)
        
        # Time the file discovery process
        start_time = time.time()
        scenario_files = glob_files(data_root, mode)
        file_discovery_time = time.time() - start_time
        
        print(f"File discovery took {file_discovery_time:.2f}s")
        print(f"Found {len(scenario_files)} scenario files")
        
        if len(scenario_files) == 0:
            print(f"Warning: no files found for mode {mode}")
            continue

        if args.parallel:
            print(f"Processing in parallel with 16 workers...")
            with multiprocessing.Pool(16) as p:
                all_name = list(tqdm(p.imap(extractor.save, scenario_files), total=len(scenario_files)))
        else:
            print(f"Processing with a single process...")
            for file in tqdm(scenario_files):
                extractor.save(file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, default='/path/to/data_root')
    parser.add_argument("--batch", "-b", type=int, default=50)
    parser.add_argument("--parallel", "-p", action="store_true")

    args = parser.parse_args()
    preprocess(args)
