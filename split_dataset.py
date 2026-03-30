import os
import random
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def split_and_copy_dataset(source_dir, dest_dir, train_ratio=0.95):
    """
    Randomly splits all files from the source directory into training 
    and validation sets and copies them to the destination.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists() or not source_path.is_dir():
        print(f"CRITICAL ERROR: Source directory '{source_dir}' does not exist.")
        return

    # Gather all files recursively (ignoring subdirectories themselves)
    print(f"Scanning '{source_dir}' for files...")
    all_files = [f for f in source_path.rglob('*') if f.is_file()]
    
    if not all_files:
        print(f"No files found in '{source_dir}'. Exiting.")
        return
        
    total_files = len(all_files)
    print(f"Found {total_files} files.")
    
    # Shuffle to ensure a random distribution
    random.shuffle(all_files)
    
    # Calculate split indices
    train_count = int(total_files * train_ratio)
    train_files = all_files[:train_count]
    val_files = all_files[train_count:]
    
    # Setup destination directories
    train_dir = dest_path / "training"
    val_dir = dest_path / "validation"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    print(f"\nSplitting Dataset:")
    print(f" - Training ({(train_ratio * 100):.1f}%): {len(train_files)} files")
    print(f" - Validation ({((1.0 - train_ratio) * 100):.1f}%): {len(val_files)} files\n")
    
    # Copy Training Files
    for file_path in tqdm(train_files, desc="Copying to 'training'  "):
        # We flatten the directory structure so all files end up directly in the target folder.
        # If two files have the same name in different subfolders, this will overwrite.
        # Adding a safety check to rename if conflict occurs is recommended if flattening.
        target_path = train_dir / file_path.name
        
        # Anti-overwrite safety net
        counter = 1
        while target_path.exists():
            target_path = train_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
            counter += 1
            
        shutil.copy2(file_path, target_path)

    # Copy Validation Files
    for file_path in tqdm(val_files, desc="Copying to 'validation'"):
        target_path = val_dir / file_path.name
        
        # Anti-overwrite safety net
        counter = 1
        while target_path.exists():
            target_path = val_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
            counter += 1
            
        shutil.copy2(file_path, target_path)

    print("\n" + "=" * 60)
    print("DATASET SPLIT COMPLETE")
    print("=" * 60)
    print(f"Training Data:   {os.path.abspath(train_dir)}")
    print(f"Validation Data: {os.path.abspath(val_dir)}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly split and copy files into training and validation sets.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to the folder containing all raw files.")
    parser.add_argument("--dest_dir", type=str, required=True, help="Path where 'training' and 'validation' folders will be created.")
    parser.add_argument("--train_split", type=float, default=0.95, help="Percentage of files to put in the training set (default: 0.95).")
    
    args = parser.parse_args()
    
    # Ensure the split is a valid percentage
    if not (0.0 < args.train_split < 1.0):
        print("ERROR: --train_split must be between 0.0 and 1.0 (e.g., 0.95 for 95%).")
    else:
        split_and_copy_dataset(args.source_dir, args.dest_dir, args.train_split)
