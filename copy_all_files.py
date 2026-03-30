import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def copy_all_files(source_dir, dest_dir):
    """
    Recursively finds all files in the source directory 
    and copies them to the destination directory.
    """
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    
    if not source_path.exists() or not source_path.is_dir():
        print(f"CRITICAL ERROR: Source directory '{source_dir}' does not exist.")
        return

    # Gather all files recursively (ignoring directories)
    print(f"Scanning '{source_dir}' for files...")
    all_files = [f for f in source_path.rglob('*') if f.is_file()]
    
    total_files = len(all_files)
    if total_files == 0:
        print(f"No files found in '{source_dir}'. Exiting.")
        return
        
    print(f"Found {total_files} total files to copy.")
    
    os.makedirs(dest_path, exist_ok=True)
    
    # Copy all files
    for file_path in tqdm(all_files, desc=f"Copying {total_files} files"):
        target_path = dest_path / file_path.name
        
        # Anti-overwrite safety net (in case of duplicate filenames in subfolders)
        counter = 1
        while target_path.exists():
            target_path = dest_path / f"{file_path.stem}_{counter}{file_path.suffix}"
            counter += 1
            
        shutil.copy2(file_path, target_path)

    print("\n" + "=" * 60)
    print("DATASET COPY COMPLETE")
    print("=" * 60)
    print(f"Source:      {os.path.abspath(source_dir)}")
    print(f"Destination: {os.path.abspath(dest_dir)}")
    print(f"Total Copied:{total_files} files")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy all files from a source folder to a destination folder.")
    parser.add_argument("--source_dir", type=str, required=True, help="Path to the folder containing the files to copy.")
    parser.add_argument("--dest_dir", type=str, required=True, help="Path where the files will be copied.")
    
    args = parser.parse_args()
    copy_all_files(args.source_dir, args.dest_dir)
