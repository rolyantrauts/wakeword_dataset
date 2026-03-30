import os
import shutil
import torch
import torchaudio
import argparse
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

# =============================================================================
# 1. ACOUSTIC ENVELOPE DETECTION
# =============================================================================
def find_acoustic_edges(waveform, sr, threshold=0.01, window_ms=10.0, leeway_ms=50.0):
    """
    Scans the waveform using RMS energy windows to find the absolute first and 
    last sounds in the file, ignoring internal pauses.
    """
    window_samples = int(sr * (window_ms / 1000.0))
    leeway_samples = int(sr * (leeway_ms / 1000.0))
    
    # 1. Calculate RMS Energy in rolling blocks
    # Square the waveform
    sq_wave = waveform ** 2
    # Apply average pooling (this acts as our rolling window)
    mean_sq = torch.nn.functional.avg_pool1d(sq_wave.unsqueeze(0), kernel_size=window_samples, stride=window_samples).squeeze(0)
    rms_contour = torch.sqrt(mean_sq)[0] # Extract the 1D tensor
    
    # 2. Find all windows that exceed our noise threshold
    active_windows = torch.where(rms_contour > threshold)[0]
    
    if len(active_windows) == 0:
        # Complete silence or broken file
        return None, None
        
    # 3. Get the absolute first and last active windows
    first_window = active_windows[0].item()
    last_window = active_windows[-1].item()
    
    # 4. Convert window indices back to raw audio sample indices
    raw_start_sample = first_window * window_samples
    raw_end_sample = (last_window + 1) * window_samples
    
    # 5. Apply the safety leeway (pre and post roll)
    final_start = max(0, raw_start_sample - leeway_samples)
    final_end = min(waveform.shape[1], raw_end_sample + leeway_samples)
    
    return final_start, final_end

# =============================================================================
# 2. MASTER TRIMMING PIPELINE
# =============================================================================
def trim_and_center_dataset(source_dir, dest_dir, unprocessed_dir, target_duration=1.4, target_sr=16000):
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(unprocessed_dir, exist_ok=True)
    
    target_samples = int(target_duration * target_sr)
    
    wav_files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    print(f"\nFound {len(wav_files)} pre-washed files in {source_dir} to process.\n")
    
    stats = {"centered": 0, "too_long": 0, "silent_failed": 0, "skipped": 0}
    
    for filename in tqdm(wav_files, desc="Trimming & Centering"):
        filepath = os.path.join(source_dir, filename)
        out_filepath = os.path.join(dest_dir, filename)
        unprocessed_filepath = os.path.join(unprocessed_dir, filename)
        
        # Skip logic for resuming interrupted runs
        if os.path.exists(out_filepath) or os.path.exists(unprocessed_filepath):
            stats["skipped"] += 1
            continue
            
        try:
            waveform, sr = torchaudio.load(filepath)
            
            # 1. Find the edges
            edges = find_acoustic_edges(waveform, sr, threshold=0.01, window_ms=10.0, leeway_ms=50.0)
            
            if edges[0] is None:
                # File had no audio exceeding the threshold
                shutil.copy2(filepath, unprocessed_filepath)
                stats["silent_failed"] += 1
                continue
                
            trim_start_idx, trim_end_idx = edges
            
            # 2. Execute the Crop
            cropped_wave = waveform[:, trim_start_idx:trim_end_idx]
            cropped_samples = cropped_wave.shape[1]
            
            # 3. Length Gate & Centering
            if cropped_samples > target_samples:
                # The phrase itself (plus leeway) is longer than 1.4s
                shutil.copy2(filepath, unprocessed_filepath)
                stats["too_long"] += 1
            else:
                # Mathematical Centering Padding
                left_pad = (target_samples - cropped_samples) // 2
                right_pad = target_samples - cropped_samples - left_pad
                
                centered_wave = torch.nn.functional.pad(cropped_wave, (left_pad, right_pad), "constant", 0.0)
                
                torchaudio.save(out_filepath, centered_wave, sr, encoding="PCM_S", bits_per_sample=16)
                stats["centered"] += 1
                
        except Exception as e:
            tqdm.write(f"  [ERROR] Processing {filename}: {e}")
            shutil.copy2(filepath, unprocessed_filepath)
            stats["silent_failed"] += 1

    print("\n" + "="*60)
    print("DATASET TRIMMING COMPLETE")
    print("="*60)
    print(f"Perfectly Centered (1.4s): {stats['centered']}")
    print(f"Copied to Unprocessed (Too Long > 1.4s): {stats['too_long']}")
    print(f"Copied to Unprocessed (Silence/Error): {stats['silent_failed']}")
    print(f"Skipped (Already Processed): {stats['skipped']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RMS Envelope Trimming & 1.4s Centering")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing washed TTS files")
    parser.add_argument("--dest_dir", type=str, required=True, help="Directory to output perfect 1.4s centered files")
    parser.add_argument("--unprocessed_dir", type=str, default="./dataset/unprocessed", help="Directory for files that are too long")
    args = parser.parse_args()
    
    trim_and_center_dataset(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        unprocessed_dir=args.unprocessed_dir,
        target_duration=1.4,
        target_sr=16000
    )