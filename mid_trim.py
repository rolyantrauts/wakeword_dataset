import os
import shutil
import torch
import torchaudio
import argparse
import warnings
from tqdm import tqdm

# Silence standard functional warnings and the specific TorchCodec save warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
warnings.filterwarnings("ignore", message=".*encoding.*")
warnings.filterwarnings("ignore", message=".*bits_per_sample.*")

# =============================================================================
# 1. ACOUSTIC ENVELOPE & GAP DETECTION
# =============================================================================
def find_acoustic_edges(waveform, sr, threshold=0.01, window_ms=10.0, leeway_ms=50.0):
    """Finds the absolute first and last sounds in the file."""
    window_samples = int(sr * (window_ms / 1000.0))
    leeway_samples = int(sr * (leeway_ms / 1000.0))
    
    sq_wave = waveform ** 2
    mean_sq = torch.nn.functional.avg_pool1d(sq_wave.unsqueeze(0), kernel_size=window_samples, stride=window_samples).squeeze(0)
    rms_contour = torch.sqrt(mean_sq)[0]
    
    active_windows = torch.where(rms_contour > threshold)[0]
    if len(active_windows) == 0:
        return None, None
        
    raw_start = active_windows[0].item() * window_samples
    raw_end = (active_windows[-1].item() + 1) * window_samples
    
    final_start = max(0, raw_start - leeway_samples)
    final_end = min(waveform.shape[1], raw_end + leeway_samples)
    
    return final_start, final_end

def find_longest_silence_gap(waveform, sr, threshold=0.015, window_ms=10.0):
    """Identifies the longest contiguous block of silence (the pause between words)."""
    window_samples = int(sr * (window_ms / 1000.0))
    
    sq_wave = waveform ** 2
    mean_sq = torch.nn.functional.avg_pool1d(sq_wave.unsqueeze(0), kernel_size=window_samples, stride=window_samples).squeeze(0)
    rms_contour = torch.sqrt(mean_sq)[0]
    
    is_silence = rms_contour < threshold
    
    padded = torch.cat([torch.tensor([False], device=is_silence.device), is_silence, torch.tensor([False], device=is_silence.device)])
    diff = torch.diff(padded.int())
    
    starts = torch.where(diff == 1)[0]
    ends = torch.where(diff == -1)[0]
    
    if len(starts) == 0:
        return 0, 0
        
    lengths = ends - starts
    max_idx = torch.argmax(lengths)
    
    gap_start_sample = starts[max_idx].item() * window_samples
    gap_end_sample = ends[max_idx].item() * window_samples
    
    return gap_start_sample, gap_end_sample

# =============================================================================
# 2. SQUEEZE & CENTER PIPELINE
# =============================================================================
def squeeze_and_center_dataset(source_dir, dest_dir, unprocessed_dir, target_duration=1.4, target_sr=16000):
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(unprocessed_dir, exist_ok=True)
    
    target_samples = int(target_duration * target_sr) # 22400
    
    # Tuning parameters for the crossfade
    crossfade_samples = int(0.02 * target_sr) # 20ms crossfade
    word_leeway_samples = int(0.05 * target_sr) # 50ms protected zone around words
    
    wav_files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    print(f"\nFound {len(wav_files)} files in {source_dir} to evaluate.\n")
    
    stats = {"centered_normally": 0, "squeezed": 0, "unsqueezable": 0, "silent_failed": 0, "skipped": 0}
    
    for filename in tqdm(wav_files, desc="Squeezing Silence & Centering"):
        filepath = os.path.join(source_dir, filename)
        out_filepath = os.path.join(dest_dir, filename)
        unprocessed_filepath = os.path.join(unprocessed_dir, filename)
        
        if os.path.exists(out_filepath) or os.path.exists(unprocessed_filepath):
            stats["skipped"] += 1
            continue
            
        try:
            waveform, sr = torchaudio.load(filepath)
            
            # 1. Strip the outer leading/trailing silence first
            edges = find_acoustic_edges(waveform, sr)
            if edges[0] is None:
                shutil.copy2(filepath, unprocessed_filepath)
                stats["silent_failed"] += 1
                continue
                
            trimmed_wave = waveform[:, edges[0]:edges[1]]
            current_samples = trimmed_wave.shape[1]
            
            # 2. Evaluate Length
            if current_samples <= target_samples:
                # File fits perfectly, just center and pad it
                left_pad = (target_samples - current_samples) // 2
                right_pad = target_samples - current_samples - left_pad
                final_wave = torch.nn.functional.pad(trimmed_wave, (left_pad, right_pad), "constant", 0.0)
                torchaudio.save(out_filepath, final_wave, sr, encoding="PCM_S", bits_per_sample=16)
                stats["centered_normally"] += 1
                
            else:
                # File is too long. We need to perform crossfaded surgery.
                excess_samples = current_samples - target_samples
                gap_start, gap_end = find_longest_silence_gap(trimmed_wave, sr)
                gap_length = gap_end - gap_start
                
                # Check if the gap is large enough to remove the excess AND respect our safety leeway
                if gap_length - excess_samples >= (2 * word_leeway_samples):
                    
                    cut_center = gap_start + (gap_length // 2)
                    
                    # Because we are overlapping by 'C' samples, the amount of pure space we skip is E - C
                    skip_samples = excess_samples - crossfade_samples
                    
                    # Define the boundaries
                    A = cut_center - (skip_samples // 2)
                    B = A + skip_samples
                    
                    # Create linear fade envelopes
                    fade_out = torch.linspace(1.0, 0.0, crossfade_samples, device=trimmed_wave.device)
                    fade_in = torch.linspace(0.0, 1.0, crossfade_samples, device=trimmed_wave.device)
                    
                    # 1. Everything before the fade
                    left_keep = trimmed_wave[:, :A - crossfade_samples]
                    
                    # 2. The overlapped mix zone
                    left_fade = trimmed_wave[:, A - crossfade_samples : A] * fade_out
                    right_fade = trimmed_wave[:, B : B + crossfade_samples] * fade_in
                    mixed_zone = left_fade + right_fade
                    
                    # 3. Everything after the fade
                    right_keep = trimmed_wave[:, B + crossfade_samples:]
                    
                    # Reassemble the audio
                    squeezed_wave = torch.cat([left_keep, mixed_zone, right_keep], dim=1)
                    
                    # Handle minor rounding errors (off by 1 sample) due to integer division
                    if squeezed_wave.shape[1] < target_samples:
                        pad = target_samples - squeezed_wave.shape[1]
                        squeezed_wave = torch.nn.functional.pad(squeezed_wave, (0, pad), "constant", 0.0)
                    elif squeezed_wave.shape[1] > target_samples:
                        squeezed_wave = squeezed_wave[:, :target_samples]
                        
                    torchaudio.save(out_filepath, squeezed_wave, sr, encoding="PCM_S", bits_per_sample=16)
                    stats["squeezed"] += 1
                else:
                    # The pause is too short (or doesn't exist). The words themselves exceed 1.4s.
                    shutil.copy2(filepath, unprocessed_filepath)
                    stats["unsqueezable"] += 1

        except Exception as e:
            tqdm.write(f"  [ERROR] Processing {filename}: {e}")
            shutil.copy2(filepath, unprocessed_filepath)
            stats["silent_failed"] += 1

    print("\n" + "="*60)
    print("DATASET SQUEEZING COMPLETE")
    print("="*60)
    print(f"Centered Normally (<= 1.4s): {stats['centered_normally']}")
    print(f"Successfully Squeezed (Crossfaded): {stats['squeezed']}")
    print(f"Copied to Unprocessed (Words literally > 1.4s): {stats['unsqueezable']}")
    print(f"Copied to Unprocessed (Silence/Error): {stats['silent_failed']}")
    print(f"Skipped (Already Processed): {stats['skipped']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Squeeze inter-word silence with crossfading to fit audio into exactly 1.4s.")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing audio to process")
    parser.add_argument("--dest_dir", type=str, required=True, help="Directory to output perfect 1.4s files")
    parser.add_argument("--unprocessed_dir", type=str, default="./dataset/unprocessed", help="Directory for files that are hopelessly too long")
    args = parser.parse_args()
    
    squeeze_and_center_dataset(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        unprocessed_dir=args.unprocessed_dir,
        target_duration=1.4,
        target_sr=16000
    )