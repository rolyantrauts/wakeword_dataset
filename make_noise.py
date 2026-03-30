import os
import random
import torch
import torchaudio
import argparse
import warnings
import shortuuid
import wave
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

# =============================================================================
# 1. AUDIO PROCESSING UTILITIES
# =============================================================================
def normalize_peak(waveform, target_peak):
    """Normalizes the audio to a specific peak amplitude."""
    max_amp = torch.max(torch.abs(waveform))
    if max_amp > 0:
        return (waveform / max_amp) * target_peak
    return waveform

def get_audio_metadata(filepath):
    """Robustly gets audio length and sample rate across different torchaudio versions."""
    if hasattr(torchaudio, 'info'):
        try:
            metadata = torchaudio.info(filepath)
            return metadata.num_frames, metadata.sample_rate
        except Exception:
            pass
            
    if filepath.lower().endswith('.wav'):
        try:
            with wave.open(filepath, 'rb') as w:
                return w.getnframes(), w.getframerate()
        except Exception:
            pass
            
    waveform, sr = torchaudio.load(filepath)
    return waveform.shape[1], sr

# =============================================================================
# 2. NOISE INDEXING & WEIGHTING
# =============================================================================
def index_noise_files(noise_dir, min_duration_sec):
    """
    Scans the noise directory, retrieves the length and sample rate of each file, 
    and calculates selection weights based on audio length.
    """
    noise_files = []
    supported_exts = ('.wav', '.flac', '.mp3')
    
    print(f"Indexing noise files in {noise_dir}...")
    
    for filename in os.listdir(noise_dir):
        if not filename.lower().endswith(supported_exts):
            continue
            
        filepath = os.path.join(noise_dir, filename)
        
        try:
            total_samples, sample_rate = get_audio_metadata(filepath)
            duration_sec = total_samples / sample_rate
            
            # Only keep files that are long enough for the requested extraction length
            if duration_sec >= min_duration_sec:
                noise_files.append({
                    'path': filepath,
                    'total_samples': total_samples,
                    'sample_rate': sample_rate,
                    'duration': duration_sec
                })
        except Exception as e:
            print(f"  [Warning] Could not read metadata for {filename}: {e}")

    if not noise_files:
        raise ValueError(f"No valid noise files (>= {min_duration_sec}s) found in {noise_dir}.")

    weights = [f['total_samples'] for f in noise_files]
    
    print(f"Successfully indexed {len(noise_files)} noise files capable of providing {min_duration_sec}s chunks.")
    return noise_files, weights

# =============================================================================
# 3. DYNAMIC NOISE EXTRACTION
# =============================================================================
def get_random_noise_chunk(noise_list, weights, target_samples, target_sr):
    """
    Selects a weighted random noise file, calculates a random window, 
    and reads only that specific chunk from the disk.
    """
    selected_noise = random.choices(noise_list, weights=weights, k=1)[0]
    native_sr = selected_noise['sample_rate']
    
    frames_needed = int((target_samples / target_sr) * native_sr)
    max_start_frame = selected_noise['total_samples'] - frames_needed
    start_frame = random.randint(0, max_start_frame)
    
    noise_chunk, sr = torchaudio.load(
        selected_noise['path'], 
        frame_offset=start_frame, 
        num_frames=frames_needed
    )
    
    # --- CONVERT TO MONO ---
    if noise_chunk.shape[0] > 1:
        noise_chunk = noise_chunk.mean(dim=0, keepdim=True)
    
    if sr != target_sr:
        noise_chunk = torchaudio.functional.resample(noise_chunk, sr, target_sr)
        
    if noise_chunk.shape[1] > target_samples:
        noise_chunk = noise_chunk[:, :target_samples]
    elif noise_chunk.shape[1] < target_samples:
        pad_amount = target_samples - noise_chunk.shape[1]
        noise_chunk = torch.nn.functional.pad(noise_chunk, (0, pad_amount), "constant", 0.0)
        
    return noise_chunk

# =============================================================================
# 4. GENERATION PIPELINE
# =============================================================================
def generate_noise_dataset(noise_dir, dest_dir, qty, length_sec, target_sr=16000):
    os.makedirs(dest_dir, exist_ok=True)
    
    # Index the noise files and get their weights, filtering out anything too short
    noise_list, noise_weights = index_noise_files(noise_dir, length_sec)
    
    target_samples = int(length_sec * target_sr)
    
    print(f"\nGenerating {qty} noise samples of {length_sec} seconds each...")
    
    stats = {"loud_065_095": 0, "quiet_00_03": 0}
    
    for _ in tqdm(range(qty), desc="Creating Samples"):
        try:
            # 1. Extract the raw noise chunk
            noise_wave = get_random_noise_chunk(noise_list, noise_weights, target_samples, target_sr)
            
            # 2. Apply DSP Wash (DC Offset)
            noise_wave = noise_wave - noise_wave.mean(dim=1, keepdim=True)
            
            # 3. Apply DSP Wash (50Hz High-Pass Biquad Filter)
            noise_wave = torchaudio.functional.highpass_biquad(noise_wave, target_sr, cutoff_freq=50.0)
            
            # 4. Determine the volume profile (70% loud range, 30% quiet range)
            is_loud = random.random() < 0.70
            
            if is_loud:
                target_peak = random.uniform(0.65, 0.95)
                stats["loud_065_095"] += 1
            else:
                target_peak = random.uniform(0.0, 0.3)
                stats["quiet_00_03"] += 1
                
            # 5. Apply the chosen volume peak
            final_wave = normalize_peak(noise_wave, target_peak)
            
            # 6. Safety clamp for 16-bit PCM export
            final_wave = torch.clamp(final_wave, -0.99, 0.99)
            
            # 7. Save with a unique ID
            uid = shortuuid.uuid()
            dest_path = os.path.join(dest_dir, f"noise_{uid}.wav")
            torchaudio.save(dest_path, final_wave, target_sr, encoding="PCM_S", bits_per_sample=16)
            
        except Exception as e:
            tqdm.write(f"  [ERROR] Failed to generate a sample: {e}")

    print("\n" + "="*60)
    print("NOISE GENERATION COMPLETE")
    print("="*60)
    print(f"Total Samples Created: {qty}")
    print(f"Files peaking 0.65-0.95 (70% target): {stats['loud_065_095']}")
    print(f"Files peaking 0.0-0.3 (30% target): {stats['quiet_00_03']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a purely noise-based dataset from longer source files.")
    parser.add_argument("--noise_dir", type=str, required=True, help="Directory containing original background noise files")
    parser.add_argument("--dest_dir", type=str, required=True, help="Directory to save the generated noise clips")
    parser.add_argument("--qty", type=int, required=True, help="Number of noise samples to generate")
    parser.add_argument("--length", type=float, required=True, help="Length of each output file in seconds (e.g., 1.4)")
    
    args = parser.parse_args()
    
    generate_noise_dataset(
        noise_dir=args.noise_dir,
        dest_dir=args.dest_dir,
        qty=args.qty,
        length_sec=args.length,
        target_sr=16000
    )