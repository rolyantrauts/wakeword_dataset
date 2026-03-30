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
def normalize_peak(waveform, target_peak=0.85):
    """Normalizes the audio to a specific peak amplitude."""
    max_amp = torch.max(torch.abs(waveform))
    if max_amp > 0:
        return (waveform / max_amp) * target_peak
    return waveform

def get_audio_metadata(filepath):
    """Robustly gets audio length and sample rate across different torchaudio versions."""
    # Method 1: Modern torchaudio (Fastest if available)
    if hasattr(torchaudio, 'info'):
        try:
            metadata = torchaudio.info(filepath)
            return metadata.num_frames, metadata.sample_rate
        except Exception:
            pass
            
    # Method 2: Python built-in wave module (Instant for .wav on older systems)
    if filepath.lower().endswith('.wav'):
        try:
            with wave.open(filepath, 'rb') as w:
                return w.getnframes(), w.getframerate()
        except Exception:
            pass
            
    # Method 3: Brute force load (Slower, but guaranteed to work for flac/mp3 on old versions)
    waveform, sr = torchaudio.load(filepath)
    return waveform.shape[1], sr

# =============================================================================
# 2. NOISE INDEXING & WEIGHTING
# =============================================================================
def index_noise_files(noise_dir):
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
            
            # Only keep files that are at least 1.4 seconds long
            if duration_sec >= 1.4:
                noise_files.append({
                    'path': filepath,
                    'total_samples': total_samples,
                    'sample_rate': sample_rate,
                    'duration': duration_sec
                })
        except Exception as e:
            print(f"  [Warning] Could not read metadata for {filename}: {e}")

    if not noise_files:
        raise ValueError(f"No valid noise files (>= 1.4s) found in {noise_dir}.")

    # Use the total number of samples as the weight for `random.choices`
    weights = [f['total_samples'] for f in noise_files]
    
    print(f"Successfully indexed {len(noise_files)} noise files.")
    return noise_files, weights

# =============================================================================
# 3. DYNAMIC NOISE EXTRACTION
# =============================================================================
def get_random_noise_chunk(noise_list, weights, target_samples, target_sr):
    """
    Selects a weighted random noise file, calculates a random 1.4s window, 
    and reads only that specific chunk from the disk.
    """
    # 1. Pick a file based on length weight
    selected_noise = random.choices(noise_list, weights=weights, k=1)[0]
    
    native_sr = selected_noise['sample_rate']
    
    # Calculate how many frames 1.4s is in the native sample rate
    frames_needed = int((target_samples / target_sr) * native_sr)
    
    # 2. Pick a random start frame
    max_start_frame = selected_noise['total_samples'] - frames_needed
    start_frame = random.randint(0, max_start_frame)
    
    # 3. Load just that tiny chunk directly from disk
    noise_chunk, sr = torchaudio.load(
        selected_noise['path'], 
        frame_offset=start_frame, 
        num_frames=frames_needed
    )
    
    # --- CONVERT TO MONO ---
    # If the noise file is stereo, average it down to mono
    if noise_chunk.shape[0] > 1:
        noise_chunk = noise_chunk.mean(dim=0, keepdim=True)
    
    # 4. Resample to match our clean audio if necessary
    if sr != target_sr:
        noise_chunk = torchaudio.functional.resample(noise_chunk, sr, target_sr)
        
    # Ensure it exactly matches the target sample length (handle minor rounding errors)
    if noise_chunk.shape[1] > target_samples:
        noise_chunk = noise_chunk[:, :target_samples]
    elif noise_chunk.shape[1] < target_samples:
        pad_amount = target_samples - noise_chunk.shape[1]
        noise_chunk = torch.nn.functional.pad(noise_chunk, (0, pad_amount), "constant", 0.0)
        
    return noise_chunk

# =============================================================================
# 4. MASTER MIXING PIPELINE
# =============================================================================
def process_augmentation(source_dir, dest_dir, noise_dir):
    os.makedirs(dest_dir, exist_ok=True)
    
    # Index the noise files and get their weights
    noise_list, noise_weights = index_noise_files(noise_dir)
    
    source_files = [f for f in os.listdir(source_dir) if f.endswith('.wav')]
    print(f"\nFound {len(source_files)} clean audio files to process.")
    
    for filename in tqdm(source_files, desc="Adding Noise"):
        source_path = os.path.join(source_dir, filename)
        
        # Generate the new unique filename using shortuuid
        uid = shortuuid.uuid()
        new_filename = f"{uid}_{filename}"
        dest_path = os.path.join(dest_dir, new_filename)
        
        try:
            # Load the clean 1.4s audio
            clean_wave, clean_sr = torchaudio.load(source_path)
            total_clean_samples = clean_wave.shape[1]
            
            # 1. Randomize clean speech volume between 0.65 and 0.95
            random_speech_vol = random.uniform(0.65, 0.95)
            clean_wave = normalize_peak(clean_wave, target_peak=random_speech_vol)
            
            # 2. Get a random 1.4s chunk of noise
            noise_wave = get_random_noise_chunk(noise_list, noise_weights, total_clean_samples, clean_sr)
            
            # 3. Apply DSP Wash to the Noise (DC Offset & 50Hz High-Pass)
            noise_wave = noise_wave - noise_wave.mean(dim=1, keepdim=True)
            noise_wave = torchaudio.functional.highpass_biquad(noise_wave, clean_sr, cutoff_freq=50.0)
            
            # 4. Apply random volume between 0.0 and 0.3 to the washed noise
            random_noise_vol = random.uniform(0.0, 0.3)
            noise_wave = normalize_peak(noise_wave, target_peak=random_noise_vol)
            
            # 5. Mix them together
            mixed_wave = clean_wave + noise_wave
            
            # 6. Safety Limiter
            mixed_wave = torch.clamp(mixed_wave, -0.99, 0.99)
            
            # Save the augmented file with its new unique ID
            torchaudio.save(dest_path, mixed_wave, clean_sr, encoding="PCM_S", bits_per_sample=16)
            
        except Exception as e:
            tqdm.write(f"  [ERROR] Processing {filename}: {e}")

    print("\n" + "="*60)
    print("NOISE AUGMENTATION COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add weighted random background noise to audio dataset.")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing clean 1.4s centered audio files")
    parser.add_argument("--dest_dir", type=str, required=True, help="Directory to output the noisy mixed files")
    parser.add_argument("--noise_dir", type=str, required=True, help="Directory containing background noise files of varying lengths")
    args = parser.parse_args()
    
    process_augmentation(
        source_dir=args.source_dir,
        dest_dir=args.dest_dir,
        noise_dir=args.noise_dir
    )