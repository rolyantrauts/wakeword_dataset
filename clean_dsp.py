import os
import torch
import torchaudio
import argparse
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

# =============================================================================
# 1. DSP PIPELINE
# =============================================================================
def apply_dsp_wash(waveform, sr):
    """Washes the audio of DC offset, sub-bass rumble, and normalizes volume."""
    
    # 1. DC Offset Correction (Center the waveform on the zero-line)
    waveform = waveform - waveform.mean(dim=1, keepdim=True)
    
    # 2. 50Hz High-Pass Biquad Filter (Remove non-speech sub-rumble)
    waveform = torchaudio.functional.highpass_biquad(waveform, sr, cutoff_freq=50.0)
    
    # 3. Peak Normalization (to 95% ceiling)
    max_amp = torch.max(torch.abs(waveform))
    if max_amp > 0:
        waveform = (waveform / max_amp) * 0.95
        
    return waveform

# =============================================================================
# 2. DIRECTORY PROCESSING
# =============================================================================
def process_directory(source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    
    # Target specifications for neural network training
    TARGET_SR = 16000
    
    # Support common audio formats
    supported_exts = ('.wav', '.flac', '.mp3')
    audio_files = [f for f in os.listdir(source_dir) if f.lower().endswith(supported_exts)]
    
    print(f"\nFound {len(audio_files)} files to process in {source_dir}.")
    
    stats = {"processed": 0, "skipped": 0, "errors": 0}

    for filename in tqdm(audio_files, desc="Applying DSP Wash & Converting"):
        filepath = os.path.join(source_dir, filename)
        
        # Always output as a clean .wav file
        out_filename = os.path.splitext(filename)[0] + ".wav"
        out_filepath = os.path.join(dest_dir, out_filename)
        
        # Skip logic so you can resume interrupted batches
        if os.path.exists(out_filepath):
            stats["skipped"] += 1
            continue
            
        try:
            # Load the raw audio
            waveform, sr = torchaudio.load(filepath)
            
            # --- CONVERT TO MONO ---
            # If the file has more than 1 channel, average them down to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
                
            # --- RESAMPLE TO 16kHz ---
            if sr != TARGET_SR:
                waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
                sr = TARGET_SR
            
            # Apply the DSP pipeline (now operating on guaranteed 16kHz Mono)
            clean_wave = apply_dsp_wash(waveform, sr)
            
            # Safety clamp to ensure we don't exceed 16-bit PCM integer limits (-1.0 to 1.0)
            clean_wave = torch.clamp(clean_wave, -1.0, 1.0)
            
            # Save the washed file as 16-bit PCM
            torchaudio.save(out_filepath, clean_wave, sr, encoding="PCM_S", bits_per_sample=16)
            stats["processed"] += 1
            
        except Exception as e:
            tqdm.write(f"  [ERROR] Processing {filename}: {e}")
            stats["errors"] += 1

    print("\n" + "="*60)
    print("DSP WASH COMPLETE")
    print("="*60)
    print(f"Successfully Washed & Converted: {stats['processed']}")
    print(f"Skipped (Already Exists): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply 16kHz Mono Conversion, DC Offset, 50Hz High-Pass, and 95% Normalization.")
    parser.add_argument("--source_dir", type=str, required=True, help="Directory containing raw audio files")
    parser.add_argument("--dest_dir", type=str, required=True, help="Directory to output the washed audio files")
    args = parser.parse_args()
    
    process_directory(
        source_dir=args.source_dir, 
        dest_dir=args.dest_dir
    )