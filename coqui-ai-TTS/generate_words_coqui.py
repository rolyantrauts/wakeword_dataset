import os
import re
import csv
import torch
import torchaudio
import warnings
import difflib
import itertools
from tqdm import tqdm
import sys
import importlib.machinery
import importlib.abc
import importlib.metadata

# Suppress the urllib3 LibreSSL warning on macOS
import urllib3
urllib3.disable_warnings(urllib3.exceptions.NotOpenSSLWarning)

# =============================================================================
# --- MONKEY PATCHES FOR TTS COMPATIBILITY ---
# 1. Inject missing transformers utility functions that HuggingFace removed.
# 2. Inject missing importlib.metadata function for Python 3.9.
# 3. Intercept TTS imports to inject __future__ annotations for Python 3.9 
#    to prevent crashes on Python 3.10+ type union syntax (e.g., str | None).
# =============================================================================
import transformers.pytorch_utils
if not hasattr(transformers.pytorch_utils, 'isin_mps_friendly'):
    transformers.pytorch_utils.isin_mps_friendly = torch.isin

import transformers.utils.import_utils
if not hasattr(transformers.utils.import_utils, 'is_torch_greater_or_equal'):
    transformers.utils.import_utils.is_torch_greater_or_equal = lambda _version: True

if not hasattr(transformers.utils.import_utils, 'is_torchcodec_available'):
    transformers.utils.import_utils.is_torchcodec_available = lambda: True

if not hasattr(importlib.metadata, 'packages_distributions'):
    importlib.metadata.packages_distributions = lambda: {"coqpit": ["coqpit"]}

class PEP604Loader(importlib.machinery.SourceFileLoader):
    def get_data(self, path):
        data = super().get_data(path)
        if path.endswith('.py'):
            try:
                source = data.decode('utf-8')
                if "from __future__ import annotations" not in source:
                    source = "from __future__ import annotations\n" + source
                return source.encode('utf-8')
            except Exception:
                pass
        return data

class PEP604Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in ("TTS", "coqpit"):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if spec and spec.origin and spec.origin.endswith('.py'):
                spec.loader = PEP604Loader(fullname, spec.origin)
                return spec
        return None

if sys.version_info < (3, 10):
    sys.meta_path.insert(0, PEP604Finder())
# =============================================================================

from TTS.api import TTS
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

# =============================================================================
# 1. CSV MANAGEMENT
# =============================================================================
def load_phrases_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CRITICAL: {csv_path} not found.")
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    
    phrases = []
    for row in reader:
        phrase = row.get('phrase', row.get('text', '')).strip()
        if phrase:
            phrases.append(phrase)
    return phrases, reader

def update_csv_remaining(csv_path, original_reader, successful_phrases):
    """Rewrites the CSV, dropping the phrases that were successfully generated."""
    keep_list = []
    success_set = set(successful_phrases)
    for row in original_reader:
        phrase = row.get('phrase', row.get('text', '')).strip()
        if phrase not in success_set:
            keep_list.append(row)
            
    if keep_list:
        keys = original_reader[0].keys()
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(keep_list)
    else:
        open(csv_path, 'w').close()

# =============================================================================
# 2. CTC ALIGNMENT & PHONETIC SCORING
# =============================================================================
def evaluate_tensor_ctc_words(waveform, sample_rate, asr_model, asr_bundle, device, labels, blank_idx):
    total_duration = waveform.shape[1] / sample_rate
    
    if sample_rate != asr_bundle.sample_rate:
        waveform_16k = torchaudio.functional.resample(waveform, sample_rate, asr_bundle.sample_rate)
    else:
        waveform_16k = waveform
        
    waveform_16k = waveform_16k.to(device)
    
    with torch.inference_mode():
        emissions, _ = asr_model(waveform_16k)
        probs = torch.nn.functional.softmax(emissions, dim=-1)
        _, top_indices = torch.max(probs, dim=-1)
        
        decoded_chars = []
        prev_idx = -1
        for i in range(top_indices.shape[1]):
            idx = top_indices[0, i].item()
            if idx != prev_idx:
                if idx != blank_idx: 
                    char = labels[idx]
                    if char == '|': char = ' '
                    decoded_chars.append((char, i))
            prev_idx = idx

    words = []
    current_word = ""
    start_f = -1
    prev_f = -1
    
    for char, frame in decoded_chars:
        if char == ' ':
            if current_word:
                words.append({'word': current_word, 'start': start_f, 'end': prev_f})
                current_word = ""
                start_f = -1
        else:
            if current_word == "": start_f = frame
            current_word += char
            prev_f = frame
            
    if current_word:
        words.append({'word': current_word, 'start': start_f, 'end': prev_f})
        
    full_transcript = " ".join([w['word'] for w in words])
    return full_transcript, words, total_duration

def phonetic_normalize(text):
    """Reduces an English string to a strict phonetic signature."""
    text = text.upper()
    text = re.sub(r'[^A-Z]', '', text) 
    
    # Consonant grouping
    text = re.sub(r'[Z]', 'S', text)
    text = re.sub(r'[CQ]', 'K', text)
    text = re.sub(r'[VF]', 'B', text) 
    text = re.sub(r'[G]', 'J', text)
    
    # Digraph mapping
    text = text.replace('PH', 'F')
    text = text.replace('SH', 'S')
    text = text.replace('CH', 'J')
    text = text.replace('TH', 'T')
    
    # Collapse all vowels to 'A'
    text = re.sub(r'[AEIOUYW]+', 'A', text) 
    
    # Remove consecutive duplicate characters
    text = re.sub(r'(.)\1+', r'\1', text)
    
    return text

def verify_match_strict(transcript, target_text, threshold):
    """Scores based on phonetic signature and strict length gates."""
    target_clean = re.sub(r'[^A-Z0-9]', '', target_text.upper())
    transcript_clean = re.sub(r'[^A-Z0-9]', '', transcript.upper())
    
    raw_score = difflib.SequenceMatcher(None, target_clean, transcript_clean).ratio()

    min_length = int(len(target_clean) * 0.70)
    if len(transcript_clean) < min_length:
        return False, raw_score, f"TOO SHORT ({len(transcript_clean)} < {min_length})"
        
    if len(transcript_clean) > len(target_clean) + 12:
        return False, raw_score, "HALLUCINATION (TOO LONG)"

    target_phonetic = phonetic_normalize(target_text)
    trans_phonetic = phonetic_normalize(transcript)
    
    phonetic_score = difflib.SequenceMatcher(None, target_phonetic, trans_phonetic).ratio()
    
    if phonetic_score < threshold:
        return False, phonetic_score, f"LOW PHONETIC MATCH ({trans_phonetic} vs {target_phonetic})"
        
    return True, phonetic_score, "PASS"

def format_tts_text(raw_text):
    """Strips punctuation, preserves spaces, and applies Sentence case."""
    no_punct = re.sub(r'[^\w\s]', '', raw_text)
    clean_text = re.sub(r'\s+', ' ', no_punct).strip()
    return clean_text.capitalize()

# =============================================================================
# 3. MASTER XTTS GENERATOR PIPELINE
# =============================================================================
def generate_xtts_dataset(
    wordlist, dest_dir, speaker_refs_dir, latents_dir,
    match_threshold=0.65, max_retries=10, max_duration=1.8, max_num=0
):
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(latents_dir, exist_ok=True)
    
    is_csv_mode = wordlist.lower().endswith('.csv')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nLoading models on hardware backend: {device.type.upper()}...")
    
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    xtts_model = tts.synthesizer.tts_model 
    
    asr_bundle = WAV2VEC2_ASR_BASE_960H
    asr_model = asr_bundle.get_model().to(device)
    asr_labels = asr_bundle.get_labels()
    asr_blank_idx = 0 
    sr = 24000
    
    supported_languages = ["ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it", "ja", "ko", "nl", "pl", "pt", "ru", "tr", "zh-cn"]
    speaker_ids = sorted([d for d in os.listdir(speaker_refs_dir) if os.path.isdir(os.path.join(speaker_refs_dir, d))])
    
    print("\nInitializing Latent Embedding Cache...")
    latents_cache = {}
    for speaker_id in tqdm(speaker_ids, desc="Caching Latents"):
        cache_path = os.path.join(latents_dir, f"{speaker_id}.pth")
        
        if os.path.exists(cache_path):
            latents_cache[speaker_id] = torch.load(cache_path, map_location=device)
        else:
            speaker_dir = os.path.join(speaker_refs_dir, speaker_id)
            speaker_wavs = [os.path.join(speaker_dir, f) for f in os.listdir(speaker_dir) if f.endswith('.wav')]
            if not speaker_wavs: continue
            
            gpt_cond, spk_emb = xtts_model.get_conditioning_latents(audio_path=speaker_wavs)
            torch.save((gpt_cond, spk_emb), cache_path)
            latents_cache[speaker_id] = (gpt_cond, spk_emb)
            
    speakers_list = sorted(list(latents_cache.keys()))
    print(f"Loaded {len(speakers_list)} speakers.")

    matrix_combinations = list(itertools.product(speakers_list, supported_languages))
    print(f"Matrix configured: {len(speakers_list)} speakers × {len(supported_languages)} languages = {len(matrix_combinations)} combinations per phrase.\n")

    # =========================================================================
    # MODE A: CSV WORDLIST
    # =========================================================================
    if is_csv_mode:
        phrases, original_reader = load_phrases_from_csv(wordlist)
        print(f"CSV Mode Detected. Loaded {len(phrases)} phrases.")
        
        matrix_cycle = itertools.cycle(matrix_combinations)
        successful_phrases = []
        
        for phrase in tqdm(phrases, desc="Processing CSV Wordlist"):
            if max_num > 0 and len(successful_phrases) >= max_num:
                print(f"\nReached requested quota of {max_num} generated files. Stopping early.")
                break
                
            speaker_id, target_lang = next(matrix_cycle)
            gpt_cond, spk_emb = latents_cache[speaker_id]
            
            safe_phrase = re.sub(r'[^a-zA-Z0-9_]', '', phrase.replace(' ', '_').lower())
            output_filepath = os.path.join(dest_dir, f"{safe_phrase}_{speaker_id}_{target_lang}.wav")
            
            if os.path.exists(output_filepath):
                successful_phrases.append(phrase)
                continue
            
            formatted_text = format_tts_text(phrase)
            
            passed_ctc = False
            overall_best_score = -1.0
            overall_best_trans = ""
            overall_best_reason = ""
            
            for attempt in range(max_retries):
                try:
                    current_temp = 0.50
                    current_speed = 1.15
                    
                    out = xtts_model.inference(
                        text=formatted_text, language=target_lang, gpt_cond_latent=gpt_cond, 
                        speaker_embedding=spk_emb, temperature=current_temp, 
                        repetition_penalty=10.0, top_k=50, top_p=0.85, speed=current_speed                 
                    )
                    
                    wav_tensor = torch.tensor(out["wav"]).unsqueeze(0)
                    full_transcript, words, total_dur = evaluate_tensor_ctc_words(wav_tensor, sr, asr_model, asr_bundle, device, asr_labels, asr_blank_idx)
                    
                    if not words: continue
                    
                    is_valid, sim, reason = verify_match_strict(full_transcript, formatted_text, match_threshold)
                    speech_dur = (words[-1]['end'] + 1) * 0.02 - (words[0]['start'] * 0.02)
                    
                    # Explictly fail it if it passes phonetics but is too long
                    if is_valid and speech_dur > max_duration:
                        is_valid = False
                        reason = f"DURATION TOO LONG ({speech_dur:.2f}s > {max_duration}s)"
                    
                    if sim > overall_best_score:
                        overall_best_score = sim
                        overall_best_trans = full_transcript
                        overall_best_reason = reason
                    
                    if is_valid:
                        torchaudio.save(output_filepath, wav_tensor, sr)
                        passed_ctc = True
                        successful_phrases.append(phrase)
                        tqdm.write(f"  [PASS] {speaker_id}[{target_lang}] Target: '{formatted_text}' -> Heard: '{full_transcript}' | Score: {sim:.2f}")
                        break
                        
                except Exception as e:
                    tqdm.write(f"  [ERROR] Attempting {formatted_text}: {e}")
            
            if not passed_ctc:
                tqdm.write(f"  [FAILED] '{formatted_text}' exhausted retries. Heard: '{overall_best_trans}' | Score: {overall_best_score:.2f} | Reason: {overall_best_reason}")
                
        update_csv_remaining(wordlist, original_reader, successful_phrases)
        print(f"\n[COMPLETE] Processed CSV. Removed {len(successful_phrases)} successful generations from {wordlist}.")

    # =========================================================================
    # MODE B: SINGLE PHRASE (Single Pass Matrix)
    # =========================================================================
    else:
        print(f"Single Phrase Mode Detected: '{wordlist}'")
        formatted_text = format_tts_text(wordlist)
        safe_phrase = re.sub(r'[^a-zA-Z0-9_]', '', wordlist.replace(' ', '_').lower())
        
        successful_count = 0
        
        for speaker_id, target_lang in tqdm(matrix_combinations, desc="Matrix Generation"):
            if max_num > 0 and successful_count >= max_num:
                print(f"\nReached requested quota of {max_num} generated files. Stopping early.")
                break
                
            output_filepath = os.path.join(dest_dir, f"{safe_phrase}_{speaker_id}_{target_lang}.wav")
            
            if os.path.exists(output_filepath):
                successful_count += 1
                continue
            
            passed_ctc = False
            overall_best_score = -1.0
            overall_best_trans = ""
            overall_best_reason = ""
            
            for attempt in range(max_retries):
                try:
                    current_temp = 0.50
                    current_speed = 1.15
                    
                    out = xtts_model.inference(
                        text=formatted_text, language=target_lang, gpt_cond_latent=gpt_cond, 
                        speaker_embedding=spk_emb, temperature=current_temp, 
                        repetition_penalty=10.0, top_k=50, top_p=0.85, speed=current_speed                 
                    )
                    
                    wav_tensor = torch.tensor(out["wav"]).unsqueeze(0)
                    full_transcript, words, total_dur = evaluate_tensor_ctc_words(wav_tensor, sr, asr_model, asr_bundle, device, asr_labels, asr_blank_idx)
                    
                    if not words: continue
                    
                    is_valid, sim, reason = verify_match_strict(full_transcript, formatted_text, match_threshold)
                    speech_dur = (words[-1]['end'] + 1) * 0.02 - (words[0]['start'] * 0.02)
                    
                    # Explictly fail it if it passes phonetics but is too long
                    if is_valid and speech_dur > max_duration:
                        is_valid = False
                        reason = f"DURATION TOO LONG ({speech_dur:.2f}s > {max_duration}s)"
                    
                    if sim > overall_best_score:
                        overall_best_score = sim
                        overall_best_trans = full_transcript
                        overall_best_reason = reason
                    
                    if is_valid:
                        torchaudio.save(output_filepath, wav_tensor, sr)
                        passed_ctc = True
                        successful_count += 1
                        tqdm.write(f"  [PASS] {speaker_id}[{target_lang}] Target: '{formatted_text}' -> Heard: '{full_transcript}' | Score: {sim:.2f}")
                        break
                        
                except Exception as e:
                    tqdm.write(f"  [ERROR] Attempting {formatted_text}: {e}")
            
            if not passed_ctc:
                tqdm.write(f"  [FAILED] '{formatted_text}' in {speaker_id}[{target_lang}] exhausted retries. Heard: '{overall_best_trans}' | Score: {overall_best_score:.2f} | Reason: {overall_best_reason}")
                
        print(f"\n[COMPLETE] Finished Matrix Generation for '{wordlist}'. Total successful files: {successful_count}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified XTTS Generation Script")
    parser.add_argument("--wordlist", type=str, required=True, help="Either a .csv file path OR a text phrase like 'Hey Jarvis'")
    parser.add_argument("--dest_dir", type=str, required=True, help="Output directory for generated WAV files")
    parser.add_argument("--speaker_dir", type=str, default="./voice-splits/", help="Directory containing speaker reference audio")
    parser.add_argument("--latents_dir", type=str, default="./xtts_latents/", help="Directory to cache latent embeddings")
    parser.add_argument("--threshold", type=float, default=0.65, help="Phonetic match threshold")
    parser.add_argument("--max_retries", type=int, default=10, help="Max retry attempts per generation")
    parser.add_argument("--max_num", type=int, default=0, help="Maximum number of successful generations to produce (0 = all)")
    args = parser.parse_args()
    
    generate_xtts_dataset(
        wordlist=args.wordlist,
        dest_dir=args.dest_dir,
        speaker_refs_dir=args.speaker_dir,
        latents_dir=args.latents_dir,
        match_threshold=args.threshold,
        max_retries=args.max_retries,
        max_num=args.max_num
    )