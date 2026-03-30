import os
import re
import csv
import json
import torch
import torchaudio
import warnings
import difflib
import itertools
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

# =============================================================================
# 1. PIPER DICTIONARY & ARPABET TRANSLATOR
# =============================================================================
ARPABET_TO_IPA = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ə', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ',
    'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', 'EH': 'ɛ', 'ER': 'ɚ',
    'EY': 'eɪ', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i',
    'JH': 'dʒ', 'K': 'k', 'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ',
    'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w',
    'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}

def load_piper_tokens(tokens_path):
    if not os.path.exists(tokens_path):
        raise FileNotFoundError(f"CRITICAL: Tokens file {tokens_path} not found.")
    token2id = {}
    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip('\n')
            if not line: continue
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                token2id[parts[0]] = int(parts[1])
    return token2id

def arpabet_to_piper_tensor(raw_phons, token2id):
    pad_id = token2id.get('_', 0)
    bos_id = token2id.get('^', 1)
    eos_id = token2id.get('$', 2)
    space_id = token2id.get(' ', None)
    
    phoneme_ids = [bos_id, pad_id]
    
    for p in raw_phons:
        if p == '|':
            if space_id is not None:
                phoneme_ids.append(space_id)
                phoneme_ids.append(pad_id)
        else:
            base_phoneme = ''.join([c for c in p if not c.isdigit()])
            if base_phoneme in ARPABET_TO_IPA:
                ipa_str = ARPABET_TO_IPA[base_phoneme]
                for char in ipa_str:
                    if char not in token2id:
                        if char == 'ɹ' and 'r' in token2id: char = 'r'
                        elif char == 'ɑ' and 'ɑː' in token2id: char = 'ɑː'
                        elif char == 'ɡ' and 'g' in token2id: char = 'g'
                    
                    if char in token2id:
                        phoneme_ids.append(token2id[char])
                        phoneme_ids.append(pad_id)
                        
    phoneme_ids.append(eos_id)
    return np.array([phoneme_ids], dtype=np.int64), np.array([len(phoneme_ids)], dtype=np.int64)

def load_piper_gender_map(config_path, speakers_txt="SPEAKERS.TXT", male_csv="Voices-Male.csv", female_csv="Voices-Female.csv"):
    """Maps raw LibriTTS IDs from local files to Piper's internal 0-903 indices."""
    raw_gender_map = {}
    
    # 1. Primary: Load from the local SPEAKERS.TXT file if available
    if os.path.exists(speakers_txt):
        with open(speakers_txt, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith(';') or not line.strip(): 
                    continue
                parts = line.split('|')
                if len(parts) >= 2:
                    raw_id = parts[0].strip()
                    gender = parts[1].strip().lower()
                    if gender in ['m', 'f']:
                        raw_gender_map[raw_id] = gender
        print(f"Loaded {len(raw_gender_map)} gender tags from {speakers_txt}.")
    
    # 2. Fallback: Load any remaining IDs from the CSV files
    if os.path.exists(male_csv):
        with open(male_csv, 'r', encoding='utf-8') as f:
            for row in csv.reader(f):
                if row and len(row) > 0:
                    raw_id = row[0].strip()
                    if raw_id not in raw_gender_map:
                        raw_gender_map[raw_id] = 'm'
                        
    if os.path.exists(female_csv):
        with open(female_csv, 'r', encoding='utf-8') as f:
            for row in csv.reader(f):
                if row and len(row) > 0:
                    raw_id = row[0].strip()
                    if raw_id not in raw_gender_map:
                        raw_gender_map[raw_id] = 'f'
                        
    # 3. Map the raw IDs to Piper's internal integer indices using the .json config
    piper_gender_map = {}
    if os.path.exists(config_path) and raw_gender_map:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                speaker_id_map = config_data.get("speaker_id_map", {})
                for real_id_str, piper_idx in speaker_id_map.items():
                    if real_id_str in raw_gender_map:
                        piper_gender_map[piper_idx] = raw_gender_map[real_id_str]
            print(f"Successfully mapped genders for {len(piper_gender_map)} Piper speakers.")
        except Exception as e:
            print(f"Warning: Error parsing Piper JSON config for gender mapping: {e}")
    else:
        print("Warning: Could not link gender map. Missing Piper .json config.")
        
    return piper_gender_map

# =============================================================================
# 2. CSV MANAGEMENT
# =============================================================================
def load_phrases_from_csv(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CRITICAL: {csv_path} not found.")
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    
    phrases_data = []
    for row in reader:
        phrase = row.get('phrase', row.get('text', '')).strip()
        phonemes = row.get('phonemes', '').strip()
        if phrase and phonemes:
            phrases_data.append({'phrase': phrase, 'phonemes': phonemes})
    return phrases_data, reader

def update_csv_remaining(csv_path, original_reader, successful_phrases):
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
# 3. CTC ALIGNMENT & PHONETIC SCORING
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
    text = text.upper()
    text = re.sub(r'[^A-Z]', '', text) 
    
    text = re.sub(r'[Z]', 'S', text)
    text = re.sub(r'[CQ]', 'K', text)
    text = re.sub(r'[VF]', 'B', text) 
    text = re.sub(r'[G]', 'J', text)
    
    text = text.replace('PH', 'F')
    text = text.replace('SH', 'S')
    text = text.replace('CH', 'J')
    text = text.replace('TH', 'T')
    
    text = re.sub(r'[AEIOUYW]+', 'A', text) 
    text = re.sub(r'(.)\1+', r'\1', text)
    
    return text

def verify_match_strict(transcript, target_text, threshold):
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
    no_punct = re.sub(r'[^\w\s]', '', raw_text)
    clean_text = re.sub(r'\s+', ' ', no_punct).strip()
    return clean_text.capitalize()

def process_and_clean_audio_tensor(waveform, sr, start_t, end_t, total_dur, buffer=0.1, fade_dur=0.05):
    trim_start_time = max(0.0, start_t - buffer)
    trim_end_time = min(total_dur, end_t + buffer)
    
    start_idx = int(trim_start_time * sr)
    end_idx = int(trim_end_time * sr)
    trimmed = waveform[:, start_idx:end_idx]
    
    trimmed = trimmed - trimmed.mean(dim=1, keepdim=True)
    trimmed = torchaudio.functional.highpass_biquad(trimmed, sr, cutoff_freq=50.0)
    
    fade_samples = int(fade_dur * sr)
    fade_samples = min(fade_samples, trimmed.shape[1] // 2)
    
    if fade_samples > 0:
        fade_in = torch.linspace(0.0, 1.0, steps=fade_samples)
        fade_out = torch.linspace(1.0, 0.0, steps=fade_samples)
        trimmed[0, :fade_samples] *= fade_in
        trimmed[0, -fade_samples:] *= fade_out

    return trimmed

# =============================================================================
# 4. MASTER PIPER GENERATOR PIPELINE
# =============================================================================
def generate_piper_dataset(args):
    os.makedirs(args.dest_dir, exist_ok=True)
    is_csv_mode = args.wordlist.lower().endswith('.csv')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nLoading Piper Pipeline on {device.type.upper()}...")
    
    # --- LOAD PIPER WEIGHTS & LOCAL GENDER MAP ---
    token2id = load_piper_tokens(args.tokens_path)
    config_json_path = args.model_path + ".json"
    gender_map = load_piper_gender_map(config_json_path)
    
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    sess = ort.InferenceSession(args.model_path, providers=providers)
    piper_native_sr = 22050 
    
    # Locked, deterministic Inference Scales (Slower speech, consistent behavior)
    scales = np.array([0.667, 1.3, 0.8], dtype=np.float32)

    print("Loading Wav2Vec2 CTC Validator...")
    asr_bundle = WAV2VEC2_ASR_BASE_960H
    asr_model = asr_bundle.get_model().to(device)
    asr_labels = asr_bundle.get_labels()
    asr_blank_idx = 0 
    
    speakers_list = list(range(args.num_speakers))
    print(f"Matrix configured: {len(speakers_list)} speakers × 1 language (en).")

    matrix_combinations = list(itertools.product(speakers_list, ["en"]))

    # =========================================================================
    # MODE A: CSV WORDLIST
    # =========================================================================
    if is_csv_mode:
        phrases_data, original_reader = load_phrases_from_csv(args.wordlist)
        print(f"CSV Mode Detected. Loaded {len(phrases_data)} phrases.")
        
        matrix_cycle = itertools.cycle(matrix_combinations)
        successful_phrases = []
        
        for item in tqdm(phrases_data, desc="Processing CSV Wordlist"):
            if args.max_num > 0 and len(successful_phrases) >= args.max_num:
                print(f"\nReached requested quota of {args.max_num}. Stopping early.")
                break
                
            speaker_idx, target_lang = next(matrix_cycle)
            
            phrase = item['phrase']
            raw_phons = item['phonemes'].split()
            
            safe_phrase = re.sub(r'[^a-zA-Z0-9_]', '', phrase.replace(' ', '_').lower())
            
            # --- Map Gender ---
            gender = gender_map.get(speaker_idx, "")
            gender_suffix = f"_{gender}" if gender else ""
            
            output_filepath = os.path.join(args.dest_dir, f"{safe_phrase}_{speaker_idx}{gender_suffix}_{target_lang}.wav")
            
            if os.path.exists(output_filepath):
                successful_phrases.append(phrase)
                continue
                
            formatted_text = format_tts_text(phrase)
            input_tensor, input_lengths = arpabet_to_piper_tensor(raw_phons, token2id)
            sid_tensor = np.array([speaker_idx], dtype=np.int64)

            passed_ctc = False
            overall_best_score = -1.0
            overall_best_trans = ""
            overall_best_reason = ""
            
            for attempt in range(args.max_retries):
                try:
                    ort_inputs = {
                        "input": input_tensor,
                        "input_lengths": input_lengths,
                        "scales": scales,
                        "sid": sid_tensor
                    }
                    
                    audio_np = sess.run(None, ort_inputs)[0]
                    wav_tensor = torch.from_numpy(audio_np).float().squeeze().unsqueeze(0)
                    
                    full_transcript, words, total_dur = evaluate_tensor_ctc_words(wav_tensor, piper_native_sr, asr_model, asr_bundle, device, asr_labels, asr_blank_idx)
                    
                    if not words: continue
                    
                    is_valid, sim, reason = verify_match_strict(full_transcript, formatted_text, args.threshold)
                    speech_dur = (words[-1]['end'] + 1) * 0.02 - (words[0]['start'] * 0.02)
                    
                    if is_valid and speech_dur > args.max_duration:
                        is_valid = False
                        reason = f"DURATION TOO LONG ({speech_dur:.2f}s > {args.max_duration}s)"
                    
                    if sim > overall_best_score:
                        overall_best_score = sim
                        overall_best_trans = full_transcript
                        overall_best_reason = reason
                    
                    if is_valid:
                        if piper_native_sr != args.target_sr:
                            wav_tensor = torchaudio.functional.resample(wav_tensor, piper_native_sr, args.target_sr)
                        
                        start_t = words[0]['start'] * 0.02
                        end_t = (words[-1]['end'] + 1) * 0.02
                        final_wav = process_and_clean_audio_tensor(wav_tensor, args.target_sr, start_t, end_t, total_dur)

                        torchaudio.save(output_filepath, final_wav, args.target_sr, encoding="PCM_S", bits_per_sample=16)
                        passed_ctc = True
                        successful_phrases.append(phrase)
                        tqdm.write(f"  [PASS] {speaker_idx}[{target_lang}] Target: '{formatted_text}' -> Heard: '{full_transcript}' | Score: {sim:.2f}")
                        break
                        
                except Exception as e:
                    tqdm.write(f"  [ERROR] Attempting {formatted_text}: {e}")
            
            if not passed_ctc:
                tqdm.write(f"  [FAILED] '{formatted_text}' exhausted retries. Heard: '{overall_best_trans}' | Score: {overall_best_score:.2f} | Reason: {overall_best_reason}")
                
        update_csv_remaining(args.wordlist, original_reader, successful_phrases)
        print(f"\n[COMPLETE] Processed CSV. Removed {len(successful_phrases)} successful generations from {args.wordlist}.")

    # =========================================================================
    # MODE B: SINGLE PHRASE
    # =========================================================================
    else:
        print(f"Single Phrase Mode Detected: '{args.wordlist}'")
        if not args.arpabet:
            raise ValueError("CRITICAL: You must provide the phonemes via --arpabet when running a single phrase bypassing the CSV.")
            
        formatted_text = format_tts_text(args.wordlist)
        safe_phrase = re.sub(r'[^a-zA-Z0-9_]', '', args.wordlist.replace(' ', '_').lower())
        raw_phons = args.arpabet.strip().split()
        
        input_tensor, input_lengths = arpabet_to_piper_tensor(raw_phons, token2id)

        successful_count = 0
        
        for speaker_idx, target_lang in tqdm(matrix_combinations, desc="Matrix Generation"):
            if args.max_num > 0 and successful_count >= args.max_num:
                print(f"\nReached requested quota of {args.max_num}. Stopping early.")
                break
                
            sid_tensor = np.array([speaker_idx], dtype=np.int64)
            
            # --- Map Gender ---
            gender = gender_map.get(speaker_idx, "")
            gender_suffix = f"_{gender}" if gender else ""
            
            output_filepath = os.path.join(args.dest_dir, f"{safe_phrase}_{speaker_idx}{gender_suffix}_{target_lang}.wav")
            
            if os.path.exists(output_filepath):
                successful_count += 1
                continue
            
            passed_ctc = False
            overall_best_score = -1.0
            overall_best_trans = ""
            overall_best_reason = ""
            
            for attempt in range(args.max_retries):
                try:
                    ort_inputs = {
                        "input": input_tensor,
                        "input_lengths": input_lengths,
                        "scales": scales,
                        "sid": sid_tensor
                    }
                    
                    audio_np = sess.run(None, ort_inputs)[0]
                    wav_tensor = torch.from_numpy(audio_np).float().squeeze().unsqueeze(0)
                    
                    full_transcript, words, total_dur = evaluate_tensor_ctc_words(wav_tensor, piper_native_sr, asr_model, asr_bundle, device, asr_labels, asr_blank_idx)
                    
                    if not words: continue
                    
                    is_valid, sim, reason = verify_match_strict(full_transcript, formatted_text, args.threshold)
                    speech_dur = (words[-1]['end'] + 1) * 0.02 - (words[0]['start'] * 0.02)
                    
                    if is_valid and speech_dur > args.max_duration:
                        is_valid = False
                        reason = f"DURATION TOO LONG ({speech_dur:.2f}s > {args.max_duration}s)"
                    
                    if sim > overall_best_score:
                        overall_best_score = sim
                        overall_best_trans = full_transcript
                        overall_best_reason = reason
                    
                    if is_valid:
                        if piper_native_sr != args.target_sr:
                            wav_tensor = torchaudio.functional.resample(wav_tensor, piper_native_sr, args.target_sr)
                        
                        start_t = words[0]['start'] * 0.02
                        end_t = (words[-1]['end'] + 1) * 0.02
                        final_wav = process_and_clean_audio_tensor(wav_tensor, args.target_sr, start_t, end_t, total_dur)

                        torchaudio.save(output_filepath, final_wav, args.target_sr, encoding="PCM_S", bits_per_sample=16)
                        passed_ctc = True
                        successful_count += 1
                        tqdm.write(f"  [PASS] {speaker_idx}[{target_lang}] Target: '{formatted_text}' -> Heard: '{full_transcript}' | Score: {sim:.2f}")
                        break
                        
                except Exception as e:
                    tqdm.write(f"  [ERROR] Attempting {formatted_text}: {e}")
            
            if not passed_ctc:
                tqdm.write(f"  [FAILED] '{formatted_text}' exhausted retries. Heard: '{overall_best_trans}' | Score: {overall_best_score:.2f} | Reason: {overall_best_reason}")
                
        print(f"\n[COMPLETE] Finished Matrix Generation. Total successful files: {successful_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Piper Generation Script")
    parser.add_argument("--wordlist", type=str, required=True, help="Either a .csv file path OR a text phrase like 'Hey Jarvis'")
    parser.add_argument("--arpabet", type=str, required=False, help="Required for single phrases. E.g., 'HH EY1 | JH AA1 R V IH0 S'")
    parser.add_argument("--dest_dir", type=str, required=True, help="Output directory for generated WAV files")
    
    parser.add_argument("--num_speakers", type=int, default=904, help="Number of speakers in the Piper model")
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate to save the final audio")
    
    parser.add_argument("--threshold", type=float, default=0.65, help="Phonetic match threshold")
    parser.add_argument("--max_retries", type=int, default=10, help="Max retry attempts per generation")
    parser.add_argument("--max_duration", type=float, default=1.8, help="Max duration in seconds")
    parser.add_argument("--max_num", type=int, default=0, help="Maximum number of successful generations to produce (0 = all)")
    args = parser.parse_args()
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    args.model_path = os.path.join(SCRIPT_DIR, "vits-piper-en_US-libritts_r-medium", "en_US-libritts_r-medium.onnx")
    args.tokens_path = os.path.join(SCRIPT_DIR, "vits-piper-en_US-libritts_r-medium", "tokens.txt")
    
    generate_piper_dataset(args)