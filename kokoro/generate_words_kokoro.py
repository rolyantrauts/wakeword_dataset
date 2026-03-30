import os
import re
import csv
import torch
import torchaudio
import warnings
import difflib
import itertools
import numpy as np
from tqdm import tqdm
from huggingface_hub import snapshot_download
from kokoro import KPipeline
from itertools import combinations
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

# =============================================================================
# 1. KOKORO ARPABET -> IPA TRANSLATOR
# =============================================================================
ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ə", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "ʧ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɚ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "ʤ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ"
}

def arpabet_to_kokoro_ipa(raw_phons):
    ipa_str = ""
    for token in raw_phons:
        if token == "|":
            ipa_str += " "
        elif token in [".", ",", "!", "?"]:
            ipa_str += token
        else:
            stress_mark = ""
            if token[-1] in "012":
                stress_val = token[-1]
                token = token[:-1]  
                if stress_val == "1":
                    stress_mark = "ˈ" 
                elif stress_val == "2":
                    stress_mark = "ˌ" 
                    
            if token in ARPABET_TO_IPA:
                ipa_str += stress_mark + ARPABET_TO_IPA[token]
    return ipa_str

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
# 4. KOKORO VOICE POOL BUILDER
# =============================================================================
AUTHORIZED_VOICES = {
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore", 
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky", "bf_alice", 
    "bf_emma", "bf_isabella", "bf_lily", "ef_dora", "ff_siwis", "hf_alpha", 
    "hf_beta", "if_sara", "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", 
    "pf_dora", "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael", "am_onyx", 
    "am_puck", "am_santa", "bm_daniel", "bm_fable", "bm_george", "bm_lewis", 
    "em_alex", "em_santa", "hm_omega", "hm_psi", "im_nicola", "jm_kumo", 
    "pm_alex", "pm_santa", "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"
}

def build_voice_pool(device):
    print("\nFetching explicit whitelist voice profiles from Hugging Face...")
    v1_dir = snapshot_download(repo_id="hexgrad/Kokoro-82M", allow_patterns="voices/*.pt")
    v1_1_dir = snapshot_download(repo_id="hexgrad/Kokoro-82M-v1.1-zh", allow_patterns="voices/*.pt")
    
    base_tensors = {}
    for d in [v1_dir, v1_1_dir]:
        v_dir_path = os.path.join(d, "voices")
        if os.path.exists(v_dir_path):
            for f in os.listdir(v_dir_path):
                if f.endswith('.pt'):
                    name = f.replace('.pt', '')
                    if name in AUTHORIZED_VOICES:
                        tensor_path = os.path.join(v_dir_path, f)
                        base_tensors[name] = torch.load(tensor_path, map_location='cpu', weights_only=True)
                        
    sorted_voices = dict(sorted(base_tensors.items()))
    male_voices = {n: t for n, t in sorted_voices.items() if len(n) > 1 and n[1].lower() == 'm'}
    female_voices = {n: t for n, t in sorted_voices.items() if len(n) > 1 and n[1].lower() == 'f'}
    
    voice_pool = {}
    
    # 1. Pure Voices
    for name, tensor in sorted_voices.items():
        voice_pool[f"pure_{name}"] = tensor.to(device)
        
    # 2. Male Blends
    for n1, n2 in combinations(male_voices.keys(), 2):
        sorted_names = sorted([n1, n2])
        blend_name = f"blend_{sorted_names[0]}_x_{sorted_names[1]}"
        min_len = min(male_voices[n1].shape[0], male_voices[n2].shape[0])
        blended_t = (male_voices[n1][:min_len] * 0.5) + (male_voices[n2][:min_len] * 0.5)
        voice_pool[blend_name] = blended_t.to(device)
        
    # 3. Female Blends
    for n1, n2 in combinations(female_voices.keys(), 2):
        sorted_names = sorted([n1, n2])
        blend_name = f"blend_{sorted_names[0]}_x_{sorted_names[1]}"
        min_len = min(female_voices[n1].shape[0], female_voices[n2].shape[0])
        blended_t = (female_voices[n1][:min_len] * 0.5) + (female_voices[n2][:min_len] * 0.5)
        voice_pool[blend_name] = blended_t.to(device)
        
    print(f"Constructed {len(voice_pool)} deterministic acoustic signatures (Pure + Blends).")
    return voice_pool

# =============================================================================
# 5. MASTER KOKORO GENERATOR PIPELINE
# =============================================================================
def generate_kokoro_dataset(args):
    os.makedirs(args.dest_dir, exist_ok=True)
    is_csv_mode = args.wordlist.lower().endswith('.csv')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nLoading Kokoro Pipeline on {device.type.upper()}...")
    
    # --- LOAD KOKORO ---
    pipeline = KPipeline(lang_code='a')
    model_device = next(pipeline.model.parameters()).device
    
    # --- BUILD VOICE POOL ---
    voice_pool = build_voice_pool(model_device)
    speakers_list = sorted(list(voice_pool.keys()))

    # --- LOAD CTC VALIDATOR ---
    print("Loading Wav2Vec2 CTC Validator...")
    asr_bundle = WAV2VEC2_ASR_BASE_960H
    asr_model = asr_bundle.get_model().to(device)
    asr_labels = asr_bundle.get_labels()
    asr_blank_idx = 0 
    kokoro_native_sr = 24000
    
    matrix_combinations = list(itertools.product(speakers_list, ["en"]))
    print(f"Matrix configured: {len(speakers_list)} speakers × 1 language (en) = {len(matrix_combinations)} combinations per phrase.\n")

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
                
            speaker_id, target_lang = next(matrix_cycle)
            voice_tensor = voice_pool[speaker_id]
            
            phrase = item['phrase']
            raw_phons = item['phonemes'].split()
            
            safe_phrase = re.sub(r'[^a-zA-Z0-9_]', '', phrase.replace(' ', '_').lower())
            output_filepath = os.path.join(args.dest_dir, f"{safe_phrase}_{speaker_id}_{target_lang}.wav")
            
            if os.path.exists(output_filepath):
                successful_phrases.append(phrase)
                continue
                
            formatted_text = format_tts_text(phrase)
            ipa_tokens = arpabet_to_kokoro_ipa(raw_phons)

            passed_ctc = False
            overall_best_score = -1.0
            overall_best_trans = ""
            overall_best_reason = ""
            
            for attempt in range(args.max_retries):
                try:
                    # STRICT PIPELINE: Zero randomization.
                    current_speed = 1.0
                    
                    generator = pipeline.generate_from_tokens(tokens=ipa_tokens, voice=voice_tensor, speed=current_speed)
                    audio_chunks = [result.audio.cpu().numpy() for result in generator if result.audio is not None]
                    
                    if not audio_chunks: continue
                    
                    audio_np = np.concatenate(audio_chunks)
                    wav_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
                    
                    full_transcript, words, total_dur = evaluate_tensor_ctc_words(wav_tensor, kokoro_native_sr, asr_model, asr_bundle, device, asr_labels, asr_blank_idx)
                    
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
                        if kokoro_native_sr != args.target_sr:
                            wav_tensor = torchaudio.functional.resample(wav_tensor, kokoro_native_sr, args.target_sr)
                        
                        start_t = words[0]['start'] * 0.02
                        end_t = (words[-1]['end'] + 1) * 0.02
                        final_wav = process_and_clean_audio_tensor(wav_tensor, args.target_sr, start_t, end_t, total_dur)

                        torchaudio.save(output_filepath, final_wav, args.target_sr, encoding="PCM_S", bits_per_sample=16)
                        passed_ctc = True
                        successful_phrases.append(phrase)
                        tqdm.write(f"  [PASS] {speaker_id}[{target_lang}] Target: '{formatted_text}' -> Heard: '{full_transcript}' | Score: {sim:.2f}")
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
        
        ipa_tokens = arpabet_to_kokoro_ipa(raw_phons)

        successful_count = 0
        
        for speaker_id, target_lang in tqdm(matrix_combinations, desc="Matrix Generation"):
            if args.max_num > 0 and successful_count >= args.max_num:
                print(f"\nReached requested quota of {args.max_num}. Stopping early.")
                break
                
            voice_tensor = voice_pool[speaker_id]
            output_filepath = os.path.join(args.dest_dir, f"{safe_phrase}_{speaker_id}_{target_lang}.wav")
            
            if os.path.exists(output_filepath):
                successful_count += 1
                continue
            
            passed_ctc = False
            overall_best_score = -1.0
            overall_best_trans = ""
            overall_best_reason = ""
            
            for attempt in range(args.max_retries):
                try:
                    # STRICT PIPELINE: Zero randomization.
                    current_speed = 1.0 
                    
                    generator = pipeline.generate_from_tokens(tokens=ipa_tokens, voice=voice_tensor, speed=current_speed)
                    audio_chunks = [result.audio.cpu().numpy() for result in generator if result.audio is not None]
                    
                    if not audio_chunks: continue
                    
                    audio_np = np.concatenate(audio_chunks)
                    wav_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
                    
                    full_transcript, words, total_dur = evaluate_tensor_ctc_words(wav_tensor, kokoro_native_sr, asr_model, asr_bundle, device, asr_labels, asr_blank_idx)
                    
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
                        if kokoro_native_sr != args.target_sr:
                            wav_tensor = torchaudio.functional.resample(wav_tensor, kokoro_native_sr, args.target_sr)
                        
                        start_t = words[0]['start'] * 0.02
                        end_t = (words[-1]['end'] + 1) * 0.02
                        final_wav = process_and_clean_audio_tensor(wav_tensor, args.target_sr, start_t, end_t, total_dur)
                        
                        torchaudio.save(output_filepath, final_wav, args.target_sr, encoding="PCM_S", bits_per_sample=16)
                        passed_ctc = True
                        successful_count += 1
                        tqdm.write(f"  [PASS] {speaker_id}[{target_lang}] Target: '{formatted_text}' -> Heard: '{full_transcript}' | Score: {sim:.2f}")
                        break
                        
                except Exception as e:
                    tqdm.write(f"  [ERROR] Attempting {formatted_text}: {e}")
            
            if not passed_ctc:
                tqdm.write(f"  [FAILED] '{formatted_text}' in {speaker_id}[{target_lang}] exhausted retries. Heard: '{overall_best_trans}' | Score: {overall_best_score:.2f} | Reason: {overall_best_reason}")
                
        print(f"\n[COMPLETE] Finished Matrix Generation. Total successful files: {successful_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Kokoro Generation Script")
    parser.add_argument("--wordlist", type=str, required=True, help="Either a .csv file path OR a text phrase like 'Hey Jarvis'")
    parser.add_argument("--arpabet", type=str, required=False, help="Required for single phrases. E.g., 'HH EY1 | JH AA1 R V IH0 S'")
    parser.add_argument("--dest_dir", type=str, required=True, help="Output directory for generated WAV files")
    
    parser.add_argument("--target_sr", type=int, default=16000, help="Target sample rate to save the final audio")
    parser.add_argument("--threshold", type=float, default=0.65, help="Phonetic match threshold")
    parser.add_argument("--max_retries", type=int, default=10, help="Max retry attempts per generation")
    parser.add_argument("--max_duration", type=float, default=1.8, help="Max duration in seconds")
    parser.add_argument("--max_num", type=int, default=0, help="Maximum number of successful generations to produce (0 = all)")
    args = parser.parse_args()
    
    generate_kokoro_dataset(args)
