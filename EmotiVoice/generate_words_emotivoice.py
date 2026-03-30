import os
import sys
import re
import csv
import torch
import torchaudio
import warnings
import difflib
import itertools
import gc
from tqdm import tqdm
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H

warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")

# =============================================================================
# 1. EMOTIVOICE CORE IMPORTS
# =============================================================================
from models.prompt_tts_modified.jets import JETSGenerator
from models.prompt_tts_modified.simbert import StyleEncoder
from transformers import AutoTokenizer
from yacs import config as CONFIG 

# =============================================================================
# 2. CSV MANAGEMENT & GENDER MAPPING
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

def load_gender_map(male_csv="Voices-Male.csv", female_csv="Voices-Female.csv"):
    """Loads speaker gender mapping from the local CSV files."""
    gender_map = {}
    
    if os.path.exists(male_csv):
        with open(male_csv, 'r', encoding='utf-8') as f:
            for row in csv.reader(f):
                if row and len(row) > 0:
                    gender_map[row[0].strip()] = 'm'
                    
    if os.path.exists(female_csv):
        with open(female_csv, 'r', encoding='utf-8') as f:
            for row in csv.reader(f):
                if row and len(row) > 0:
                    gender_map[row[0].strip()] = 'f'
                    
    return gender_map

# =============================================================================
# 3. CTC ALIGNMENT & PHONETIC SCORING
# =============================================================================
def get_style_embedding(prompt, tokenizer, style_encoder, device):
    prompt_tokens = tokenizer([prompt], return_tensors="pt")
    input_ids = prompt_tokens["input_ids"].to(device)
    token_type_ids = prompt_tokens["token_type_ids"].to(device)
    attention_mask = prompt_tokens["attention_mask"].to(device)
    
    with torch.no_grad():
        output = style_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
    return output["pooled_output"].detach().cpu().squeeze().numpy()

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
# 4. MASTER EMOTIVOICE GENERATOR PIPELINE
# =============================================================================
def generate_emotivoice_dataset(args, config):
    os.makedirs(args.dest_dir, exist_ok=True)
    is_csv_mode = args.wordlist.lower().endswith('.csv')
    
    # Load the gender mappings mapping from CSVs in the root folder
    gender_map = load_gender_map()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"\nLoading EmotiVoice Pipeline on {device.type.upper()}...")
    
    # --- LOAD EMOTIVOICE WEIGHTS ---
    ckpt_path = os.path.join(config.output_directory, args.logdir, "ckpt")
    checkpoint_file = args.checkpoint if args.checkpoint else os.listdir(ckpt_path)[0]
    checkpoint_path = os.path.join(ckpt_path, checkpoint_file)

    with open(config.model_config_path, 'r') as fin: 
        conf = CONFIG.load_cfg(fin)
    
    conf.n_vocab = config.n_symbols
    conf.n_speaker = config.speaker_n_labels

    style_encoder = StyleEncoder(config).to(device)
    style_CKPT = torch.load(config.style_encoder_ckpt, map_location=device, weights_only=False)
    style_encoder.load_state_dict({k[7:]: v for k, v in style_CKPT['model'].items()}, strict=False)

    generator = JETSGenerator(conf).to(device)
    gen_CKPT = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator.load_state_dict(gen_CKPT['generator'])
    generator.eval()

    with open(config.token_list_path, 'r') as f: 
        token2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}
    with open(config.speaker2id_path, encoding='utf-8') as f: 
        speaker2id = {t.strip(): idx for idx, t in enumerate(f.readlines())}
    
    tokenizer = AutoTokenizer.from_pretrained(config.bert_path)
    neutral_style_emb = get_style_embedding("Neutral", tokenizer, style_encoder, device)
    del style_encoder, tokenizer

    # --- LOAD CTC VALIDATOR ---
    print("Loading Wav2Vec2 CTC Validator...")
    asr_bundle = WAV2VEC2_ASR_BASE_960H
    asr_model = asr_bundle.get_model().to(device)
    asr_labels = asr_bundle.get_labels()
    asr_blank_idx = 0 
    sr = config.sampling_rate
    
    speakers_list = sorted(list(speaker2id.keys()))
    print(f"Loaded {len(speakers_list)} EmotiVoice speakers.")

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
                
            speaker_str, target_lang = next(matrix_cycle)
            speaker_idx = speaker2id[speaker_str]
            
            phrase = item['phrase']
            raw_phons = item['phonemes'].split()
            
            safe_phrase = re.sub(r'[^a-zA-Z0-9_]', '', phrase.replace(' ', '_').lower())
            
            # --- Append Gender Flag ---
            gender = gender_map.get(speaker_str, "")
            gender_suffix = f"_{gender}" if gender else ""
            
            output_filepath = os.path.join(args.dest_dir, f"{safe_phrase}_{speaker_str}{gender_suffix}_{target_lang}.wav")
            
            if os.path.exists(output_filepath):
                successful_phrases.append(phrase)
                continue
                
            formatted_text = format_tts_text(phrase)
            
            # --- Format phonemes for EmotiVoice Tensor ---
            formatted_phons = ["<sos/eos>"]
            for p in raw_phons:
                if p == '|':
                    formatted_phons.append(args.pause_token)
                else:
                    formatted_phons.append(f"[{p}]")
            formatted_phons.append("<sos/eos>")
            
            if not all(ph in token2id for ph in formatted_phons):
                tqdm.write(f"  [SKIPPED] Missing phoneme mapping for: '{phrase}'. Check token2id for elements like {formatted_phons}")
                continue 
                
            text_int = [token2id[ph] for ph in formatted_phons]
            sequence = torch.tensor(text_int, device=device).unsqueeze(0)
            sequence_len = torch.tensor([len(text_int)], device=device)
            style_tensor = torch.tensor(neutral_style_emb, device=device).unsqueeze(0)
            speaker_tensor = torch.tensor([speaker_idx], device=device)

            passed_ctc = False
            overall_best_score = -1.0
            overall_best_trans = ""
            overall_best_reason = ""
            
            for attempt in range(args.max_retries):
                try:
                    current_alpha = 1.0
                    
                    with torch.no_grad():
                        infer_output = generator(
                            inputs_ling=sequence,
                            inputs_style_embedding=style_tensor,
                            input_lengths=sequence_len,
                            inputs_content_embedding=style_tensor,
                            inputs_speaker=speaker_tensor,
                            alpha=current_alpha 
                        )
                        
                    wav_tensor = infer_output["wav_predictions"].squeeze().unsqueeze(0).cpu()
                    full_transcript, words, total_dur = evaluate_tensor_ctc_words(wav_tensor, sr, asr_model, asr_bundle, device, asr_labels, asr_blank_idx)
                    
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
                        wav_tensor = torch.clamp(wav_tensor, -1.0, 1.0)
                        torchaudio.save(output_filepath, wav_tensor, sr, encoding="PCM_S", bits_per_sample=16)
                        passed_ctc = True
                        successful_phrases.append(phrase)
                        tqdm.write(f"  [PASS] {speaker_str}[{target_lang}] Target: '{formatted_text}' -> Heard: '{full_transcript}' | Score: {sim:.2f}")
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
        
        # --- Format phonemes for EmotiVoice Tensor ---
        formatted_phons = ["<sos/eos>"]
        for p in raw_phons:
            if p == '|':
                formatted_phons.append(args.pause_token)
            else:
                formatted_phons.append(f"[{p}]")
        formatted_phons.append("<sos/eos>")
        
        if not all(ph in token2id for ph in formatted_phons):
            raise ValueError(f"CRITICAL: Provided Arpabet phonemes do not exist in the EmotiVoice token2id mapping. Parsed list: {formatted_phons}")
            
        text_int = [token2id[ph] for ph in formatted_phons]
        sequence = torch.tensor(text_int, device=device).unsqueeze(0)
        sequence_len = torch.tensor([len(text_int)], device=device)
        style_tensor = torch.tensor(neutral_style_emb, device=device).unsqueeze(0)

        successful_count = 0
        
        for speaker_str, target_lang in tqdm(matrix_combinations, desc="Matrix Generation"):
            if args.max_num > 0 and successful_count >= args.max_num:
                print(f"\nReached requested quota of {args.max_num}. Stopping early.")
                break
                
            speaker_idx = speaker2id[speaker_str]
            speaker_tensor = torch.tensor([speaker_idx], device=device)
            
            # --- Append Gender Flag ---
            gender = gender_map.get(speaker_str, "")
            gender_suffix = f"_{gender}" if gender else ""
            
            output_filepath = os.path.join(args.dest_dir, f"{safe_phrase}_{speaker_str}{gender_suffix}_{target_lang}.wav")
            
            if os.path.exists(output_filepath):
                successful_count += 1
                continue
            
            passed_ctc = False
            overall_best_score = -1.0
            overall_best_trans = ""
            overall_best_reason = ""
            
            for attempt in range(args.max_retries):
                try:
                    current_alpha = 1.0 
                    
                    with torch.no_grad():
                        infer_output = generator(
                            inputs_ling=sequence,
                            inputs_style_embedding=style_tensor,
                            input_lengths=sequence_len,
                            inputs_content_embedding=style_tensor,
                            inputs_speaker=speaker_tensor,
                            alpha=current_alpha 
                        )
                        
                    wav_tensor = infer_output["wav_predictions"].squeeze().unsqueeze(0).cpu()
                    full_transcript, words, total_dur = evaluate_tensor_ctc_words(wav_tensor, sr, asr_model, asr_bundle, device, asr_labels, asr_blank_idx)
                    
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
                        wav_tensor = torch.clamp(wav_tensor, -1.0, 1.0)
                        torchaudio.save(output_filepath, wav_tensor, sr, encoding="PCM_S", bits_per_sample=16)
                        passed_ctc = True
                        successful_count += 1
                        tqdm.write(f"  [PASS] {speaker_str}[{target_lang}] Target: '{formatted_text}' -> Heard: '{full_transcript}' | Score: {sim:.2f}")
                        break
                        
                except Exception as e:
                    tqdm.write(f"  [ERROR] Attempting {formatted_text}: {e}")
            
            if not passed_ctc:
                tqdm.write(f"  [FAILED] '{formatted_text}' in {speaker_str}[{target_lang}] exhausted retries. Heard: '{overall_best_trans}' | Score: {overall_best_score:.2f} | Reason: {overall_best_reason}")
                
        print(f"\n[COMPLETE] Finished Matrix Generation. Total successful files: {successful_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified EmotiVoice Generation Script")
    parser.add_argument("--wordlist", type=str, required=True, help="Either a .csv file path OR a text phrase like 'Hey Jarvis'")
    parser.add_argument("--arpabet", type=str, required=False, help="Required for single phrases. E.g., 'HH EY1 | JH AA1 R V IH0 S'")
    parser.add_argument("--dest_dir", type=str, required=True, help="Output directory for generated WAV files")
    
    # EmotiVoice Core Paths
    parser.add_argument('-d', '--logdir', type=str, default='prompt_tts_open_source_joint', help='EmotiVoice logdir name')
    parser.add_argument("-c", "--config_folder", type=str, default='config/joint', help='EmotiVoice config path')
    parser.add_argument("--checkpoint", type=str, required=False, default='g_00140000', help='EmotiVoice checkpoint')
    
    parser.add_argument("--threshold", type=float, default=0.65, help="Phonetic match threshold")
    parser.add_argument("--pause_token", type=str, default="engsp4", help="Token used to replace the | boundary marker")
    parser.add_argument("--max_retries", type=int, default=10, help="Max retry attempts per generation")
    parser.add_argument("--max_duration", type=float, default=1.8, help="Max duration in seconds")
    parser.add_argument("--max_num", type=int, default=0, help="Maximum number of successful generations to produce (0 = all)")
    args = parser.parse_args()
    
    # Enable internal module loading directly from EmotiVoice directory
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), args.config_folder))
    from config import Config
    
    generate_emotivoice_dataset(args, Config())