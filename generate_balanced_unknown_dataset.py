import sqlite3
import csv
import random

def load_exclusion_set(adversarial_csv):
    """
    Loads the previously generated adversarial phrases into an O(1) 
    lookup set to guarantee zero cross-contamination.
    """
    exclusion_set = set()
    try:
        with open(adversarial_csv, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                exclusion_set.add(row['phrase'].upper())
        print(f"Loaded {len(exclusion_set):,} phrases into the exclusion filter.")
    except FileNotFoundError:
        print(f"Warning: {adversarial_csv} not found. Proceeding without exclusion list.")
    return exclusion_set

def fetch_words_by_stress(cursor, syllable_count, stress_pattern):
    """
    Queries the database for words matching exact syllable and stress constraints.
    stress_pattern is a tuple, e.g., ('1',) for a stressed 1-syllable word, 
    or ('1', '0') for a trochee.
    """
    table_name = "syllable"
    
    if syllable_count == 1:
        cursor.execute(f"SELECT word, s1 FROM {table_name} WHERE scount = 1 AND s1 LIKE '%{stress_pattern[0]}%'")
    elif syllable_count == 2:
        cursor.execute(f"SELECT word, s1, s2 FROM {table_name} WHERE scount = 2 AND s1 LIKE '%{stress_pattern[0]}%' AND s2 LIKE '%{stress_pattern[1]}%'")
    else:
        return []

    # Clean the data and return as a list of dictionaries
    results = []
    for row in cursor.fetchall():
        word = row[0].upper()
        # Reconstruct the phoneme string safely handling None values
        phons = " ".join([str(p).replace(',', '').strip() for p in row[1:] if p])
        results.append({"word": word, "phonemes": phons})
        
    return results

def generate_balanced_unknown_dataset(db_path, adversarial_csv, output_csv):
    """
    Generates exactly 80,000 phonetically and structurally balanced phrases,
    ensuring none exist in the adversarial exclusion list.
    """
    print(f"Connecting to database: {db_path}...")
    
    exclusion_set = load_exclusion_set(adversarial_csv)
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("\nFetching structural building blocks from database...")
        # 1-Syllable Blocks
        w1_stress_1 = fetch_words_by_stress(cursor, 1, ('1',))
        w1_stress_0 = fetch_words_by_stress(cursor, 1, ('0',))
        
        # 2-Syllable Blocks
        w2_stress_1_0 = fetch_words_by_stress(cursor, 2, ('1', '0')) # Trochee (like Jarvis)
        w2_stress_0_1 = fetch_words_by_stress(cursor, 2, ('0', '1')) # Iamb
        w2_stress_0_0 = fetch_words_by_stress(cursor, 2, ('0', '0')) # Pyrrhic
        
        print(f"  -> Found {len(w1_stress_1):,} Primary-Stressed 1-syllable words")
        print(f"  -> Found {len(w2_stress_1_0):,} Trochaic (1-0) 2-syllable words")
        
        final_dataset = []
        
        # Define our architectural quotas
        categories = [
            {"name": "Structural_Mimic_1-1-0", "w1_pool": w1_stress_1, "w2_pool": w2_stress_1_0, "quota": 20000},
            {"name": "Inverted_Cadence_0-1-0", "w1_pool": w1_stress_0, "w2_pool": w2_stress_1_0, "quota": 20000},
            {"name": "Bouncing_Cadence_1-0-1", "w1_pool": w1_stress_1, "w2_pool": w2_stress_0_1, "quota": 20000},
            {"name": "Dactyl_Cadence_1-0-0",   "w1_pool": w1_stress_1, "w2_pool": w2_stress_0_0, "quota": 20000}
        ]
        
        print("\nExecuting randomized, collision-free phrase generation...")
        
        for cat in categories:
            cat_name = cat["name"]
            w1_pool = cat["w1_pool"]
            w2_pool = cat["w2_pool"]
            target_quota = cat["quota"]
            
            accepted_phrases = 0
            # Safety counter to prevent infinite loops if the dictionary is too small
            attempts = 0 
            max_attempts = target_quota * 10
            
            while accepted_phrases < target_quota and attempts < max_attempts:
                attempts += 1
                
                # Pick random components
                word1 = random.choice(w1_pool)
                word2 = random.choice(w2_pool)
                
                phrase = f"{word1['word']} {word2['word']}"
                
                # Verify it is not in the adversarial list AND not already generated
                if phrase not in exclusion_set:
                    combined_phons = f"{word1['phonemes']} {word2['phonemes']}"
                    
                    final_dataset.append({
                        "phrase": phrase,
                        "phonemes": combined_phons,
                        "type": cat_name,
                        "distance": "N/A" # Not applicable for general unknown class
                    })
                    
                    # Add to exclusion set to prevent internal duplicates
                    exclusion_set.add(phrase)
                    accepted_phrases += 1
                    
            print(f"  -> Generated {accepted_phrases:,} phrases for {cat_name}.")

        print(f"\nWriting dataset to {output_csv}...")
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["phrase", "phonemes", "type", "distance"])
            writer.writeheader()
            writer.writerows(final_dataset)
            
        print("\n============================================================")
        print("BALANCED UNKNOWN DATASET GENERATION COMPLETE")
        print("============================================================")

    except sqlite3.Error as e:
        print(f"SQLite Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    DATABASE_PATH = "./words.db"
    ADVERSARIAL_CSV = "./adversarial_balanced_20k.csv"
    OUTPUT_CSV = "./unknown_balanced_80k.csv"
    
    generate_balanced_unknown_dataset(DATABASE_PATH, ADVERSARIAL_CSV, OUTPUT_CSV)
