import sqlite3
import time
import itertools

def calculate_levenshtein(seq1, seq2):
    """
    Calculates the Levenshtein Edit Distance between two lists of phonemes.
    """
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = [[0] * size_y for _ in range(size_x)]
    
    for x in range(size_x):
        matrix[x][0] = x
    for y in range(size_y):
        matrix[0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1],
                    matrix[x][y-1] + 1
                )
            else:
                matrix[x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1] + 1,
                    matrix[x][y-1] + 1
                )
    return matrix[size_x - 1][size_y - 1]

def build_exhaustive_adversarial_table(db_path):
    """
    Connects to the SQLite database, creates the adversarial table, 
    and exhaustively populates it with all PAD mimics and sound-alikes.
    """
    print(f"Connecting to database: {db_path}...")
    
    target_phonemes = ['HH', 'EY1', 'JH', 'AA1', 'R', 'V', 'AH0', 'S']
    batch_size = 25000  # Number of rows to hold in memory before committing to disk
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Set up the target table
        cursor.execute("DROP TABLE IF EXISTS adversarial")
        cursor.execute("""
            CREATE TABLE adversarial (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phrase TEXT,
                phonemes TEXT,
                adversarial_type TEXT,
                distance INTEGER
            )
        """)
        conn.commit()
        
        table_name = "syllable" 
        
        print("\n--- PHASE 1: Exhaustive PAD Correlation & Spectral Tilt Mimics ---")
        t_start = time.perf_counter()
        
        # Find ALL 1-syllable words with high-energy open vowels
        cursor.execute(f"""
            SELECT word, s1 FROM {table_name} 
            WHERE scount = 1 
            AND (s1 LIKE '%EY1%' OR s1 LIKE '%AY1%' OR s1 LIKE '%IY1%')
        """)
        pad_word1 = cursor.fetchall()
        
        # Find ALL 2-syllable words matching the "Jarvis" energy envelope
        cursor.execute(f"""
            SELECT word, s1, s2 FROM {table_name} 
            WHERE scount = 2 
            AND (s1 LIKE '%AA1%' OR s1 LIKE '%AO1%')
            AND (s2 LIKE '%AH0%' OR s2 LIKE '%IH0%')
            AND (s2 LIKE '%S%' OR s2 LIKE '%Z%' OR s2 LIKE '%F%' OR s2 LIKE '%V%')
        """)
        pad_word2 = cursor.fetchall()
        
        pad_insert_batch = []
        pad_total_count = 0
        
        # Cross-join all valid PAD components
        for w1, w2 in itertools.product(pad_word1, pad_word2):
            phrase = f"{w1[0]} {w2[0]}"
            # Safely handle nulls and clean the DB output
            s1_p1 = (w1[1] or "").replace(",", "").strip()
            s2_p1 = (w2[1] or "").replace(",", "").strip()
            s2_p2 = (w2[2] or "").replace(",", "").strip()
            phoneme_str = f"{s1_p1} {s2_p1} {s2_p2}".strip()
            
            pad_insert_batch.append((phrase, phoneme_str, "PAD_Spectral_Mimic", 0))
            pad_total_count += 1
            
            if len(pad_insert_batch) >= batch_size:
                cursor.executemany(
                    "INSERT INTO adversarial (phrase, phonemes, adversarial_type, distance) VALUES (?, ?, ?, ?)", 
                    pad_insert_batch
                )
                conn.commit()
                pad_insert_batch.clear()

        # Commit remaining PAD mimics
        if pad_insert_batch:
            cursor.executemany(
                "INSERT INTO adversarial (phrase, phonemes, adversarial_type, distance) VALUES (?, ?, ?, ?)", 
                pad_insert_batch
            )
            conn.commit()
            
        t_end = time.perf_counter()
        print(f"Generated and inserted {pad_total_count} exhaustive PAD Mimics in {t_end - t_start:.2f}s.")
        
        print("\n--- PHASE 2: Exhaustive Phonetic Levenshtein Distance (Sound-Alikes) ---")
        print("Note: This performs hundreds of millions of cross-comparisons. It will take some time.")
        t_start = time.perf_counter()
        
        cursor.execute(f"SELECT word, s1 FROM {table_name} WHERE scount = 1")
        all_word1 = cursor.fetchall()
        
        cursor.execute(f"SELECT word, s1, s2 FROM {table_name} WHERE scount = 2")
        all_word2 = cursor.fetchall()

        w1_dict = []
        for w in all_word1:
            phons = [p.strip() for p in (w[1] or "").replace(',', ' ').split()]
            if 1 <= len(phons) <= 3:
                w1_dict.append((w[0], phons))
                
        w2_dict = []
        for w in all_word2:
            s1_phons = [p.strip() for p in (w[1] or "").replace(',', ' ').split()]
            s2_phons = [p.strip() for p in (w[2] or "").replace(',', ' ').split() if p.strip()]
            phons = s1_phons + s2_phons
            if 4 <= len(phons) <= 7:
                w2_dict.append((w[0], phons))

        total_comparisons = len(w1_dict) * len(w2_dict)
        print(f"Combinatorial space to evaluate: {total_comparisons:,} phrases.")

        lev_insert_batch = []
        lev_total_count = 0
        calc_counter = 0

        for w1 in w1_dict:
            for w2 in w2_dict:
                calc_counter += 1
                
                # Print progress every 5 million calculations
                if calc_counter % 5000000 == 0:
                    percent_done = (calc_counter / total_comparisons) * 100
                    print(f"  -> Evaluated {calc_counter:,} combinations ({percent_done:.1f}%) | Found {lev_total_count:,} matches...")

                combined_phons = w1[1] + w2[1]
                dist = calculate_levenshtein(target_phonemes, combined_phons)
                
                if 1 <= dist <= 3:
                    phrase = f"{w1[0]} {w2[0]}"
                    phoneme_str = " ".join(combined_phons)
                    
                    lev_insert_batch.append((phrase, phoneme_str, f"Phonetic_Distance_{dist}", dist))
                    lev_total_count += 1
                    
                    if len(lev_insert_batch) >= batch_size:
                        cursor.executemany(
                            "INSERT INTO adversarial (phrase, phonemes, adversarial_type, distance) VALUES (?, ?, ?, ?)", 
                            lev_insert_batch
                        )
                        conn.commit()
                        lev_insert_batch.clear()

        # Commit remaining Levenshtein sound-alikes
        if lev_insert_batch:
            cursor.executemany(
                "INSERT INTO adversarial (phrase, phonemes, adversarial_type, distance) VALUES (?, ?, ?, ?)", 
                lev_insert_batch
            )
            conn.commit()

        t_end = time.perf_counter()
        print(f"Generated and inserted {lev_total_count:,} Phonetic Sound-Alikes in {t_end - t_start:.2f}s.")

        print("\n--- PHASE 3: Database Indexing ---")
        # Build indexes so our subsequent stratified sampling query runs instantly
        cursor.execute("CREATE INDEX idx_adv_type ON adversarial(adversarial_type)")
        cursor.execute("CREATE INDEX idx_adv_distance ON adversarial(distance)")
        conn.commit()
        print("Indexes created successfully.")

    except sqlite3.Error as e:
        print(f"SQLite Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            
    print("\n============================================================")
    print("EXHAUSTIVE ADVERSARIAL DATABASE GENERATION COMPLETE")
    print("============================================================")

if __name__ == "__main__":
    DATABASE_PATH = "./words.db"
    build_exhaustive_adversarial_table(DATABASE_PATH)