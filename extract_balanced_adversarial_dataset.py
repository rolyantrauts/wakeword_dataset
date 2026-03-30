import sqlite3
import csv
import random

def extract_balanced_adversarial_dataset(db_path, output_csv, target_total=20000):
    """
    Performs fast stratified random sampling on the exhaustive adversarial table
    to extract a perfectly balanced dataset, dynamically redistributing quotas 
    if a category is too small.
    """
    print(f"Connecting to database: {db_path}...")
    
    categories = [
        "PAD_Spectral_Mimic",
        "Phonetic_Distance_1",
        "Phonetic_Distance_2",
        "Phonetic_Distance_3"
    ]
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Audit the available population for each category
        population_map = {}
        print("Auditing available combinatorial space...")
        for cat in categories:
            cursor.execute("SELECT id FROM adversarial WHERE adversarial_type = ?", (cat,))
            ids = [row[0] for row in cursor.fetchall()]
            population_map[cat] = ids
            print(f"  -> {cat}: {len(ids):,} total phrases available.")
            
        # 2. Dynamic Quota Allocation
        print("\nCalculating dynamic stratified quotas...")
        base_quota = target_total // len(categories)
        quotas = {cat: base_quota for cat in categories}
        shortfall = 0
        
        # Check for categories that cannot meet the base quota (e.g., Distance 1)
        for cat in categories:
            available = len(population_map[cat])
            if available < base_quota:
                shortfall += (base_quota - available)
                quotas[cat] = available
                
        # Redistribute the shortfall evenly among the categories that have a surplus
        if shortfall > 0:
            surplus_cats = [c for c in categories if len(population_map[c]) > quotas[c]]
            while shortfall > 0 and surplus_cats:
                # Add to the surplus categories one by one
                for cat in surplus_cats:
                    if shortfall > 0 and len(population_map[cat]) > quotas[cat]:
                        quotas[cat] += 1
                        shortfall -= 1
                        
        # 3. Perform the Randomized Extraction
        print("\nExecuting in-memory randomized sampling...")
        selected_ids = []
        for cat in categories:
            target_amount = quotas[cat]
            available_ids = population_map[cat]
            
            sampled = random.sample(available_ids, target_amount)
            selected_ids.extend(sampled)
            print(f"  -> Extracted {target_amount:,} random samples for {cat}.")
            
        # 4. Fetch the actual data for the selected IDs
        print(f"\nFetching exactly {len(selected_ids):,} rows from the database...")
        
        # SQLite limits the number of variables in a query, so we batch the fetch
        chunk_size = 900
        final_dataset = []
        
        for i in range(0, len(selected_ids), chunk_size):
            chunk = selected_ids[i:i + chunk_size]
            placeholders = ",".join("?" * len(chunk))
            cursor.execute(f"""
                SELECT phrase, phonemes, adversarial_type, distance 
                FROM adversarial 
                WHERE id IN ({placeholders})
            """, chunk)
            final_dataset.extend(cursor.fetchall())
            
        # 5. Export to CSV
        print(f"Writing dataset to {output_csv}...")
        with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["phrase", "phonemes", "type", "distance"])
            writer.writerows(final_dataset)
            
        print("\n============================================================")
        print("BALANCED ADVERSARIAL EXTRACTION COMPLETE")
        print("============================================================")

    except sqlite3.Error as e:
        print(f"SQLite Error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    DATABASE_PATH = "./words.db"
    OUTPUT_CSV = "./adversarial_balanced_20k.csv"
    
    extract_balanced_adversarial_dataset(DATABASE_PATH, OUTPUT_CSV, target_total=20000)
