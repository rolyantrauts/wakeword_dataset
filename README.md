# wakeword_dataset
Create synthetic unknown and adversarial phrases for a wakeword

Run the scripts in this order after editing 

1. adversarial_wordlist.py
Action: Connects to words.db and generates the exhaustive adversarial table containing hundreds of millions of combinatorial phonetic matches.
`python3 adversarial_wordlist.py --wakeword "HEY JARVIS"`
Grab a cuppa :)

2. extract_balanced_adversarial_dataset.py
Action: Connects to words.db, queries the adversarial table, and uses stratified random sampling to pull exactly 20,000 balanced records, saving them to adversarial_balanced_20k.csv.
`python3 extract_balanced_adversarial_dataset.py --output_csv "./my_custom_adversarial.csv" --total 20000`

3. generate_balanced_unknown_dataset.py
Action: Connects to the syllable table in words.db to build structural cadences, loads adversarial_balanced_20k.csv into an exclusion set, and generates 80,000 brand new phrases to save as unknown_balanced_80k.csv
`python3 generate_balanced_unknown_dataset.py --exclude_csv "./my_adversarial.csv" --output_csv "./my_unknown.csv" --total 80000`

For Coqui voice clone you can use https://drive.google.com/file/d/12nUmOvpPITjbu9an98Re0M_hxKN1jxFE/view?usp=sharing  
Or grab your own from the excellent https://accent.gmu.edu/  
