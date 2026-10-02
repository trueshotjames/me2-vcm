import os
import csv

csv_files = [
    'asian_train_data.csv',
    'asian_val_data.csv',
    'train_data.csv',
    'val_data.csv'
]

def check_audio_files():
    total_missing = 0
    # Gets the exact folder where this Python script is saved
    base_folder = os.path.dirname(os.path.abspath(__file__))

    for csv_file in csv_files:
        csv_path = os.path.join(base_folder, csv_file)
        
        if not os.path.exists(csv_path):
            print(f"Skipping {csv_file}: CSV not found.")
            continue

        missing_in_file = 0
        print(f"\nChecking {csv_file}...")

        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file) 
            
            for row_number, row in enumerate(reader, start=2): 
                # Glues the base folder to the audio path from the CSV
                full_audio_path = os.path.join(base_folder, row['filepath'])
                
                if not os.path.exists(full_audio_path):
                    print(f"  Missing: {row['filepath']} (Row {row_number})")
                    missing_in_file += 1
                    total_missing += 1

        if missing_in_file == 0:
            print(f"  All audio files in {csv_file} were found!")
        else:
            print(f"  Total missing in {csv_file}: {missing_in_file}")

    print(f"\nScan complete. Total missing files: {total_missing}")

if __name__ == '__main__':
    check_audio_files()