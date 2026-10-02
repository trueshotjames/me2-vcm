import pandas as pd
import soundfile as sf
import os

def add_duration_column(csv_filename):
    if not os.path.exists(csv_filename):
        print(f"File not found: {csv_filename}")
        return

    df = pd.read_csv(csv_filename)
    
    if df.empty:
        print(f"{csv_filename} is empty.")
        return

    # Dynamically grab the name of the 1st column
    filepath_col = df.columns[0]
    
    durations = []
    for filepath in df[filepath_col]:
        try:
            # sf.info() reads only the file metadata, making it much faster than loading the audio array
            info = sf.info(filepath)
            durations.append(info.duration)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            durations.append(None)
            
    # Add the new column
    df['duration'] = durations
    
    # Overwrite the original CSV file
    df.to_csv(csv_filename, index=False)
    print(f"Successfully added duration column to {csv_filename}")

# Process both files
add_duration_column('wake_train_data.csv')
add_duration_column('wake_val_data.csv')