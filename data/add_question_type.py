import pandas as pd
import os

# List of CSV files to process
csv_files = ['/media/gpus/Data/AES/ESL-Grading/data/Full/Full_train.csv']

for csv_file in csv_files:
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract question_type from audio_path
    def extract_question_type(path):
        # Get the filename
        filename = os.path.basename(path)
        # Split by '-' and then by '.' to get the number before .ogg
        try:
            return int(filename.split('-')[-1].split('.')[0])
        except Exception:
            return None

    df['question_type'] = df['audio_path'].apply(extract_question_type)
    
    # Save the updated CSV
    df.to_csv(csv_file, index=False)