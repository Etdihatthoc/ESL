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
    
python Classifier/train_classifier.py \
    --train_path /media/gpus/Data/AES/ESL-Grading/data/PreprocessData/new_full_train_removenoise_aug.csv \
    --val_path /media/gpus/Data/AES/ESL-Grading/data/Full/val_pro.csv \
    --test_path /media/gpus/Data/AES/ESL-Grading/data/Full/test_pro.csv \
    --epochs 20 \
    --batch_size 16 --device cpu