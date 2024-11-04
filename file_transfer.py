import os
import shutil
import pandas as pd

# Loading the meta data csv
csv_path = 'C:/Users/8897p/OneDrive/Desktop/NLP/Project/AXR/Folder_Data_exploration/metadata_analysis.csv'  # Replace with your CSV file path
metadata_df = pd.read_csv(csv_path)

# source directories for cnn and daily mail
source_dirs = {
    'stories_cnn': 'C:/Users/8897p/OneDrive/Desktop/NLP/Project/brainstorm/DATA/stories_cnn',
    'stories_dail': 'C:/Users/8897p/OneDrive/Desktop/NLP/Project/brainstorm/DATA/stories_dail'
}
target_dir = 'C:/Users/8897p/OneDrive/Desktop/NLP/Project/AXR/Raw_data_corpus'

# checking if the folder exists
os.makedirs(target_dir, exist_ok=True)

# Filter 100 files from each folder with size >= 5 KB and word count >= 800
cnn_files = metadata_df[
    (metadata_df['folder'] == 'stories_cnn') &
    (metadata_df['file_size_kb'] >= 6) &
    (metadata_df['word_count'] >= 900)
].head(100)

dail_files = metadata_df[
    (metadata_df['folder'] == 'stories_dail') &
    (metadata_df['file_size_kb'] >= 6) &
    (metadata_df['word_count'] >= 900)
].head(100)

# data frame concatination
selected_files = pd.concat([cnn_files, dail_files])

# Loop through the selected files and copy each to the target directory
for _, row in selected_files.iterrows():
    folder = row['folder']
    file_name = row['file_name']
    
    # Defining the source and destination paths
    source_path = os.path.join(source_dirs[folder], file_name)
    destination_path = os.path.join(target_dir, file_name)
    
    # Copying the file to the target directory
    try:
        shutil.copy2(source_path, destination_path)
        print(f"Copied {file_name} to {target_dir}")
    except FileNotFoundError:
        print(f"File {file_name} not found in {source_dirs[folder]}")

print("File transfer complete.")