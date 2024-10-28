import os
import shutil
import pandas as pd

# Load the CSV file with file metadata
csv_path = 'C:/Users/8897p/OneDrive/Desktop/NLP/Project/AXR/Folder_Data_exploration/metadata_analysis.csv'  # Replace with your CSV file path
metadata_df = pd.read_csv(csv_path)

# Define source directories and the target directory
source_dirs = {
    'stories_cnn': 'C:/Users/8897p/OneDrive/Desktop/NLP/Project/brainstorm/DATA/stories_cnn',
    'stories_dail': 'C:/Users/8897p/OneDrive/Desktop/NLP/Project/brainstorm/DATA/stories_dail'
}
target_dir = 'C:/Users/8897p/OneDrive/Desktop/NLP/Project/AXR/Raw_data_corpus'

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

# Filter 2000 files from each folder with size >= 5 KB and word count >= 800
cnn_files = metadata_df[
    (metadata_df['folder'] == 'stories_cnn') &
    (metadata_df['file_size_kb'] >= 5) &
    (metadata_df['word_count'] >= 800)
].head(2000)

dail_files = metadata_df[
    (metadata_df['folder'] == 'stories_dail') &
    (metadata_df['file_size_kb'] >= 5) &
    (metadata_df['word_count'] >= 800)
].head(2000)

# Combine the filtered data
selected_files = pd.concat([cnn_files, dail_files])

# Loop through the selected files and copy each to the target directory
for _, row in selected_files.iterrows():
    folder = row['folder']
    file_name = row['file_name']
    
    # Define the source and destination paths
    source_path = os.path.join(source_dirs[folder], file_name)
    destination_path = os.path.join(target_dir, file_name)
    
    # Copy the file to the target directory
    try:
        shutil.copy2(source_path, destination_path)
        print(f"Copied {file_name} to {target_dir}")
    except FileNotFoundError:
        print(f"File {file_name} not found in {source_dirs[folder]}")

print("File transfer complete.")