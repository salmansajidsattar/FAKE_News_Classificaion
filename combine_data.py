import pandas as pd
import os
import glob

def combine_csv_files(folder_path, output_file):
    # Get all CSV files in the specified folder
    all_files = os.listdir(folder_path)
    
    # List to hold each DataFrame with the source column
    data_frames = []
    combined_df=None
    import data_alter as da
    print(all_files)
    for file in all_files:
        # Read each CSV file
        df = pd.read_csv(folder_path+'/'+file)
        
        # Add a new column with the filename as the source
        df['source'] = os.path.basename(str(file.split('.')[0]))
        df['label'] = 1
        da.Main(df)
        # Append the DataFrame to the list
        data_frames.append(df)
    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(data_frames, ignore_index=True)
    
    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved as: {output_file}")

# Usage
folder_path = "./News_Data/"
output_file = "combined_News_output.csv"
combine_csv_files(folder_path, output_file)
