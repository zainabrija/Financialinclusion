import kagglehub
import pandas as pd
import os

def download_and_process_dataset():
    # Download the dataset
    print("Downloading dataset...")
    path = kagglehub.dataset_download("annalie/findex-world-bank")
    print(f"Dataset downloaded to: {path}")
    
    # Create a data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # The dataset files will be in the downloaded path
    # You can process the data here as needed
    print("Dataset download complete. You can now process the data in the 'data' directory.")

if __name__ == "__main__":
    download_and_process_dataset() 