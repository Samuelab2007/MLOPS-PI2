import os
import subprocess
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Get the absolute path to the directory where this script is located
SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define the download directory relative to the script
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

def download_kaggle_data(dataset_slug, save_root=DATA_DIR):
    today_str = datetime.today().strftime('%Y%m%d')
    version_dir = os.path.join(save_root, today_str)

    if os.path.exists(version_dir):
        print(f"📦 Dataset directory {version_dir} already exists.")

    os.makedirs(version_dir, exist_ok=True)

    print("📥 Downloading from Kaggle...")
    os.system(f'kaggle datasets download -p {version_dir} --unzip {dataset_slug}')

    csv_files = [f for f in os.listdir(version_dir) if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV files found after Kaggle download.")

    print(f"✅ Data saved and versioned in {version_dir}")
    return version_dir

def preprocess_data(version_dir):
    df = pd.read_csv(os.path.join(version_dir, "bots_vs_users.csv"))
    
    # Preprocess
    df = df.dropna(subset=df.columns[:1])
    df = df.select_dtypes(include=["number", "object"]).copy()
    df = df.fillna(method='ffill').fillna(method='bfill')

    for col in df.select_dtypes(include=['object']):
        df[col] = LabelEncoder().fit_transform(df[col])

    # Save dataset
    df.to_csv(os.path.join(version_dir, "dataset_preprocessed.csv"))

    # Update "latest" symlink
    save_root = os.path.dirname(version_dir)
    latest_link = os.path.join(save_root, "latest")
    if os.path.islink(latest_link) or os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(version_dir, latest_link)
    

if __name__ == "__main__":
    dataset_slug = "juice0lover/users-vs-bots-classification"
    dataset_dir = download_kaggle_data(dataset_slug)
    print(dataset_dir)
    preprocess_data(dataset_dir)
