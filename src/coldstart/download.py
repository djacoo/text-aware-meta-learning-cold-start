import os
import zipfile
import urllib.request

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"

def download_movielens_1m(output_dir="data/raw"):
    """
    Downloads and extracts the MovieLens-1M dataset into the specified raw data directory.
    
    Args:
        output_dir (str): Relative or absolute path where raw data should be stored.
    """
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "ml-1m.zip")
    extracted_folder = os.path.join(output_dir, "ml-1m")

    if not os.path.exists(extracted_folder):
        print(f"[Data Download] Downloading MovieLens-1M dataset from {MOVIELENS_URL}...")
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        
        print("[Data Download] Extracting archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
            
        os.remove(zip_path)
        print("[Data Download] Download and extraction completed successfully!")
    else:
        print("[Data Download] MovieLens-1M dataset already exists locally. Skipping download.")

if __name__ == "__main__":
    download_movielens_1m()