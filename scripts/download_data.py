import os
import sys
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = "KhoaDuongDang/nmt-bugfix-data"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    filenames = sys.argv[1:]
    root_dir = Path(__file__).resolve().parent.parent

    for filename in filenames:
        print(f"Downloading {filename} from {REPO_ID}...")
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename=filename, repo_type="dataset", local_dir=root_dir)
            
            if path.endswith(".zip"):
                print(f"Extracting {filename} to {root_dir}...")
                with zipfile.ZipFile(path, "r") as zf:
                    zf.extractall(root_dir)
                os.remove(path)
                
            print(f"Successfully processed {filename}.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print("All tasks completed.")