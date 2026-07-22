import os
from huggingface_hub import snapshot_download

def download_model():
    repo_id = "facebook/wav2vec2-xls-r-300m"
    local_dir = os.path.abspath("pretrained/wav2vec2-xls-r-300m")
    
    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading checkpoint: {repo_id}...")
    print(f"Saving locally to: {local_dir}")
    
    # Download files using huggingface_hub snapshot_download
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]  # Skip unnecessary formats
    )
    print("Download completed successfully.")

if __name__ == "__main__":
    download_model()
