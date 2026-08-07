from huggingface_hub import snapshot_download
import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
HF_TOKEN = None

snapshot_download(
    repo_id="wilfredk/raw_dataset",
    repo_type="dataset",
    local_dir="./spectral_data",
    token=HF_TOKEN,
    max_workers=4
    
)