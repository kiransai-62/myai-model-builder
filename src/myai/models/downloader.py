from pathlib import Path
from ..core.console import print_info, print_error

def download_model(repo_id: str, dest_dir: Path) -> bool:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print_error("huggingface_hub is not installed. Run: pip install huggingface_hub")
        return False
        
    print_info(f"Downloading {repo_id} to {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download safetensors and configs. Ignore old .bin/.pt formats to save space.
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dest_dir),
            ignore_patterns=["*.pt", "*.bin", "*.h5", "*.ot", "*.msgpack"]
        )
        return True
    except Exception as e:
        print_error(f"Download failed: {e}")
        return False