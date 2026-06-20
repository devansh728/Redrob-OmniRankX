import os
import shutil
from huggingface_hub import hf_hub_download

def main():
    repo_id = "Qwen/Qwen2.5-3B-Instruct-GGUF"
    filename = "qwen2.5-3b-instruct-q4_k_m.gguf"
    print(f"Downloading {filename} from {repo_id}...")
    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename
    )
    local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "qwen2.5-3b-instruct"))
    os.makedirs(local_dir, exist_ok=True)
    target_path = os.path.join(local_dir, "Qwen2.5-3B-Instruct-Q4_K_M.gguf")
    print(f"Copying {cached_path} to {target_path}...")
    shutil.copy2(cached_path, target_path)
    print(f"Model saved to {target_path}")

if __name__ == "__main__":
    main()

