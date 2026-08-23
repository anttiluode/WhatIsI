from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "microsoft/Phi-3-mini-4k-instruct-gguf"
FILE = "Phi-3-mini-4k-instruct-q4.gguf"

path = hf_hub_download(repo_id=REPO, filename=FILE, local_dir="models")
print(Path(path).resolve())
