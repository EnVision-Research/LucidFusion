# HF_ENDPOINT=https://hf-mirror.com python download.py

from huggingface_hub import snapshot_download
snapshot_download(repo_id='heye0507/LucidFusion', local_dir='pretrained')