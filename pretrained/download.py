# HF_ENDPOINT=https://hf-mirror.com python download.py

from huggingface_hub import snapshot_download
snapshot_download(repo_id='stabilityai/stable-diffusion-2-1-base', local_dir='pretrained/sd-2-1-base')