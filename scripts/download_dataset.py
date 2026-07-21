from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id='pablovela5620/nerf-synthetic-mirror',
    repo_type='dataset',
    allow_patterns=['lego/*'],
    local_dir='./nerf_data'
)
print(f"Dataset path: {path}")