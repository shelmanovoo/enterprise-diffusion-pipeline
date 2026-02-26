from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="stabilityai/stable-diffusion-xl-base-1.0",
    resume_download=True
)