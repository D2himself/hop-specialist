"""Push the Hop-1 demo to a Hugging Face Space.

Run once, after space/app.py, space/requirements.txt, and space/README.md
exist. Requires a cached HF token (hf auth login). 
"""

from huggingface_hub import HfApi, SpaceHardware

SPACE_REPO = "Omokemi/hop-1-specialist"

api = HfApi()
api.create_repo(
    repo_id=SPACE_REPO,
    repo_type="space",
    private=False,
    space_sdk="gradio",
    space_hardware=SpaceHardware.ZERO_A10G,
    exist_ok=True,
)
api.upload_folder(
    folder_path="space",
    repo_id=SPACE_REPO,
    repo_type="space",
)

print(f"Space pushed to https://huggingface.co/spaces/{SPACE_REPO}")