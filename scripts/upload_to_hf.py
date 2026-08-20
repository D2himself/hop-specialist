"""Push the Hop-1 sys-arm model and dataset to the Hugging Face Hub.

Run once after the model folder is downloaded from Colab and unzipped locally,
and after model_card.md / dataset_card.md exist. Requires a cached HF token
(run `huggingface-cli login` once in this terminal before running this script).
"""

from pathlib import Path

from huggingface_hub import HfApi

MODEL_REPO = "Omokemi/hop-1-gemma-270m"
DATASET_REPO = "Omokemi/hop-1-hotpotqa-decomposition"
MODEL_FOLDER = Path("gemma3-270m-hop1-sys")  # unzipped folder from Colab

api = HfApi()

# --- Model repo ---
api.create_repo(repo_id=MODEL_REPO, repo_type="model", private=False, exist_ok=True)
api.upload_folder(
    folder_path=str(MODEL_FOLDER),
    repo_id=MODEL_REPO,
    repo_type="model",
    ignore_patterns=["training_args.bin", "checkpoint-*"],
)
api.upload_file(
    path_or_fileobj="model_card.md",
    path_in_repo="README.md",
    repo_id=MODEL_REPO,
    repo_type="model",
)

# --- Dataset repo ---
api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", private=False, exist_ok=True)
api.upload_file(
    path_or_fileobj="data/train_sys.jsonl",
    path_in_repo="train.jsonl",
    repo_id=DATASET_REPO,
    repo_type="dataset",
)
api.upload_file(
    path_or_fileobj="data/test_sys.jsonl",
    path_in_repo="test.jsonl",
    repo_id=DATASET_REPO,
    repo_type="dataset",
)
api.upload_file(
    path_or_fileobj="dataset_card.md",
    path_in_repo="README.md",
    repo_id=DATASET_REPO,
    repo_type="dataset",
)

print(f"Model pushed to https://huggingface.co/{MODEL_REPO}")
print(f"Dataset pushed to https://huggingface.co/datasets/{DATASET_REPO}")
