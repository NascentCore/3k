from huggingface_hub import hf_hub_download
import pandas as pd

REPO_ID = "wikitext"
FILENAME = "data.csv"

hf_hub_download(repo_id=REPO_ID, repo_type="dataset")
