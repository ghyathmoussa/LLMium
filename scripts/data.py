from datasets import load_dataset
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()

hf_token = os.environ.get("HF_TOKEN")
login(token=hf_token)
# trust_remote_code=True is required for datasets with custom loading scripts
dataset = load_dataset("oscar-corpus/OSCAR-2301", language="ar", trust_remote_code=True, cache_dir="./data/")

for item in dataset["train"]:
    print(item)
    break