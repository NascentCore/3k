from transformers import AutoTokenizer

# This meant to download the requested pretrained model from HuggingFace.
# Should be replaced by a git lfs command instead.
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
