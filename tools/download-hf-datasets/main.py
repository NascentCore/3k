import datasets

dataset = datasets.load_dataset("wikitext","wikitext-2-v1",split="train")
dataset.save_to_disk('wikitext')
