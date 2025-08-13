from sys import stderr, argv
if len(argv) < 3:
    print("python tokenize.py [input] [output]", file=stderr)
    exit(1)
with open(argv[1], 'r') as f:
    import csv
    r = csv.reader(f)
    rows = list(r)
    if len(argv) == 4:
        limit = int(argv[3])
        if len(rows) > limit:
            print(f"Applying limit of {limit}.")
            rows = rows[:limit]
    stimulii = [row[1] for row in rows]
    response = [row[0] for row in rows]
print(len(stimulii))
import torch
from transformers import AutoTokenizer
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
stimulii_t = tokenizer(stimulii, return_tensors="pt", padding=True, add_special_tokens=False)
response_t = tokenizer(response, return_tensors="pt", add_special_tokens=False)
assert stimulii_t['input_ids'].shape[0] == response_t['input_ids'].shape[0]
bundle = (stimulii_t['input_ids'], stimulii_t['attention_mask'], response_t['input_ids'], response_t['attention_mask'])
with open(argv[2], 'wb') as f:
    torch.save(bundle, f)
