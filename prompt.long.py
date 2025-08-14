from os.path import exists
from sys import stderr
if not exists('test.pt'):
    print("test.py not found!", file=stderr)
    exit(1)
import torch
from torch import tensor
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
def load(s: str) -> tuple[tensor, tensor, tensor, tensor]:
    with open(s, 'rb') as f:
        return torch.load(f)
batch_size = 16 
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
device = "cuda"
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device)
pad_id = 2
prompt_prefix = '''<s>[INST] <<SYS>>
Classify the given text as either negative (`1`) or positive (`2`). Write nothing else.

# Example One

## Input

```
Worst sandwich ever!
```

## Output

```
1
```

# Example Two

## Input

```
I love this bar!
```

## Output

```
2
```
<</SYS>>

# Your Task

## Input

```
'''
prompt_suffix = '''
```

## Output

```
'''
import torch
def encode_fragment(s: str) -> torch.Tensor:
    b = tokenizer(s, return_tensors="pt", add_special_tokens=False)
    ids = b['input_ids'].squeeze(dim=0)
    return ids
model.eval()
primer_prefix = encode_fragment(prompt_prefix)
primer_suffix = encode_fragment(prompt_suffix)
def first_zero_per_row(mask: torch.Tensor) -> torch.Tensor:
    # This function body from LLM
    zeros = (mask == 0).float()
    first = zeros.argmax(dim=1)
    no_zeros = (zeros.sum(dim=1) == 0)
    first[no_zeros] = mask.size(1)
    return first
def left_pad_embeddings(batch: list[torch.Tensor]) -> tuple[torch.Tensor,
                                                            torch.Tensor]:
    # This function body from LLM 
    B      = len(batch)
    max_L  = max(t.size(0) for t in batch)
    pad_row = tensor([pad_id], device=device).expand(max_L)
    padded  = pad_row.unsqueeze(0).repeat(B, 1).clone()
    attention = torch.zeros(B, max_L, dtype=torch.long, device=device)
    for i, seq in enumerate(batch):
        L = seq.size(0)
        padded[i, -L:]  = seq
        attention[i, -L:]  = 1
    return padded, attention
def recompose(sample_ids, sample_mask) -> tuple[torch.Tensor, torch.Tensor]:
    row_count = sample_ids.shape[0]
    sample_length_by_row = first_zero_per_row(sample_mask)
    input_builder: list[tensor] = []
    for i in range(row_count):
        length = sample_length_by_row[i]
        cropped_ids = sample_ids[i, :length]
        input_ids = torch.cat([
            primer_prefix,
            cropped_ids,
            primer_suffix
            ], dim=0)
        input_builder.append(input_ids)
    return left_pad_embeddings(input_builder)
log_rate = 8
train = load('test.pt') # input IDs; input mask; output IDs; output mask
tds = TensorDataset(train[0], train[1], train[2], train[3])
tdl = DataLoader(tds, batch_size, shuffle=True)
total_score = 0
accumulator = 0
with torch.no_grad():
    for step, (sample_ids, sample_mask, output_ids, output_mask) in enumerate(tdl):
        ids, attention_mask = recompose(sample_ids, sample_mask)
        outputs = model(
            input_ids=ids,
            attention_mask=attention_mask)
        logits = outputs.logits[:, -1]
        targets = output_ids.squeeze(dim=1).to(device)
        values, indices = logits.max(dim=1)
        matches = [x[0] == x[1] for x in zip(indices, targets)]
        count = len([x for x in matches if x])
        accumulator = accumulator + count
        total_score = total_score + count
        if step % log_rate == 0 and step > 0:
            sample_count = batch_size * log_rate
            print(f"score = {accumulator} / {sample_count}")
            accumulator = 0
print(f"total_score = {total_score} / {len(tds)}")
