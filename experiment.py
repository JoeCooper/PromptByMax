# Config
batch_size = 6 
primer_length = 4 
rate_by_epoch = [ 1e-4, 1e-5, 1e-6, 5e-7, 1e-7 ]

# Reszta
from os.path import exists
from sys import stderr
if not exists('train.pt'):
    print("train.py not found!", file=stderr)
    exit(1)
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

train = load('train.pt') # input IDs; input mask; output IDs; output mask
ds = TensorDataset(train[0], train[1], train[2], train[3])
dl = DataLoader(ds, batch_size, shuffle=True)
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
device = "cuda"
dtype = torch.bfloat16
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map=device)
pad_embed = model.model.embed_tokens.weight[2, :]
import torch
@torch.no_grad()
def random_embeddings(model: torch.nn.Module, rows: int) -> torch.Tensor:
    E = model.embed_tokens.weight.detach()
    device = E.device
    D = E.size(1)
    mean_vec = E.mean(dim=0, keepdim=True)
    std_vec = E.std(dim=0, unbiased=False, keepdim=True)
    l2_norms = E.norm(dim=1)
    z = torch.randn(rows, D, device=device) 
    synth = mean_vec + std_vec * z
    synth = synth / synth.norm(dim=1, keepdim=True)
    rand_norms = l2_norms[torch.randint(
        0, l2_norms.numel(), (rows,), device=device
    )]
    synth = synth * rand_norms.unsqueeze(1)
    return synth
def embed_fragment(s: str | torch.Tensor) -> torch.Tensor:
    if isinstance(s, str):
        b = tokenizer(s, return_tensors="pt", add_special_tokens=False)
        ids = b['input_ids']
    elif isinstance(s, torch.Tensor) and len(s.shape) == 1:
        ids = s.unsqueeze(0)
    elif isinstance(s, torch.Tensor) and len(s.shape) == 2:
        ids = s
    else:
        raise ValueError(f"Can't embed fragment: {str(s)}")
    ids = ids.to(device)
    embeds = model.get_input_embeddings()(ids)
    return embeds.squeeze(dim=0)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
primer_prefix = embed_fragment('<s>[INST] <<SYS>>\n')
primer_suffix = embed_fragment('\n<</SYS>>\n\n')
input_suffix = embed_fragment('\n[/INST]\n\n')
primer = torch.nn.Parameter(
    random_embeddings(model.model, primer_length).to(device, dtype))
def first_zero_per_row(mask: torch.Tensor) -> torch.Tensor:
    # This function body from LLM
    zeros = (mask == 0).float()
    first = zeros.argmax(dim=1)
    no_zeros = (zeros.sum(dim=1) == 0)
    first[no_zeros] = mask.size(1)
    return first
def left_pad_embeddings(batch: list[torch.Tensor],
                        pad_vector: torch.Tensor) -> tuple[torch.Tensor,
                                                            torch.Tensor]:
    # This function body from LLM 
    B      = len(batch)
    H      = pad_vector.size(0)
    max_L  = max(t.size(0) for t in batch)
    device = pad_vector.device
    dtype  = pad_vector.dtype
    pad_row = pad_vector.expand(max_L, H)
    padded  = pad_row.unsqueeze(0).repeat(B, 1, 1).clone()
    attention = torch.zeros(B, max_L, dtype=torch.long, device=device)
    for i, seq in enumerate(batch):
        L = seq.size(0)
        padded[i, -L:, :]  = seq
        attention[i, -L:]  = 1
    return padded, attention
def recompose(sample_ids, sample_mask) -> tuple[torch.Tensor, torch.Tensor]:
    row_count = sample_ids.shape[0]
    sample_length_by_row = first_zero_per_row(sample_mask)
    input_builder: list[tensor] = []
    for i in range(row_count):
        length = sample_length_by_row[i]
        cropped_ids = sample_ids[i, :length]
        cropped_embeds = embed_fragment(cropped_ids)
        input_ids = torch.cat([
            primer_prefix,
            primer,
            primer_suffix,
            cropped_embeds
            ], dim=0)
        input_builder.append(input_ids)
    return left_pad_embeddings(input_builder, pad_embed)
import torch.nn.functional as F
log_rate = 8
accumulator = 0.0
for epoch, lr in enumerate(rate_by_epoch, 1):
    print(f"epoch = {epoch}")
    optimizer = torch.optim.Adam([primer], lr=lr)
    for step, (sample_ids, sample_mask, output_ids, output_mask) in enumerate(dl):
        optimizer.zero_grad()
        embeds, attention_mask = recompose(sample_ids, sample_mask)
        outputs = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask)
        logits = outputs.logits[:, -1]
        targets = output_ids.squeeze(dim=1).to(device)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        accumulator = accumulator + loss.item()
        if step % log_rate == 0 and step > 0:
            average = accumulator / float(log_rate)
            accumulator = 0.0
            print(f"step = {step}; average_loss = {average}")
train = load('test.pt') # input IDs; input mask; output IDs; output mask
tds = TensorDataset(train[0], train[1], train[2], train[3])
tdl = DataLoader(tds, batch_size, shuffle=True)
total_score = 0
with torch.no_grad():
    for step, (sample_ids, sample_mask, output_ids, output_mask) in enumerate(tdl):
        embeds, attention_mask = recompose(sample_ids, sample_mask)
        outputs = model(
            inputs_embeds=embeds,
            attention_mask=attention_mask)
        logits = outputs.logits[:, -1]
        targets = output_ids.squeeze(dim=1).to(device)
        values, indices = logits.max(dim=1)
        matches = [x[0] == x[1] for x in zip(indices, targets)]
        count = len([x for x in matches if x])
        total_score = total_score + count
        print(f"found = {indices.tolist()}; expected = {targets.tolist()}; score = {count} / {len(matches)}")
print(f"total_score = {total_score} / {len(tds)}")
