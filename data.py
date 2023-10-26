import os
import torch
import torch.nn as nn
from transformer_lens import HookedTransformer
from datasets import load_dataset
import einops


class Buffer:
    def __init__(self, cfg, model, device='cuda'):
        self.cfg = cfg
        self.model = model
        self.device = device
        self.buffer = torch.zeros((cfg["buffer_size"], cfg["d_in"]), device=device)
        self.token_pointer = 0
        self.first = True

        self.load_data()
        self.refresh()
        print("Buffer initialised")
    
    def load_data(self):
        os.makedirs("data", exist_ok=True)
        cache_path = "data/c4_code_2b_tokens_reshaped.pt"
        dataset_path = "data/c4_code_tokenized_2b.hf"

        if not os.path.exists(cache_path):
            data = load_dataset("NeelNanda/c4-code-tokenized-2b", split="train")
            data.save_to_disk(dataset_path)
            data.set_format(type="torch", columns=["tokens"])
            all_tokens = data["tokens"]
            all_tokens.shape

            all_tokens_reshaped = einops.rearrange(all_tokens, "batch (x seq_len) -> (batch x) seq_len", x=8, seq_len=128)
            all_tokens_reshaped[:, 0] = self.model.tokenizer.bos_token_id
            all_tokens_reshaped = all_tokens_reshaped[torch.randperm(all_tokens_reshaped.shape[0])]
            torch.save(all_tokens_reshaped, cache_path)
        else:
            all_tokens = torch.load(cache_path)
            print("Shuffling the data")
            all_tokens = all_tokens[torch.randperm(all_tokens.shape[0])]

    
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        with torch.autocast("cuda", torch.bfloat16):
            if self.first:
                num_batches = self.cfg["buffer_batches"]
            else:
                num_batches = self.cfg["buffer_batches"]//2
            self.first = False
            for _ in range(0, num_batches, self.cfg["model_batch_size"]):
                tokens = self.all_tokens[self.token_pointer:self.token_pointer+self.cfg["model_batch_size"]]
                _, cache = self.model.run_with_cache(tokens, stop_at_layer=1, names_filter=utils.get_act_name("post", 0))
                mlp_acts = cache[utils.get_act_name("post", 0)].reshape(-1, self.cfg["d_mlp"]) ### TODO: fix this

                self.buffer[self.pointer: self.pointer+mlp_acts.shape[0]] = mlp_acts
                self.pointer += mlp_acts.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]

        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0]).cuda()]

    @torch.no_grad()
    def next(self):
        out = self.buffer[self.pointer:self.pointer+self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0]//2 - self.cfg["batch_size"]:
            # print("Refreshing the buffer!")
            self.refresh()
        return out


