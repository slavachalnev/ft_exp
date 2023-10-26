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
        self.token_pointer = 0
        self.first = True
        self.layer_idx = 0

        self.buffer_in = torch.zeros((cfg["buffer_size"], cfg["d_in"]), device=device)
        self.buffer_out = torch.zeros((cfg["buffer_size"], cfg["d_in"]), device=device)

        self.pre_h = None
        self.post_h = None

        def pre_hook(value, hook):
            h = value.detach().clone()
            h = h.reshape(-1, self.cfg["d_in"])
            self.pre_h = h
            return value
        
        def post_hook(value, hook):
            h = value.detach().clone()
            h = h.reshape(-1, self.d_model)
            self.post_h = h
            return value

        self.fwd_hooks = [
            (f"blocks.{self.layer_idx}.ln2.hook_normalized", pre_hook),
            (f"blocks.{self.layer_idx}.hook_mlp_out", post_hook),
            ]

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

                with self.model.hooks(fwd_hooks=self.fwd_hooks):
                    self.model(tokens, stop_at_layer=self.layer_idx+1)

                self.buffer_in[self.pointer: self.pointer+self.pre_h.shape[0]] = self.pre_h
                self.buffer_out[self.pointer: self.pointer+self.post_h.shape[0]] = self.post_h

                self.pointer += self.pre_h.shape[0]
                self.token_pointer += self.cfg["model_batch_size"]

        self.pointer = 0
        # Not sure if re-shuffling is necessary
        perm = torch.randperm(self.buffer_in.shape[0]).cuda()
        self.buffer_in = self.buffer_in[perm]
        self.buffer_out = self.buffer_out[perm]

    @torch.no_grad()
    def next(self):
        res_in = self.buffer_in[self.pointer:self.pointer+self.cfg["batch_size"]]
        res_out = self.buffer_out[self.pointer:self.pointer+self.cfg["batch_size"]]

        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0]//2 - self.cfg["batch_size"]:
            print("Refreshing the buffer")
            self.refresh()

        return res_in, res_out

