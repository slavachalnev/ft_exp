from dataclasses import dataclass, field
from typing import List, Optional
import os
import json
import datetime


@dataclass
class Config:
    save_dir: str
    device: str = "cuda"
    original_model: str = "gelu-1l"
    batch_size: int = 2048
    buffer_mult: int = 384
    num_tokens: int = int(3e9)
    seq_len: int = 128
    in_hook: str = "ln2.hook_normalized" # "hook_resid_mid"
    out_hook: str = "hook_mlp_out" # "hook_resid_post"
    lr: float = 1e-4
    l1_coeff: float = 0.005
    l1_warmup: Optional[int] = None
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 1e-4
    d_hidden_mult: int = 4*8  # ratio of hidden to input dimension
    d_in: int = 512
    act: str = "gelu"
    leq_renorm: bool = True
    per_neuron_coeff: bool = False
    model_batch_size: int = field(init=False)
    buffer_size: int = field(init=False)
    buffer_batches: int = field(init=False)

    def __post_init__(self):
        # Calculate dependent default values
        self.model_batch_size = self.batch_size // self.seq_len * 16
        self.buffer_size = self.batch_size * self.buffer_mult
        self.buffer_batches = self.buffer_size // self.seq_len
    
    @classmethod
    def from_json(cls, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)
        data.pop("model_batch_size", None)
        data.pop("buffer_size", None)
        data.pop("buffer_batches", None)
        return Config(**data)
    
    def to_json(self, file_name: str):
        assert self.save_dir is not None

        file_path = os.path.join(self.save_dir, file_name)
        with open(file_path, "w") as f:
            json.dump(self.__dict__, f)
        