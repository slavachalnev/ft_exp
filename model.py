from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, cfg: Dict):
        super(MLP, self).__init__()

        d_in = cfg.d_in
        d_hidden = d_in * cfg.d_hidden_mult

        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_in, d_hidden)))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_in)))
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        self.b_pre = nn.Parameter(torch.zeros(d_in)) # pre-mlp bias
        self.add_pre_bias = cfg.add_pre_bias

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_in = d_in
        self.d_hidden = d_hidden

        if cfg.per_neuron_coeff:
            # Linearly decreasing coefficients from 2 to 0
            self.per_neuron_coeff = torch.linspace(2, 0, steps=d_hidden, device=cfg.device)
        else:
            self.per_neuron_coeff = torch.ones(d_hidden, device=cfg.device)
        
        self.l1_sqrt = cfg.l1_sqrt
        
        if cfg.act == "relu":
            self.act = nn.ReLU()
        elif cfg.act == "gelu":
            self.act = nn.GELU()
        else:
            raise NotImplementedError
        
        self.renorm_to = cfg.renorm_to
        
        self.to(cfg.device)

    def forward(self, x, y, l1_coeff=0.0):
        activations = self.encode(x)
        x_pred = activations @ self.W_dec + self.b_dec

        # compute losses
        l2_loss = (x_pred.float() - y.float()).pow(2).sum(-1).mean(0)
        positive_activations = F.relu(activations)
        if self.l1_sqrt:
            positive_activations = torch.sqrt(positive_activations)
        positive_activations = positive_activations.sum(0) # sum over batch
        l1_loss = l1_coeff * (positive_activations.float() * self.per_neuron_coeff).sum()
        loss = l2_loss + l1_loss
        return loss, x_pred, activations, l2_loss, l1_loss

    @torch.no_grad()
    def renormalise_decoder(self, leq=False):
        # self.W_dec.data = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        # renormalise decoder weights. If leq=True, then renormalise to be <= 1, else renormalise to be = 1
        norms = torch.norm(self.W_dec, dim=-1, keepdim=True) / self.renorm_to
        if leq:
            mask = (norms > 1).squeeze()
            self.W_dec.data[mask] = self.W_dec[mask] / norms[mask]
        else:
            self.W_dec.data = self.W_dec / norms
        
    def encode(self, x):
        if self.add_pre_bias:
            x_cent = x + self.b_pre
        else:
            x_cent = x
        activations = self.act(x_cent @ self.W_enc + self.b_enc)
        return activations
    
    def predict(self, x):
        activations = self.encode(x)
        x_pred = activations @ self.W_dec + self.b_dec
        return x_pred
    

class AutoEncoder(nn.Module):
    def __init__(
        self, d_hidden: int, l1_coeff: float, d_in: int, dtype=torch.float32, seed=47
    ):
        super().__init__()
        torch.manual_seed(seed)
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_in, d_hidden, dtype=dtype))
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(d_hidden, d_in, dtype=dtype))
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff

    def forward(self, x: torch.Tensor):
        acts = self.encode(x)
        x_reconstruct = acts @ self.W_dec + self.b_dec
        l2_loss = (x_reconstruct - x).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(
            -1, keepdim=True
        ) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj
    
    def encode(self, x: torch.Tensor):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)
        return acts
