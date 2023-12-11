import datetime
import os
import sys
import json
import tqdm

import torch
import torch.nn as nn
import wandb
from transformer_lens import HookedTransformer
from torch.optim.lr_scheduler import CosineAnnealingLR


from model import MLP
from buffer import Buffer
from utils import get_recons_loss, get_freqs
from config import Config


def train(cfg, model, buffer, save_dir):
    try:
        wandb.init(project="autoencoder", config=cfg)
        num_batches = cfg.num_tokens // cfg.batch_size
        encoder_optim = torch.optim.Adam(model.parameters(),
                                         lr=cfg.lr,
                                         betas=(cfg.beta1, cfg.beta2),
                                         weight_decay=cfg.weight_decay)

        scheduler = CosineAnnealingLR(encoder_optim, T_max=num_batches, eta_min=cfg.lr*0.01)
        recons_scores = []
        for i in tqdm.trange(num_batches):
            mlp_in, mlp_out = buffer.next()

            if cfg.l1_warmup is not None:
                # l1_coeff = cfg.l1_coeff * min(1, (i+1) / cfg.l1_warmup)
                l1_coeff = cfg.l1_coeff * min(1, (i*cfg.batch_size) / cfg.l1_warmup)
            else:
                l1_coeff = cfg.l1_coeff

            loss, x_reconstruct, mid_acts, l2_loss, l1_loss = model(mlp_in, mlp_out, l1_coeff=l1_coeff)

            loss.backward()
            encoder_optim.step()
            encoder_optim.zero_grad()
            scheduler.step()
            model.renormalise_decoder(leq=cfg.leq_renorm)

            loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}

            del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, mlp_in, mlp_out

            if (i) % 200 == 0:
                wandb.log(loss_dict)

            if (i) % 2000 == 0:
                x = get_recons_loss(original_model=original_model, local_encoder=model, all_tokens=buffer.all_tokens, cfg=cfg)
                print("Reconstruction:", x)
                recons_scores.append(x[0])
                freqs = get_freqs(
                    original_model=original_model,
                    local_encoder=model,
                    all_tokens=buffer.all_tokens,
                    batch_size=cfg.model_batch_size,
                    layer_idx=cfg.layer_idx,
                    in_hook=cfg.in_hook,
                    d_in=cfg.d_in,
                    device=cfg.device,
                    num_batches=5,
                )
                wandb.log({
                    "recons_score": x[0],
                    "dead": (freqs==0).float().mean().item(),
                    "below_1e-6": (freqs<1e-6).float().mean().item(),
                    "below_1e-5": (freqs<1e-5).float().mean().item(),
                    "num_activated": freqs.sum().item(),
                    "l1_coeff": l1_coeff,
                })

            if (i+1) % 200000 == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"mlp_{i}.pt"))
    finally:
        torch.save(model.state_dict(), os.path.join(save_dir, "mlp_final.pt"))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "mps"

    # create workdir
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"mlps/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    default_cfg = Config(
        save_dir=save_dir,
        device=device,

        dataset="roneneldan/TinyStories",
        original_model="tiny-stories-2L-33M",
        d_in=1024,
        layer_idx=1,

        add_pre_bias=False,

        act='relu',

        num_tokens=int(4e9),
        d_hidden_mult=4*4,
        l1_coeff=0.0002,
        lr=5e-5,
    )

    default_cfg.to_json("cfg.json")

    original_model = HookedTransformer.from_pretrained(default_cfg.original_model, device=device)
    print(original_model)

    model = MLP(default_cfg)
    model.to(device)

    buffer = Buffer(cfg=default_cfg, model=original_model, device=device)

    train(default_cfg, model, buffer, save_dir)


