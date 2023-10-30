import datetime
import os
import sys
import json
import tqdm

import torch
import torch.nn as nn
import wandb
from transformer_lens import HookedTransformer

from model import MLP
from buffer import Buffer
from utils import get_recons_loss, get_freqs


def train(cfg, model, buffer, save_dir):
    try:
        wandb.init(project="autoencoder", config=cfg)
        num_batches = cfg["num_tokens"] // cfg["batch_size"]
        encoder_optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
        recons_scores = []
        for i in tqdm.trange(num_batches):
            mlp_in, mlp_out = buffer.next()
            loss, x_reconstruct, mid_acts, l2_loss, l1_loss = model(mlp_in, mlp_out)

            loss.backward()
            encoder_optim.step()
            encoder_optim.zero_grad()
            model.renormalise_decoder()

            loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}

            del loss, x_reconstruct, mid_acts, l2_loss, l1_loss, mlp_in, mlp_out

            if (i) % 100 == 0:
                wandb.log(loss_dict)

            if (i) % 1000 == 0:
                x = get_recons_loss(original_model=original_model, local_encoder=model, all_tokens=buffer.all_tokens, cfg=cfg)
                print("Reconstruction:", x)
                recons_scores.append(x[0])
                freqs = get_freqs(original_model=original_model, local_encoder=model, all_tokens=buffer.all_tokens, cfg=cfg, num_batches=5)
                # histogram(freqs.log10(), marginal="box", histnorm="percent", title="Frequencies")
                wandb.log({
                    "recons_score": x[0],
                    "dead": (freqs==0).float().mean().item(),
                    "below_1e-6": (freqs<1e-6).float().mean().item(),
                    "below_1e-5": (freqs<1e-5).float().mean().item(),
                    "num_activated": freqs.sum().item(),
                })

            # if (i) % 30000 == 0:
                # wandb.log({"reset_neurons": 0.0})
                # freqs = get_freqs(50)
                # to_be_reset = (freqs<10**(-5.5))
                # print("Resetting neurons!", to_be_reset.sum())
                # re_init(to_be_reset, model)

            if (i+1) % 100000 == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f"mlp_{i}.pt"))
    finally:
        torch.save(model.state_dict(), os.path.join(save_dir, "mlp_final.pt"))


if __name__ == "__main__":
    default_cfg = {
        "original_model": "gelu-1l",
        # "batch_size": 4096,
        "batch_size": 2048,
        "buffer_mult": 384,
        "num_tokens": int(3e9),
        "seq_len": 128,

        "lr": 1e-4,
        "l1_coeff": 5e-3,
        "beta1": 0.9,
        "beta2": 0.99,

        "d_hidden_mult": 4*8,  # ratio of hidden to input dimension
        "d_in": 512,
        "act": "gelu",
    }

    default_cfg["model_batch_size"] = default_cfg["batch_size"] // default_cfg["seq_len"] * 16
    default_cfg["buffer_size"] = default_cfg["batch_size"] * default_cfg["buffer_mult"]
    default_cfg["buffer_batches"] = default_cfg["buffer_size"] // default_cfg["seq_len"]

    # create workdir
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"mlps/{timestamp}"
    default_cfg["save_dir"] = save_dir

    # save cfg
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/cfg.json", "w") as f:
        json.dump(default_cfg, f)

    original_model = HookedTransformer.from_pretrained(default_cfg["original_model"])

    model = MLP(default_cfg)
    model.to("cuda")

    buffer = Buffer(cfg=default_cfg, model=original_model)

    train(default_cfg, model, buffer, save_dir)


