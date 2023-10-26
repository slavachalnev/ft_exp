import datetime
import os
import sys
import json
import tqdm

import torch
import torch.nn as nn
import wandb

from model import MLP
from buffer import Buffer


def train(cfg, model, buffer):
    try:
        wandb.init(project="autoencoder", config=cfg)
        num_batches = cfg["num_tokens"] // cfg["batch_size"]
        encoder_optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
        recons_scores = []
        act_freq_scores_list = []
        for i in tqdm.trange(num_batches):
            mlp_in, mlp_out = buffer.next()
            loss, x_reconstruct, mid_acts, l2_loss, l1_loss = model(mlp_in, mlp_out)

            loss.backward()
            encoder_optim.step()
            encoder_optim.zero_grad()
            model.renormalise_decoder()

            loss_dict = {"loss": loss.item(), "l2_loss": l2_loss.item(), "l1_loss": l1_loss.item()}

            if (i) % 100 == 0:
                wandb.log(loss_dict)
                print(loss_dict)

            # TODO: log stuff
            # if (i) % 1000 == 0:
            #     x = (get_recons_loss())
            #     print("Reconstruction:", x)
            #     recons_scores.append(x[0])
            #     freqs = get_freqs(5)
            #     act_freq_scores_list.append(freqs)
            #     # histogram(freqs.log10(), marginal="box", histnorm="percent", title="Frequencies")
            #     wandb.log({
            #         "recons_score": x[0],
            #         "dead": (freqs==0).float().mean().item(),
            #         "below_1e-6": (freqs<1e-6).float().mean().item(),
            #         "below_1e-5": (freqs<1e-5).float().mean().item(),
            #     })

            if (i+1) % 30000 == 0:
                model.save()
                # wandb.log({"reset_neurons": 0.0})
                # freqs = get_freqs(50)
                # to_be_reset = (freqs<10**(-5.5))
                # print("Resetting neurons!", to_be_reset.sum())
                # re_init(to_be_reset, model)
    finally:
        model.save()


if __name__ == "__main__":
    default_cfg = {
        "original_model": "gelu-1l",
        "batch_size": 4096,
        "buffer_mult": 384,
        "num_tokens": int(2e9),
        "seq_len": 128,

        "lr": 1e-4,
        "l1_coeff": 3e-4,
        "beta1": 0.9,
        "beta2": 0.99,

        "d_hidden_mult": 4*8,  # ratio of hidden to input dimension
        "d_in": 512,
        "act": "gelu",
    }

    # create workdir
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = f"mlps/{timestamp}"

    # save cfg
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/cfg.json", "w") as f:
        json.dump(default_cfg, f)


    model = MLP(default_cfg)
    buffer = Buffer(default_cfg)

    train(default_cfg, model, buffer)


