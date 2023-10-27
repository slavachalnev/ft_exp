from transformer_lens import utils as tutils

import tqdm
import torch
from functools import partial

@torch.no_grad()
def get_recons_loss(original_model, local_encoder, all_tokens, cfg, num_batches=5):
    # Warning: layer idx is hardcoded.

    def zero_ablate_hook(mlp_post, hook):
        mlp_post[:] = 0.
        return mlp_post
    
    pre_h = None

    def pre_hook(value, hook):
        nonlocal pre_h
        h = value.detach().clone()
        pre_h = h
        return value
    
    def replacement_hook(mlp_post, hook):
        # have to run pre_hook first
        mlp_post_reconstr = local_encoder(pre_h, torch.zeros_like(pre_h))[1]
        return mlp_post_reconstr

    fwd_hooks = [
        (f"blocks.0.ln2.hook_normalized", pre_hook),
        (f"blocks.0.hook_mlp_out", replacement_hook),
        ]

    loss_list = []
    for i in range(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        loss = original_model(tokens, return_type="loss")
        recons_loss = original_model.run_with_hooks(tokens, return_type="loss", fwd_hooks=fwd_hooks)

        # zero ablation may not be doing what I think it's doing...
        zero_abl_loss = original_model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[(tutils.get_act_name("post", 0), zero_ablate_hook)])

        loss_list.append((loss, recons_loss, zero_abl_loss))
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    print(loss, recons_loss, zero_abl_loss)
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    print(f"{score:.2%}")
    return score, loss, recons_loss, zero_abl_loss


@torch.no_grad()
def get_freqs(original_model, local_encoder, all_tokens, cfg, num_batches=25):
    # Warning: layer idx is hardcoded.
    act_freq_scores = torch.zeros(local_encoder.d_hidden, dtype=torch.float32).cuda()
    total = 0
    for i in tqdm.trange(num_batches):
        tokens = all_tokens[torch.randperm(len(all_tokens))[:cfg["model_batch_size"]]]
        
        _, cache = original_model.run_with_cache(tokens, stop_at_layer=1, names_filter="blocks.0.ln2.hook_normalized")
        mlp_acts = cache["blocks.0.ln2.hook_normalized"]
        mlp_acts = mlp_acts.reshape(-1, cfg["d_in"])

        hidden = local_encoder(mlp_acts, torch.zeros_like(mlp_acts))[2]
        
        act_freq_scores += (hidden > 0).sum(0)
        total+=hidden.shape[0]
    act_freq_scores /= total
    num_dead = (act_freq_scores==0).float().mean()
    print("Num dead", num_dead)
    return act_freq_scores
