from transformer_lens import utils as tutils

import tqdm
import torch
from functools import partial
from model import MLP

@torch.no_grad()
def get_recons_loss(original_model, local_encoder, all_tokens, cfg, num_batches=5):

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
        mlp_post_reconstr = local_encoder.predict(pre_h)
        return mlp_post_reconstr

    fwd_hooks = [
        (f"blocks.{cfg.layer_idx}.{cfg.in_hook}", pre_hook),
        (f"blocks.{cfg.layer_idx}.{cfg.out_hook}", replacement_hook),
        ]

    loss_list = []
    for i in range(num_batches):
        tokens = all_tokens[i*cfg.model_batch_size:(i+1)*cfg.model_batch_size]
        loss = original_model(tokens, return_type="loss")
        recons_loss = original_model.run_with_hooks(tokens, return_type="loss", fwd_hooks=fwd_hooks)

        # zero ablation may not be doing what I think it's doing...
        zero_abl_loss = original_model.run_with_hooks(tokens, return_type="loss", fwd_hooks=[
            (tutils.get_act_name("post", cfg.layer_idx), zero_ablate_hook),
            ])

        loss_list.append((loss, recons_loss, zero_abl_loss))
        pre_h = None
        
    losses = torch.tensor(loss_list)
    loss, recons_loss, zero_abl_loss = losses.mean(0).tolist()

    del losses, loss_list
    torch.cuda.empty_cache()

    # print(loss, recons_loss, zero_abl_loss)
    score = ((zero_abl_loss - recons_loss)/(zero_abl_loss - loss))
    # print(f"{score:.2%}")
    return score, loss, recons_loss, zero_abl_loss


@torch.no_grad()
def get_freqs(original_model, local_encoder, all_tokens, cfg, num_batches=25):
    act_freq_scores = torch.zeros(local_encoder.d_hidden, dtype=torch.float32).to(cfg.device)
    total = 0
    for i in tqdm.trange(num_batches):
        tokens = all_tokens[i*cfg.model_batch_size:(i+1)*cfg.model_batch_size]
        
        _, cache = original_model.run_with_cache(tokens, stop_at_layer=cfg.layer_idx + 1,
                                                 names_filter=f"blocks.{cfg.layer_idx}.{cfg.in_hook}")
        mlp_acts = cache[f"blocks.{cfg.layer_idx}.{cfg.in_hook}"]
        mlp_acts = mlp_acts.reshape(-1, cfg.d_in)

        hidden = local_encoder.encode(mlp_acts)
        
        act_freq_scores += (hidden > 0).sum(0)
        total+=hidden.shape[0]

    del mlp_acts, hidden, cache
    torch.cuda.empty_cache()

    act_freq_scores /= total
    return act_freq_scores


@torch.no_grad()
def find_similar_decoder_weights(model: MLP, feature: int):
    # sort decoder weights by cosine similarity to feature
    feature_weight = model.W_dec[feature]
    similarities = torch.cosine_similarity(model.W_dec, feature_weight.unsqueeze(0), dim=-1)
    similarities, indices = torch.sort(similarities, descending=True)
    return similarities, indices


@torch.no_grad()
def find_similar_encoder_weights(model: MLP, feature: int):
    # sort encoder weights by cosine similarity to feature
    enc_weights = model.W_enc.T
    feature_weight = enc_weights[feature]
    similarities = torch.cosine_similarity(enc_weights, feature_weight.unsqueeze(0), dim=-1)
    similarities, indices = torch.sort(similarities, descending=True)
    return similarities, indices

