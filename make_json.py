import json

import pandas as pd
import torch
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt


SPACE = "·"
NEWLINE="↩"
TAB = "→"


def make_json(tokens, acts, ft_id, original_model, save_path, top_n = 10):
    """
    End up with
    [
        {
            "neuron_id": "0",
            "snippets": [
            {
                "text": "example text 1",
                "max_activation": 0.95,
                "token_activation_pairs": [
                ["tok1", 0.2],
                ["tok2", 0.95],
                ...
                ]
            },
            ...
            ]
        },
        ...
    ]
    """
    # tokens shape is (n_examples, len_example)
    # acts shape is (n_examples, len_example, n_neurons)
    tokens = tokens.cpu()
    acts = acts.cpu()

    all_neurons = []
    for n in tqdm(range(acts.shape[2])):
        neuron = single_neuron(tokens, acts[:, :, ft_id], n, original_model, top_n=top_n)
        all_neurons.append(neuron)

    with open(save_path, "w") as f:
        json.dump(all_neurons, f, indent=4)
    print(f"Saved to {save_path}")


def single_neuron(tokens: torch.Tensor, acts: torch.Tensor, ft_id, original_model, top_n=10):
    # tokens shape is (n_examples, len_example)
    # acts shape is (n_examples, len_example)

    # get the indexes of examples with the highest max activation
    max_acts = acts.max(dim=1)
    sorted_top_indices = torch.argsort(max_acts.values, descending=True)[:top_n]

    # Process only the top_n activations and corresponding tokens
    snippets = []
    for idx in sorted_top_indices:
        snippet = {}
        example = tokens[idx]
        snippet["text"] = original_model.to_string(example)
        snippet["max_activation"] = float(acts[idx].max())
        snippet["token_activation_pairs"] = [
            [
                original_model.to_string(example[j]).replace(" ", SPACE)
                                                      .replace("\n", NEWLINE + "\n")
                                                      .replace("\t", TAB),
                float(acts[idx, j])
            ]
            for j in range(example.shape[0])
        ]
        snippets.append(snippet)

    res = {
        "neuron_id": str(ft_id),
        "snippets": snippets
    }
    return res


def style_snippet(snippet_idx, act_data):
    tokens_with_activations = act_data['snippets'][snippet_idx]["token_activation_pairs"]
    max_act = act_data['snippets'][snippet_idx]["max_activation"]
    if max_act == 0:
        return ""
    
    # Function to map activation to color
    def activation_to_color(activation):
        if activation < 0:
            return '#FFFFFF'
        normalized_activation = activation / max_act*0.6
        return plt.cm.Reds(normalized_activation)
    
    styled_text = ''.join(f'<span style="background-color: {matplotlib.colors.rgb2hex(activation_to_color(activation))}; margin-right: 0px;">{token}</span>'
                          for token, activation in tokens_with_activations)
    return styled_text


def save_snippets_to_html(feature_id, snippets_data, first_batch, feature_freq, batch_size, n_batches, results_dir):
    filename = f"{results_dir}/neuron_{feature_id}_highlights.html"
    with open(filename, "w") as file:
        file.write(f"<html><head><title>Neuron {feature_id} Highlights</title></head><body>")
        file.write(f"<h1>Neuron {feature_id} Highlights</h1>")
        file.write(f"<div><strong>Number of batches processed:</strong> {n_batches}</div>")
        file.write(f"<div><strong>Batch size:</strong> {batch_size}</div>")
        file.write(f"<div><strong>Feature frequency:</strong> {feature_freq:.6f}</div>")
        file.write("<hr>")
        if feature_freq > 0:

            for snippet_idx, snippet in enumerate(snippets_data['snippets']):
                styled_text = style_snippet(snippet_idx, snippets_data)
                if not styled_text:
                    continue
                snippet_info = (f'<div style="word-wrap: break-word; margin-bottom: 10px;">'
                                f'<strong>Snippet number:</strong> {snippet_idx}<br>'
                                f'<strong>Max activation:</strong> {snippet["max_activation"]:.6f}<br>'
                                f'{styled_text}</div>')
                file.write(snippet_info)

        file.write("<hr>")
            
        if first_batch and feature_freq > 0:
            # Append first batch results
            file.write("<h2>First Batch Results</h2>")
            for snippet_idx, snippet in enumerate(first_batch['snippets']):
                styled_text = style_snippet(snippet_idx, first_batch)
                if not styled_text:
                    continue
                snippet_info = (f'<div style="word-wrap: break-word; margin-bottom: 10px;">'
                                f'<strong>Snippet number (First Batch):</strong> {snippet_idx}<br>'
                                f'<strong>Max activation (First Batch):</strong> {snippet["max_activation"]:.6f}<br>'
                                f'{styled_text}</div>')
                file.write(snippet_info)

        file.write("</body></html>")

