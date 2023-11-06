import json

import pandas as pd
import torch
from tqdm import tqdm


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
        neuron = single_neuron(tokens, acts[:, :, n], n, original_model, top_n=top_n)
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


