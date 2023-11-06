import json

import pandas as pd
import numpy as np
import torch


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
    # acts shape is (n_examples, len_example)
    tokens = tokens.cpu().numpy()
    acts = acts.cpu().numpy()

    snippets = []
    for i in range(tokens.shape[0]):
        snippet = {}
        example = tokens[i]
        snippet["text"] = original_model.to_string(example)
        snippet["max_activation"] = float(acts[i].max())
        snippet["token_activation_pairs"] = []
        for j in range(tokens.shape[1]):
            snippet["token_activation_pairs"].append(
                [
                    original_model.to_string(example[j]),
                    float(acts[i, j]),
                ]
            )
        snippets.append(snippet)
    
    # sort by max_activation
    snippets = sorted(snippets, key=lambda x: x["max_activation"], reverse=True)

    # keep top_n
    snippets = snippets[:top_n]

    
    res = [{
        "neuron_id": str(ft_id),
        "snippets": snippets,
    }]

    with open(save_path, "w") as f:
        json.dump(res, f, indent=4)
    print(f"Saved to {save_path}")

