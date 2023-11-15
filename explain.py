# Adapted from https://github.com/HoagyC/sparse_coding

import asyncio
import copy
import importlib
import json
import multiprocessing as mp
import os
import pickle
import sys
from datetime import datetime
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import argparse
from dataclasses import dataclass


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformer_lens import HookedTransformer

from tqdm import tqdm

mp.set_start_method("spawn", force=True)

from neuron_explainer.activations.activation_records import \
    calculate_max_activation
from neuron_explainer.activations.activations import (
    ActivationRecord, ActivationRecordSliceParams, NeuronId, NeuronRecord)
from neuron_explainer.explanations.calibrated_simulator import \
    UncalibratedNeuronSimulator
from neuron_explainer.explanations.explainer import \
    TokenActivationPairExplainer
from neuron_explainer.explanations.prompt_builder import PromptFormat
from neuron_explainer.explanations.scoring import (
    aggregate_scored_sequence_simulations, simulate_and_score)
from neuron_explainer.explanations.simulator import ExplanationNeuronSimulator
from neuron_explainer.fast_dataclasses import loads

from model import MLP
from config import Config


EXPLAINER_MODEL_NAME = "gpt-4"  # "gpt-3.5-turbo"
SIMULATOR_MODEL_NAME = "text-davinci-003"

OPENAI_MAX_FRAGMENTS = 50000
OPENAI_FRAGMENT_LEN = 64
OPENAI_EXAMPLES_PER_SPLIT = 5
N_SPLITS = 4
TOTAL_EXAMPLES = OPENAI_EXAMPLES_PER_SPLIT * N_SPLITS
REPLACEMENT_CHAR = "�"
MAX_CONCURRENT = None

DATASET_NAME = "NeelNanda/c4-code-tokenized-2b"


@dataclass
class BaseArgs:
    def parse_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        for key, value in vars(self).items():
            parser.add_argument(f"--{key}", type=type(value), default=None)
        return parser.parse_args()
        
    def __post_init__(self) -> None:
        # parse command line arguments and update the class
        command_line_args = self.parse_args()
        extra_args = set(vars(command_line_args)) - set(vars(self))
        if extra_args: 
            raise ValueError(f"Unknown arguments: {extra_args}")
        self.update(command_line_args)
    
    def update(self, args: Any) -> None:
        for key, value in vars(args).items():
            if value is not None:
                print(f"From command line, setting {key} to {value}")
                setattr(self, key, value)


@dataclass
class InterpArgs(BaseArgs):
    layer: int = 2
    model_name: str = "EleutherAI/pythia-70m-deduped"
    layer_loc: str = "residual"
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    n_feats_explain: int = 10
    # load_interpret_autoencoder: str = ""
    tied_ae: bool = False
    interp_name: str = ""
    sort_mode: str = "max"
    use_decoder: bool = True
    df_n_feats: int = 200
    top_k: int = 50
    save_loc: str = ""


def make_tensor_name(layer: int, layer_loc: str) -> str:
    """Make the tensor name for a given layer and model."""
    assert layer_loc in [
        "residual",
        "mlp",
        "attn",
        "mlpout",
        "mlpin",
    ], f"Layer location {layer_loc} not supported"

    if layer_loc == "residual":
        tensor_name = f"blocks.{layer}.hook_resid_post"
    elif layer_loc == "mlp":
        tensor_name = f"blocks.{layer}.mlp.hook_post"
    elif layer_loc == "attn":
        tensor_name = f"blocks.{layer}.hook_resid_post"
    elif layer_loc == "mlpout":
        tensor_name = f"blocks.{layer}.hook_mlp_out"
    elif layer_loc == "mlpin":
        tensor_name = f"blocks.{layer}.ln2.hook_normalized"

    return tensor_name


def select_active_neurons(model, learned_dict, layer, layer_loc, device="cpu", n_examples=100):
    active_neurons = set()
    
    model.to(device)
    model.eval()
    learned_dict.to(device)

    tensor_name = make_tensor_name(layer, layer_loc)
    sentence_dataset = load_dataset(DATASET_NAME, split="train", streaming=True, )
    iter_dataset = iter(sentence_dataset)

    with torch.no_grad():
        for _ in tqdm(range(n_examples)):
            sentence = next(iter_dataset)
            sentence_tokens = model.to_tokens(sentence["text"], prepend_bos=False).to(device)
            sentence_tokens = sentence_tokens[:, :50] # first 50 tokens
            _, cache = model.run_with_cache(sentence_tokens)
            mlp_activation_data = cache[tensor_name].to(device)
            feature_activation_data = learned_dict.encode(mlp_activation_data)[0]
            # feature_activation_data has shape (n_tokens, n_features)
            for i in range(feature_activation_data.shape[1]):
                if torch.max(feature_activation_data[:, i]) > 0:
                    active_neurons.add(i)
            
            non_zero = []
            for pos in range(sentence_tokens.shape[1]):
                non_zero.append(torch.count_nonzero(feature_activation_data[pos, :]).item())
            print(non_zero)

    
    return list(active_neurons)



def make_feature_activation_dataset(
    model: HookedTransformer,
    learned_dict,
    layer: int,
    layer_loc: str,
    device: str = "cpu",
    n_fragments=OPENAI_MAX_FRAGMENTS,
    max_features: int = 0,  # number of features to store activations for, 0 for all
    random_fragment=True,  # used for debugging
):
    """
    Takes a specified point of a model, and a dataset.
    Returns a dataset which contains the activations of the model at that point,
    for each fragment in the dataset, transformed into the feature space
    """
    model.to(device)
    model.eval()
    learned_dict.to(device)

    if max_features:
        feat_dim = min(max_features, learned_dict.d_hidden)
    else:
        feat_dim = learned_dict.d_hidden

    sentence_dataset = load_dataset(DATASET_NAME, split="train", streaming=True)

    tokenizer_model = model

    tensor_name = make_tensor_name(layer, layer_loc)
    # make list of sentence, tokenization pairs

    iter_dataset = iter(sentence_dataset)

    # Make dataframe with columns for each feature, and rows for each sentence fragment
    # each row should also have the full sentence, the current tokens and the previous tokens

    n_thrown = 0
    n_added = 0
    batch_size = min(20, n_fragments)

    fragment_token_ids_list = []
    fragment_token_strs_list = []

    activation_maxes_table = np.zeros((n_fragments, feat_dim), dtype=np.float16)
    activation_data_table = np.zeros((n_fragments, feat_dim * OPENAI_FRAGMENT_LEN), dtype=np.float16)
    with torch.no_grad():
        while n_added < n_fragments:
            fragments: List[torch.Tensor] = []
            fragment_strs: List[str] = []
            while len(fragments) < batch_size:
                print(
                    f"Added {n_added} fragments, thrown {n_thrown} fragments\t\t\t\t\t\t",
                    end="\r",
                )
                sentence = next(iter_dataset)
                # split the sentence into fragments
                # sentence_tokens = tokenizer_model.to_tokens(sentence["text"], prepend_bos=False).to(device)
                # sentence_tokens = sentence["tokens"]
                sentence_tokens = torch.tensor(sentence["tokens"], dtype=torch.long, device=device)
                sentence_tokens = sentence_tokens.unsqueeze(0)
                n_tokens = sentence_tokens.shape[1]
                # get a random fragment from the sentence - only taking one fragment per sentence so examples aren't correlated]
                if random_fragment:
                    token_start = np.random.randint(0, n_tokens - OPENAI_FRAGMENT_LEN)
                else:
                    token_start = 0
                fragment_tokens = sentence_tokens[:, token_start : token_start + OPENAI_FRAGMENT_LEN]
                token_strs = tokenizer_model.to_str_tokens(fragment_tokens[0])
                if REPLACEMENT_CHAR in token_strs:
                    n_thrown += 1
                    continue

                fragment_strs.append(token_strs)
                fragments.append(fragment_tokens)

            tokens = torch.cat(fragments, dim=0)
            assert tokens.shape == (batch_size, OPENAI_FRAGMENT_LEN), tokens.shape

            _, cache = model.run_with_cache(tokens)
            mlp_activation_data = cache[tensor_name].to(device)

            for i in range(batch_size):
                fragment_tokens = tokens[i : i + 1, :]
                activation_data = mlp_activation_data[i : i + 1, :].squeeze(0)
                token_ids = fragment_tokens[0].tolist()

                feature_activation_data = learned_dict.encode(activation_data)
                feature_activation_data = F.relu(feature_activation_data) # only keep positive activations
                feature_activation_maxes = torch.max(feature_activation_data, dim=0)[0]

                activation_maxes_table[n_added, :] = feature_activation_maxes.cpu().numpy()[:feat_dim]

                feature_activation_data = feature_activation_data.cpu().numpy()[:, :feat_dim]

                activation_data_table[n_added, :] = feature_activation_data.flatten()

                fragment_token_ids_list.append(token_ids)
                fragment_token_strs_list.append(fragment_strs[i])

                n_added += 1

                if n_added >= n_fragments:
                    break

    print(f"Added {n_added} fragments, thrown {n_thrown} fragments")
    # Now we build the dataframe from the numpy arrays and the lists
    print(f"Making dataframe from {n_added} fragments")
    df = pd.DataFrame()
    df["fragment_token_ids"] = fragment_token_ids_list
    df["fragment_token_strs"] = fragment_token_strs_list
    maxes_column_names = [f"feature_{i}_max" for i in range(feat_dim)]
    activations_column_names = [
        f"feature_{i}_activation_{j}" for j in range(OPENAI_FRAGMENT_LEN) for i in range(feat_dim)
    ]  # nested for loops are read left to right

    assert feature_activation_data.shape == (OPENAI_FRAGMENT_LEN, feat_dim)
    df = pd.concat([df, pd.DataFrame(activation_maxes_table, columns=maxes_column_names)], axis=1)
    df = pd.concat(
        [df, pd.DataFrame(activation_data_table, columns=activations_column_names)],
        axis=1,
    )
    print(f"Threw away {n_thrown} fragments, made {len(df)} fragments")
    return df


def get_df(
    feature_dict,
    model_name: str,
    layer: int,
    layer_loc: str,
    n_feats: int,
    save_loc: str,
    device: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    # Load feature dict
    feature_dict.to(device)

    df_loc = os.path.join(save_loc, f"activation_df.hdf")

    reload_data = True
    if os.path.exists(df_loc) and not force_refresh:
        start_time = datetime.now()
        base_df = pd.read_hdf(df_loc)
        print(f"Loaded dataset in {datetime.now() - start_time}")

        # Check that the dataset has enough features saved
        if f"feature_{n_feats - 1}_activation_0" in base_df.keys():
            reload_data = False
        else:
            print("Dataset does not have enough features, remaking")

    if reload_data:
        model = HookedTransformer.from_pretrained(model_name, device=device)

        base_df = make_feature_activation_dataset(
            model,
            learned_dict=feature_dict,
            layer=layer,
            layer_loc=layer_loc,
            device=device,
            max_features=n_feats,
        )
        # save the dataset, saving each column separately so that we can retrive just the columns we want later
        print(f"Saving dataset to {df_loc}")
        os.makedirs(save_loc, exist_ok=True)
        base_df.to_hdf(df_loc, key="df", mode="w")

    # save the autoencoder being investigated
    os.makedirs(save_loc, exist_ok=True)
    torch.save(feature_dict, os.path.join(save_loc, "expander_mlp.pt"))

    return base_df


async def interpret(base_df: pd.DataFrame, save_folder: str, n_feats_to_explain: int) -> None:
    for feat_n in range(0, n_feats_to_explain):
        if os.path.exists(os.path.join(save_folder, f"feature_{feat_n}")):
            print(f"Feature {feat_n} already exists, skipping")
            continue

        activation_col_names = [f"feature_{feat_n}_activation_{i}" for i in range(OPENAI_FRAGMENT_LEN)]
        read_fields = [
            "fragment_token_strs",
            f"feature_{feat_n}_max",
            *activation_col_names,
        ]
        # check that the dataset has the required columns
        if not all([field in base_df.columns for field in read_fields]):
            print(f"Dataset does not have all required columns for feature {feat_n}, skipping")
            continue
        df = base_df[read_fields].copy()
        sorted_df = df.sort_values(by=f"feature_{feat_n}_max", ascending=False)
        sorted_df = sorted_df.head(TOTAL_EXAMPLES)
        top_activation_records = []
        for i, row in sorted_df.iterrows():
            top_activation_records.append(
                ActivationRecord(
                    row["fragment_token_strs"],
                    [row[f"feature_{feat_n}_activation_{j}"] for j in range(OPENAI_FRAGMENT_LEN)],
                )
            )

        random_activation_records: List[ActivationRecord] = []
        # Adding random fragments
        # random_df = df.sample(n=TOTAL_EXAMPLES)
        # for i, row in random_df.iterrows():
        #     random_activation_records.append(ActivationRecord(row["fragment_token_strs"], [row[f"feature_{feat_n}_activation_{j}"] for j in range(OPENAI_FRAGMENT_LEN)]))

        # making sure that the have some variation in each of the features, though need to be careful that this doesn't bias the results
        random_ordering = torch.randperm(len(df)).tolist()
        skip_feature = False
        while len(random_activation_records) < TOTAL_EXAMPLES:
            try:
                i = random_ordering.pop()
            except IndexError:
                skip_feature = True
                break
            # if there are no activations for this fragment, skip it
            if df.iloc[i][f"feature_{feat_n}_max"] == 0:
                continue
            random_activation_records.append(
                ActivationRecord(
                    df.iloc[i]["fragment_token_strs"],
                    [df.iloc[i][f"feature_{feat_n}_activation_{j}"] for j in range(OPENAI_FRAGMENT_LEN)],
                )
            )
        if skip_feature:
            # Add placeholder folder so that we don't try to recompute this feature
            os.makedirs(os.path.join(save_folder, f"feature_{feat_n}"), exist_ok=True)
            print(f"Skipping feature {feat_n} due to lack of activating examples")
            continue

        neuron_id = NeuronId(layer_index=2, neuron_index=feat_n)

        neuron_record = NeuronRecord(
            neuron_id=neuron_id,
            random_sample=random_activation_records,
            most_positive_activation_records=top_activation_records,
        )
        slice_params = ActivationRecordSliceParams(n_examples_per_split=OPENAI_EXAMPLES_PER_SPLIT)
        train_activation_records = neuron_record.train_activation_records(slice_params)
        valid_activation_records = neuron_record.valid_activation_records(slice_params)

        explainer = TokenActivationPairExplainer(
            model_name=EXPLAINER_MODEL_NAME,
            prompt_format=PromptFormat.HARMONY_V4,
            max_concurrent=MAX_CONCURRENT,
        )
        explanations = await explainer.generate_explanations(
            all_activation_records=train_activation_records,
            max_activation=calculate_max_activation(train_activation_records),
            num_samples=1,
        )
        assert len(explanations) == 1
        explanation = explanations[0]
        print(f"Feature {feat_n}, {explanation=}")

        # Simulate and score the explanation.
        format = PromptFormat.HARMONY_V4 if SIMULATOR_MODEL_NAME == "gpt-3.5-turbo" else PromptFormat.INSTRUCTION_FOLLOWING
        simulator = UncalibratedNeuronSimulator(
            ExplanationNeuronSimulator(
                SIMULATOR_MODEL_NAME,
                explanation,
                max_concurrent=MAX_CONCURRENT,
                prompt_format=format,
            )
        )
        scored_simulation = await simulate_and_score(simulator, valid_activation_records)
        score = scored_simulation.get_preferred_score()
        assert len(scored_simulation.scored_sequence_simulations) == 10
        top_only_score = aggregate_scored_sequence_simulations(
            scored_simulation.scored_sequence_simulations[:5]
        ).get_preferred_score()
        random_only_score = aggregate_scored_sequence_simulations(
            scored_simulation.scored_sequence_simulations[5:]
        ).get_preferred_score()
        print(
            f"Feature {feat_n}, score={score:.2f}, top_only_score={top_only_score:.2f}, random_only_score={random_only_score:.2f}"
        )

        feature_name = f"feature_{feat_n}"
        feature_folder = os.path.join(save_folder, feature_name)
        os.makedirs(feature_folder, exist_ok=True)
        pickle.dump(
            scored_simulation,
            open(os.path.join(feature_folder, "scored_simulation.pkl"), "wb"),
        )
        pickle.dump(neuron_record, open(os.path.join(feature_folder, "neuron_record.pkl"), "wb"))
        # write a file with the explanation and the score
        with open(os.path.join(feature_folder, "explanation.txt"), "w") as f:
            f.write(
                f"{explanation}\nScore: {score:.2f}\nExplainer model: {EXPLAINER_MODEL_NAME}\nSimulator model: {SIMULATOR_MODEL_NAME}\n"
            )
            f.write(f"Top only score: {top_only_score:.2f}\n")
            f.write(f"Random only score: {random_only_score:.2f}\n")


def run(dict, cfg: InterpArgs):
    assert cfg.df_n_feats >= cfg.n_feats_explain
    df = get_df(
        feature_dict=dict,
        model_name=cfg.model_name,
        layer=cfg.layer,
        layer_loc=cfg.layer_loc,
        n_feats=cfg.df_n_feats,
        save_loc=cfg.save_loc,
        device=cfg.device,
    )
    asyncio.run(interpret(df, cfg.save_loc, n_feats_to_explain=cfg.n_feats_explain))


def get_score(lines: List[str], mode: str):
    if mode == "top":
        return float(lines[-3].split(" ")[-1])
    elif mode == "random":
        return float(lines[-2].split(" ")[-1])
    elif mode == "top_random":
        score_line = [line for line in lines if "Score: " in line][0]
        return float(score_line.split(" ")[1])
    else:
        raise ValueError(f"Unknown mode: {mode}")


def read_transform_scores(transform_loc: str, score_mode: str, verbose: bool = False) -> Tuple[List[int], List[float]]:
    transform_scores = []
    transform_ndxs = []
    # list all the features by looking for folders
    feat_folders = [x for x in os.listdir(transform_loc) if x.startswith("feature_")]
    if len(feat_folders) == 0:
        return [], []
    
    transform = transform_loc.split('/')[-1]
    print(f"{transform=} {len(feat_folders)=}")
    for feature_folder in feat_folders:
        feature_ndx = int(feature_folder.split("_")[1])
        folder = os.path.join(transform_loc, feature_folder)
        if not os.path.exists(folder):
            continue
        if not os.path.exists(os.path.join(folder, "explanation.txt")):
            continue
        explanation_text = open(os.path.join(folder, "explanation.txt")).read()
        # score should be on the second line but if explanation had newlines could be on the third or below
        # score = float(explanation_text.split("\n")[1].split(" ")[1])
        lines = explanation_text.split("\n")
        score = get_score(lines, score_mode)

        if verbose:
            print(f"{feature_ndx=}, {transform=}, {score=}")
        transform_scores.append(score)
        transform_ndxs.append(feature_ndx)
    
    return transform_ndxs, transform_scores


def read_scores(results_folder: str, score_mode: str = "top") -> Dict[str, Tuple[List[int], List[float]]]:
    assert score_mode in ["top", "random", "top_random"]
    scores: Dict[str, Tuple[List[int], List[float]]] = {}
    transforms = os.listdir(results_folder)
    transforms = [transform for transform in transforms if os.path.isdir(os.path.join(results_folder, transform))]
    if "sparse_coding" in transforms:
        transforms.remove("sparse_coding")
        transforms = ["sparse_coding"] + transforms

    for transform in transforms:
        transform_ndxs, transform_scores = read_transform_scores(os.path.join(results_folder, transform), score_mode)
        if len(transform_ndxs) > 0:
            scores[transform] = (transform_ndxs, transform_scores)

    return scores


def read_results(base_dir: str, activation_name: str, score_mode: str) -> None:
    # results_folder = os.path.join(base_dir, activation_name)
    results_folder = base_dir
    print('results_folder = ', results_folder)

    scores = read_scores(
        results_folder, score_mode
    )  # Dict[str, Tuple[List[int], List[float]]], where the tuple is (feature_ndxs, scores)
    if len(scores) == 0:
        print(f"No scores found for {activation_name}")
        return
    transforms = scores.keys()

    plt.clf()  # clear the plot

    # plot the scores as a violin plot
    colors = [
        "red",
        "blue",
        "green",
        "orange",
        "purple",
        "pink",
        "black",
        "brown",
        "cyan",
        "magenta",
        "grey",
    ]

    # fix yrange from -0.2 to 0.6
    plt.ylim(-0.2, 0.6)
    # add horizontal grid lines every 0.1
    plt.yticks(np.arange(-0.2, 0.6, 0.1))
    plt.grid(axis="y", color="grey", linestyle="-", linewidth=0.5, alpha=0.3)
    # first we need to get the scores into a list of lists
    scores_list = [scores[transform][1] for transform in transforms]
    # remove any transforms that have no scores
    scores_list = [scores for scores in scores_list if len(scores) > 0]
    violin_parts = plt.violinplot(scores_list, showmeans=False, showextrema=False)
    for i, pc in enumerate(violin_parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor(colors[i % len(colors)])
        pc.set_alpha(0.3)

    # add x labels
    plt.xticks(np.arange(1, len(transforms) + 1), transforms, rotation=90)

    # add standard errors around the means but don't plot the means
    cis = [1.96 * np.std(scores[transform][1], ddof=1) / np.sqrt(len(scores[transform][1])) for transform in transforms]
    for i, transform in enumerate(transforms):
        plt.errorbar(
            i + 1,
            np.mean(scores[transform][1]),
            yerr=cis[i],
            fmt="o",
            color=colors[i % len(colors)],
            elinewidth=2,
            capsize=20,
        )

    plt.title(f"{activation_name} {score_mode}")
    plt.xlabel("Transform")
    plt.ylabel("GPT-4-based interpretability score")
    plt.xticks(rotation=90)

    # and a thicker line at 0
    plt.axhline(y=0, linestyle="-", color="black", linewidth=1)

    plt.tight_layout()
    save_path = os.path.join(results_folder, f"{score_mode}_means_and_violin.png")
    print(f"Saving means and violin graph to {save_path}")
    plt.savefig(save_path)


if __name__ == "__main__":
    layer = 0

    # mlp = ReluMLP(d_model, d_model*4*expansion_factor, d_model)
    # mlp_dir = "mlps/2023-10-24_10-11-56"
    # mlp.load_state_dict(torch.load(os.path.join(mlp_dir, f"mlp_8192_layer_{layer}.pt")))

    mlp_dir = "mlps/2023-11-12_23-53-18"
    config = Config.from_json(os.path.join(mlp_dir, "cfg.json"))
    layer_loc = "mlpin"

    model_path = os.path.join(mlp_dir, "mlp_final.pt")
    mlp = MLP(cfg=config)
    mlp.load_state_dict(torch.load(model_path))
    mlp.to('cuda')

    mlp_dir = "mlps/id"
    mlp = Identity(size=d_model*4)
    layer_loc = "mlp"

    n_feats = 20
    save_loc = os.path.join(mlp_dir, "activations")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    force_refresh = False

    cfg = InterpArgs(
        layer=layer,
        model_name=config.original_model,
        layer_loc=layer_loc,
        n_feats_explain=n_feats,
        save_loc=save_loc,
        device=device,
        df_n_feats=n_feats,
    )

    # model = HookedTransformer.from_pretrained(model_name, device=device)
    # active = select_active_neurons(model, mlp, layer, layer_loc, device=device)
    # print(f"len active: {len(active)}")


    # run(mlp, cfg)

    read_results(mlp_dir, "activations", "top_random")




