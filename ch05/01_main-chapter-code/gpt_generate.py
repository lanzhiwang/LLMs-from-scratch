# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import argparse
import json
import numpy as np
import os
import urllib.request

# import requests
import tensorflow as tf
import tiktoken
import torch
from tqdm import tqdm

# Import from local files
from previous_chapters import GPTModel


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""


def download_file(url, destination):
    # Send a GET request to download the file
    with urllib.request.urlopen(url) as response:
        # Get the total file size from headers, defaulting to 0 if not present
        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has the same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte

        # Initialize the progress bar with total file size
        progress_bar_description = os.path.basename(url)  # Extract filename from URL
        with tqdm(
            total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description
        ) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Read the file in chunks and write to destination
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    file.write(chunk)
                    progress_bar.update(len(chunk))  # Update progress bar


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T
        )
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T
        )
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T
        )

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b
        )
        gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b
        )

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T,
        )
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"],
        )

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
        )
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
        )
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
        )
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def generate(
    model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None
):

    print(f"generate idx: {idx.shape}")
    print(f"generate max_new_tokens: {max_new_tokens}")
    print(f"generate context_size: {context_size}")
    print(f"generate temperature: {temperature}")
    print(f"generate top_k: {top_k}")
    print(f"generate eos_id: {eos_id}")

    # For-loop is the same as before: Get logits, and only focus on last time step
    for i in range(max_new_tokens):
        print(f"\ngenerate i: {i}")
        print(f"generate idx: {idx.shape}")

        idx_cond = idx[:, -context_size:]
        print(f"generate idx_cond: {idx_cond.shape}")

        with torch.no_grad():
            logits = model(idx_cond)
        print(f"generate logits1: {logits.shape}")

        logits = logits[:, -1, :]
        print(f"generate logits2: {logits.shape}")

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            print(f"generate top_logits: {top_logits.shape}")

            min_val = top_logits[:, -1]
            print(f"generate min_val: {min_val}")

            logits = torch.where(
                logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits
            )
            print(f"generate logits3: {logits.shape}")

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            print(f"generate logits4: {logits.shape}")

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values
            print(f"generate logits5: {logits.max(dim=-1, keepdim=True).values}")
            print(f"generate logits6: {logits.shape}")

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            print(f"generate probs: {probs.shape}")

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        print(f"generate idx_next: {idx_next}")

        if (
            idx_next == eos_id
        ):  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def main(gpt_config, input_prompt, model_size, device):

    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    gpt = GPTModel(gpt_config)
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    gpt.eval()

    tokenizer = tiktoken.get_encoding("gpt2")
    torch.manual_seed(123)

    token_ids = generate(
        model=gpt,
        idx=text_to_token_ids(input_prompt, tokenizer).to(device),
        max_new_tokens=25,
        context_size=gpt_config["context_length"],
        top_k=50,
        temperature=1.0,
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate text with a pretrained GPT-2 model."
    )
    parser.add_argument(
        "--prompt",
        default="Every effort moves you",
        help="Prompt text used to seed the generation (default matches the script's built-in prompt).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for running inference, e.g., cpu, cuda, mps, or auto. Defaults to cpu.",
    )

    args = parser.parse_args()

    torch.manual_seed(123)

    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = args.prompt
    DEVICE = torch.device(args.device)

    print("PyTorch:", torch.__version__)
    print("Device:", DEVICE)

    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    main(BASE_CONFIG, INPUT_PROMPT, model_size, DEVICE)

"""
$ python gpt_generate.py
2025-10-05 07:34:06.612574: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.

PyTorch: 2.8.0+cu128
Device: cpu

checkpoint: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 77.0/77.0 [00:00<00:00, 219kiB/s]
encoder.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.04M/1.04M [00:01<00:00, 661kiB/s]
hparams.json: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 90.0/90.0 [00:00<00:00, 276kiB/s]
model.ckpt.data-00000-of-00001: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 498M/498M [00:56<00:00, 8.89MiB/s]
model.ckpt.index: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5.21k/5.21k [00:00<00:00, 14.9MiB/s]
model.ckpt.meta: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 471k/471k [00:00<00:00, 481kiB/s]
vocab.bpe: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 456k/456k [00:00<00:00, 466kiB/s]

2025-10-05 07:35:19.152399: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:84] Allocation of 154389504 exceeds 10% of free system memory.

generate idx: torch.Size([1, 4])
generate max_new_tokens: 25
generate context_size: 1024
generate temperature: 1.0
generate top_k: 50
generate eos_id: None

generate i: 0
generate idx: torch.Size([1, 4])
generate idx_cond: torch.Size([1, 4])
generate logits1: torch.Size([1, 4, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-136.2895])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[3812]])

generate i: 1
generate idx: torch.Size([1, 5])
generate idx_cond: torch.Size([1, 5])
generate logits1: torch.Size([1, 5, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-87.7108])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[4917]])

generate i: 2
generate idx: torch.Size([1, 6])
generate idx_cond: torch.Size([1, 6])
generate logits1: torch.Size([1, 6, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-88.0116])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[281]])

generate i: 3
generate idx: torch.Size([1, 7])
generate idx_cond: torch.Size([1, 7])
generate logits1: torch.Size([1, 7, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-98.2421])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[7306]])

generate i: 4
generate idx: torch.Size([1, 8])
generate idx_cond: torch.Size([1, 8])
generate logits1: torch.Size([1, 8, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-107.3792])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[1204]])

generate i: 5
generate idx: torch.Size([1, 9])
generate idx_cond: torch.Size([1, 9])
generate logits1: torch.Size([1, 9, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-105.1868])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[13]])

generate i: 6
generate idx: torch.Size([1, 10])
generate idx_cond: torch.Size([1, 10])
generate logits1: torch.Size([1, 10, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-157.9654])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[921]])

generate i: 7
generate idx: torch.Size([1, 11])
generate idx_cond: torch.Size([1, 11])
generate logits1: torch.Size([1, 11, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-154.1304])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[836]])

generate i: 8
generate idx: torch.Size([1, 12])
generate idx_cond: torch.Size([1, 12])
generate logits1: torch.Size([1, 12, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-262.2390])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[470]])

generate i: 9
generate idx: torch.Size([1, 13])
generate idx_cond: torch.Size([1, 13])
generate logits1: torch.Size([1, 13, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-152.5251])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[423]])

generate i: 10
generate idx: torch.Size([1, 14])
generate idx_cond: torch.Size([1, 14])
generate logits1: torch.Size([1, 14, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([13.5658])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[284]])

generate i: 11
generate idx: torch.Size([1, 15])
generate idx_cond: torch.Size([1, 15])
generate logits1: torch.Size([1, 15, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-142.6607])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[2453]])

generate i: 12
generate idx: torch.Size([1, 16])
generate idx_cond: torch.Size([1, 16])
generate logits1: torch.Size([1, 16, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-107.0751])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[534]])

generate i: 13
generate idx: torch.Size([1, 17])
generate idx_cond: torch.Size([1, 17])
generate logits1: torch.Size([1, 17, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-118.9789])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[2761]])

generate i: 14
generate idx: torch.Size([1, 18])
generate idx_cond: torch.Size([1, 18])
generate logits1: torch.Size([1, 18, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-112.0261])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[416]])

generate i: 15
generate idx: torch.Size([1, 19])
generate idx_cond: torch.Size([1, 19])
generate logits1: torch.Size([1, 19, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-124.3666])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[2111]])

generate i: 16
generate idx: torch.Size([1, 20])
generate idx_cond: torch.Size([1, 20])
generate logits1: torch.Size([1, 20, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-47.4673])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[284]])

generate i: 17
generate idx: torch.Size([1, 21])
generate idx_cond: torch.Size([1, 21])
generate logits1: torch.Size([1, 21, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-153.0382])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[21210]])

generate i: 18
generate idx: torch.Size([1, 22])
generate idx_cond: torch.Size([1, 22])
generate logits1: torch.Size([1, 22, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-82.9265])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[606]])

generate i: 19
generate idx: torch.Size([1, 23])
generate idx_cond: torch.Size([1, 23])
generate logits1: torch.Size([1, 23, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-86.9916])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[11]])

generate i: 20
generate idx: torch.Size([1, 24])
generate idx_cond: torch.Size([1, 24])
generate logits1: torch.Size([1, 24, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-113.5787])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[780]])

generate i: 21
generate idx: torch.Size([1, 25])
generate idx_cond: torch.Size([1, 25])
generate logits1: torch.Size([1, 25, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-144.0747])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[326]])

generate i: 22
generate idx: torch.Size([1, 26])
generate idx_cond: torch.Size([1, 26])
generate logits1: torch.Size([1, 26, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-109.2713])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[561]])

generate i: 23
generate idx: torch.Size([1, 27])
generate idx_cond: torch.Size([1, 27])
generate logits1: torch.Size([1, 27, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-133.1093])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[307]])

generate i: 24
generate idx: torch.Size([1, 28])
generate idx_cond: torch.Size([1, 28])
generate logits1: torch.Size([1, 28, 50257])
generate logits2: torch.Size([1, 50257])
generate top_logits: torch.Size([1, 50])
generate min_val: tensor([-118.1486])
generate logits3: torch.Size([1, 50257])
generate logits4: torch.Size([1, 50257])
generate logits5: tensor([[0.]])
generate logits6: torch.Size([1, 50257])
generate probs: torch.Size([1, 50257])
generate idx_next: tensor([[19538]])
Output text:
 Every effort moves you toward finding an ideal life. You don't have to accept your problems by trying to remedy them, because that would be foolish
$
"""
