# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

import matplotlib.pyplot as plt
import os
import torch
import urllib.request
import tiktoken


# Import from local files
from previous_chapters import GPTModel, create_dataloader_v1, generate_text_simple


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()


def train_model_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
    tokenizer,
):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # Main training loop
    for epoch in range(num_epochs):
        print(f"\n------------------train_model_simple epoch: {epoch}------------------")

        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            global_step += 1
            print(f"\ntrain_model_simple global_step: {global_step}")

            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            print(f"train_model_simple loss: {loss}")

            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            print(f"train_model_simple tokens_seen: {tokens_seen}")

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"train_model_simple train_losses: {train_losses}")
                print(f"train_model_simple val_losses: {val_losses}")
                print(f"train_model_simple track_tokens_seen: {track_tokens_seen}")

                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        print()
        # Print a sample text after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots()

    # Plot training and validation loss against epochs
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # Create a second x-axis for tokens seen
    ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
    ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # Adjust layout to make room
    # plt.show()


def main(gpt_config, settings):
    print(f"main gpt_config: {gpt_config}")
    print(f"main settings: {settings}")

    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"main device: {device}")

    ##############################
    # Download data if necessary
    ##############################

    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    print(f"main text_data: {text_data[:10]}")
    print(f"main text_data: {len(text_data)}")

    ##############################
    # Initialize model
    ##############################

    model = GPTModel(gpt_config)
    # no assignment model = model.to(device) necessary for nn.Module classes
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )

    ##############################
    # Set up dataloaders
    ##############################

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))

    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    ##############################
    # Train model
    ##############################

    tokenizer = tiktoken.get_encoding("gpt2")

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=settings["num_epochs"],
        eval_freq=5,
        eval_iter=1,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":

    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-key-value bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 10,
        "batch_size": 2,
        "weight_decay": 0.1,
    }

    ###########################
    # Initiate training
    ###########################

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)
    print()
    print(f"train_losses: {train_losses}")
    print(f"val_losses: {val_losses}")
    print(f"tokens_seen: {tokens_seen}")

    ###########################
    # After training
    ###########################

    # Plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", weights_only=True))

"""
$ python gpt_train.py
main gpt_config: {'vocab_size': 50257, 'context_length': 256, 'emb_dim': 768, 'n_heads': 12, 'n_layers': 12, 'drop_rate': 0.1, 'qkv_bias': False}
main settings: {'learning_rate': 0.0005, 'num_epochs': 10, 'batch_size': 2, 'weight_decay': 0.1}
main device: cpu
main text_data: I HAD alwa
main text_data: 20479

------------------train_model_simple epoch: 0------------------

train_model_simple global_step: 0
train_model_simple loss: 10.999968528747559
train_model_simple tokens_seen: 512
train_model_simple train_losses: [9.812252044677734]
train_model_simple val_losses: [9.846261024475098]
train_model_simple track_tokens_seen: [512]
Ep 1 (Step 000000): Train loss 9.812, Val loss 9.846

train_model_simple global_step: 1
train_model_simple loss: 9.759974479675293
train_model_simple tokens_seen: 1024

train_model_simple global_step: 2
train_model_simple loss: 9.241597175598145
train_model_simple tokens_seen: 1536

train_model_simple global_step: 3
train_model_simple loss: 9.1541109085083
train_model_simple tokens_seen: 2048

train_model_simple global_step: 4
train_model_simple loss: 8.801714897155762
train_model_simple tokens_seen: 2560

train_model_simple global_step: 5
train_model_simple loss: 8.402204513549805
train_model_simple tokens_seen: 3072
train_model_simple train_losses: [9.812252044677734, 7.587954521179199]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498]
train_model_simple track_tokens_seen: [512, 3072]
Ep 1 (Step 000005): Train loss 7.588, Val loss 8.044

train_model_simple global_step: 6
train_model_simple loss: 8.355217933654785
train_model_simple tokens_seen: 3584

train_model_simple global_step: 7
train_model_simple loss: 7.666808605194092
train_model_simple tokens_seen: 4096

train_model_simple global_step: 8
train_model_simple loss: 7.857648849487305
train_model_simple tokens_seen: 4608

Every effort moves you,,,,,,,,,,,,.                                     

------------------train_model_simple epoch: 1------------------

train_model_simple global_step: 9
train_model_simple loss: 6.906002998352051
train_model_simple tokens_seen: 5120

train_model_simple global_step: 10
train_model_simple loss: 6.686219692230225
train_model_simple tokens_seen: 5632
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289]
train_model_simple track_tokens_seen: [512, 3072, 5632]
Ep 2 (Step 000010): Train loss 6.582, Val loss 6.800

train_model_simple global_step: 11
train_model_simple loss: 6.496884822845459
train_model_simple tokens_seen: 6144

train_model_simple global_step: 12
train_model_simple loss: 6.448155403137207
train_model_simple tokens_seen: 6656

train_model_simple global_step: 13
train_model_simple loss: 6.106821537017822
train_model_simple tokens_seen: 7168

train_model_simple global_step: 14
train_model_simple loss: 6.228374004364014
train_model_simple tokens_seen: 7680

train_model_simple global_step: 15
train_model_simple loss: 5.938908576965332
train_model_simple tokens_seen: 8192
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192]
Ep 2 (Step 000015): Train loss 5.920, Val loss 6.592

train_model_simple global_step: 16
train_model_simple loss: 6.221467018127441
train_model_simple tokens_seen: 8704

train_model_simple global_step: 17
train_model_simple loss: 6.26823616027832
train_model_simple tokens_seen: 9216

Every effort moves you, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and, and

------------------train_model_simple epoch: 2------------------

train_model_simple global_step: 18
train_model_simple loss: 5.921519756317139
train_model_simple tokens_seen: 9728

train_model_simple global_step: 19
train_model_simple loss: 5.826990127563477
train_model_simple tokens_seen: 10240

train_model_simple global_step: 20
train_model_simple loss: 5.502048969268799
train_model_simple tokens_seen: 10752
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752]
Ep 3 (Step 000020): Train loss 15.767, Val loss 16.154

train_model_simple global_step: 21
train_model_simple loss: 15.673274040222168
train_model_simple tokens_seen: 11264

train_model_simple global_step: 22
train_model_simple loss: 6.041833877563477
train_model_simple tokens_seen: 11776

train_model_simple global_step: 23
train_model_simple loss: 5.571107387542725
train_model_simple tokens_seen: 12288

train_model_simple global_step: 24
train_model_simple loss: 5.651607990264893
train_model_simple tokens_seen: 12800

train_model_simple global_step: 25
train_model_simple loss: 5.343605041503906
train_model_simple tokens_seen: 13312
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752, 13312]
Ep 3 (Step 000025): Train loss 5.494, Val loss 6.384

train_model_simple global_step: 26
train_model_simple loss: 5.8229265213012695
train_model_simple tokens_seen: 13824

Every effort moves you of the                                                

------------------train_model_simple epoch: 3------------------

train_model_simple global_step: 27
train_model_simple loss: 4.864375591278076
train_model_simple tokens_seen: 14336

train_model_simple global_step: 28
train_model_simple loss: 5.252681732177734
train_model_simple tokens_seen: 14848

train_model_simple global_step: 29
train_model_simple loss: 5.221714019775391
train_model_simple tokens_seen: 15360

train_model_simple global_step: 30
train_model_simple loss: 5.66737699508667
train_model_simple tokens_seen: 15872
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652, 4.3731770515441895]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957, 6.327761173248291]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752, 13312, 15872]
Ep 4 (Step 000030): Train loss 4.373, Val loss 6.328

train_model_simple global_step: 31
train_model_simple loss: 4.677751064300537
train_model_simple tokens_seen: 16384

train_model_simple global_step: 32
train_model_simple loss: 5.227112293243408
train_model_simple tokens_seen: 16896

train_model_simple global_step: 33
train_model_simple loss: 5.286459445953369
train_model_simple tokens_seen: 17408

train_model_simple global_step: 34
train_model_simple loss: 4.6352458000183105
train_model_simple tokens_seen: 17920

train_model_simple global_step: 35
train_model_simple loss: 5.193264961242676
train_model_simple tokens_seen: 18432
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652, 4.3731770515441895, 2.823535203933716]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957, 6.327761173248291, 6.258398532867432]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752, 13312, 15872, 18432]
Ep 4 (Step 000035): Train loss 2.824, Val loss 6.258

Every effort moves you know it                                                

------------------train_model_simple epoch: 4------------------

train_model_simple global_step: 36
train_model_simple loss: 4.82653284072876
train_model_simple tokens_seen: 18944

train_model_simple global_step: 37
train_model_simple loss: 3.7299602031707764
train_model_simple tokens_seen: 19456

train_model_simple global_step: 38
train_model_simple loss: 3.8165040016174316
train_model_simple tokens_seen: 19968

train_model_simple global_step: 39
train_model_simple loss: 4.398902416229248
train_model_simple tokens_seen: 20480

train_model_simple global_step: 40
train_model_simple loss: 4.163385391235352
train_model_simple tokens_seen: 20992
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652, 4.3731770515441895, 2.823535203933716, 3.9686191082000732]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957, 6.327761173248291, 6.258398532867432, 6.221040725708008]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752, 13312, 15872, 18432, 20992]
Ep 5 (Step 000040): Train loss 3.969, Val loss 6.221

train_model_simple global_step: 41
train_model_simple loss: 3.4650542736053467
train_model_simple tokens_seen: 21504

train_model_simple global_step: 42
train_model_simple loss: 4.513875484466553
train_model_simple tokens_seen: 22016

train_model_simple global_step: 43
train_model_simple loss: 4.452080249786377
train_model_simple tokens_seen: 22528

train_model_simple global_step: 44
train_model_simple loss: 4.039610385894775
train_model_simple tokens_seen: 23040

Every effort moves you know the "Oh, and in the fact, and I had been him.     "Oh, and in the first I, and I had been the donkey. "Oh, and in the first, and in the

------------------train_model_simple epoch: 5------------------

train_model_simple global_step: 45
train_model_simple loss: 2.948657274246216
train_model_simple tokens_seen: 23552
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652, 4.3731770515441895, 2.823535203933716, 3.9686191082000732, 2.6375129222869873]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957, 6.327761173248291, 6.258398532867432, 6.221040725708008, 6.221099376678467]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752, 13312, 15872, 18432, 20992, 23552]
Ep 6 (Step 000045): Train loss 2.638, Val loss 6.221

train_model_simple global_step: 46
train_model_simple loss: 3.3571271896362305
train_model_simple tokens_seen: 24064

train_model_simple global_step: 47
train_model_simple loss: 3.6566128730773926
train_model_simple tokens_seen: 24576

train_model_simple global_step: 48
train_model_simple loss: 2.930422782897949
train_model_simple tokens_seen: 25088

train_model_simple global_step: 49
train_model_simple loss: 3.5749242305755615
train_model_simple tokens_seen: 25600

train_model_simple global_step: 50
train_model_simple loss: 3.398691415786743
train_model_simple tokens_seen: 26112
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652, 4.3731770515441895, 2.823535203933716, 3.9686191082000732, 2.6375129222869873, 3.1558034420013428]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957, 6.327761173248291, 6.258398532867432, 6.221040725708008, 6.221099376678467, 6.19390344619751]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752, 13312, 15872, 18432, 20992, 23552, 26112]
Ep 6 (Step 000050): Train loss 3.156, Val loss 6.194

train_model_simple global_step: 51
train_model_simple loss: 2.7192423343658447
train_model_simple tokens_seen: 26624

train_model_simple global_step: 52
train_model_simple loss: 3.6439614295959473
train_model_simple tokens_seen: 27136

train_model_simple global_step: 53
train_model_simple loss: 2.944955825805664
train_model_simple tokens_seen: 27648

Every effort moves you know the "I turned, I was not the fact with a little a little: "Yes, and in fact, and I had been the moment--as Jack himself, and in the fact, I had been the picture--I had the

------------------train_model_simple epoch: 6------------------

train_model_simple global_step: 54
train_model_simple loss: 1.5553947687149048
train_model_simple tokens_seen: 28160

train_model_simple global_step: 55
train_model_simple loss: 2.5803592205047607
train_model_simple tokens_seen: 28672
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652, 4.3731770515441895, 2.823535203933716, 3.9686191082000732, 2.6375129222869873, 3.1558034420013428, 2.675602674484253]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957, 6.327761173248291, 6.258398532867432, 6.221040725708008, 6.221099376678467, 6.19390344619751, 6.161670207977295]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752, 13312, 15872, 18432, 20992, 23552, 26112, 28672]
Ep 7 (Step 000055): Train loss 2.676, Val loss 6.162

train_model_simple global_step: 56
train_model_simple loss: 2.7151834964752197
train_model_simple tokens_seen: 29184

train_model_simple global_step: 57
train_model_simple loss: 1.704060673713684
train_model_simple tokens_seen: 29696

train_model_simple global_step: 58
train_model_simple loss: 2.363643169403076
train_model_simple tokens_seen: 30208

train_model_simple global_step: 59
train_model_simple loss: 2.363053798675537
train_model_simple tokens_seen: 30720

train_model_simple global_step: 60
train_model_simple loss: 2.4932701587677
train_model_simple tokens_seen: 31232
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652, 4.3731770515441895, 2.823535203933716, 3.9686191082000732, 2.6375129222869873, 3.1558034420013428, 2.675602674484253, 1.7275314331054688]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957, 6.327761173248291, 6.258398532867432, 6.221040725708008, 6.221099376678467, 6.19390344619751, 6.161670207977295, 6.27439022064209]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752, 13312, 15872, 18432, 20992, 23552, 26112, 28672, 31232]
Ep 7 (Step 000060): Train loss 1.728, Val loss 6.274

train_model_simple global_step: 61
train_model_simple loss: 2.7818965911865234
train_model_simple tokens_seen: 31744

train_model_simple global_step: 62
train_model_simple loss: 2.406967878341675
train_model_simple tokens_seen: 32256

Every effort moves you know," was one of the picture for nothing--I told Mrs.  "I looked--I looked up, I felt to see a smile behind his close that he had been; and as I felt, and down, with a small picture

------------------train_model_simple epoch: 7------------------

train_model_simple global_step: 63
train_model_simple loss: 1.899996280670166
train_model_simple tokens_seen: 32768

train_model_simple global_step: 64
train_model_simple loss: 1.6872522830963135
train_model_simple tokens_seen: 33280

train_model_simple global_step: 65
train_model_simple loss: 1.642266869544983
train_model_simple tokens_seen: 33792
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652, 4.3731770515441895, 2.823535203933716, 3.9686191082000732, 2.6375129222869873, 3.1558034420013428, 2.675602674484253, 1.7275314331054688, 0.9325865507125854]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957, 6.327761173248291, 6.258398532867432, 6.221040725708008, 6.221099376678467, 6.19390344619751, 6.161670207977295, 6.27439022064209, 6.259196758270264]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752, 13312, 15872, 18432, 20992, 23552, 26112, 28672, 31232, 33792]
Ep 8 (Step 000065): Train loss 0.933, Val loss 6.259

train_model_simple global_step: 66
train_model_simple loss: 1.6228649616241455
train_model_simple tokens_seen: 34304

train_model_simple global_step: 67
train_model_simple loss: 1.4192227125167847
train_model_simple tokens_seen: 34816

train_model_simple global_step: 68
train_model_simple loss: 1.6120798587799072
train_model_simple tokens_seen: 35328

train_model_simple global_step: 69
train_model_simple loss: 1.3073221445083618
train_model_simple tokens_seen: 35840

train_model_simple global_step: 70
train_model_simple loss: 1.379565954208374
train_model_simple tokens_seen: 36352
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652, 4.3731770515441895, 2.823535203933716, 3.9686191082000732, 2.6375129222869873, 3.1558034420013428, 2.675602674484253, 1.7275314331054688, 0.9325865507125854, 1.3708552122116089]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957, 6.327761173248291, 6.258398532867432, 6.221040725708008, 6.221099376678467, 6.19390344619751, 6.161670207977295, 6.27439022064209, 6.259196758270264, 6.272136688232422]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752, 13312, 15872, 18432, 20992, 23552, 26112, 28672, 31232, 33792, 36352]
Ep 8 (Step 000070): Train loss 1.371, Val loss 6.272

train_model_simple global_step: 71
train_model_simple loss: 1.6121065616607666
train_model_simple tokens_seen: 36864

Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to look up at the honour being _mine_--because he had always his

------------------train_model_simple epoch: 8------------------

train_model_simple global_step: 72
train_model_simple loss: 1.2026715278625488
train_model_simple tokens_seen: 37376

train_model_simple global_step: 73
train_model_simple loss: 1.073652744293213
train_model_simple tokens_seen: 37888

train_model_simple global_step: 74
train_model_simple loss: 1.1782258749008179
train_model_simple tokens_seen: 38400

train_model_simple global_step: 75
train_model_simple loss: 0.777051568031311
train_model_simple tokens_seen: 38912
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652, 4.3731770515441895, 2.823535203933716, 3.9686191082000732, 2.6375129222869873, 3.1558034420013428, 2.675602674484253, 1.7275314331054688, 0.9325865507125854, 1.3708552122116089, 0.7620922327041626]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957, 6.327761173248291, 6.258398532867432, 6.221040725708008, 6.221099376678467, 6.19390344619751, 6.161670207977295, 6.27439022064209, 6.259196758270264, 6.272136688232422, 6.329254627227783]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752, 13312, 15872, 18432, 20992, 23552, 26112, 28672, 31232, 33792, 36352, 38912]
Ep 9 (Step 000075): Train loss 0.762, Val loss 6.329

train_model_simple global_step: 76
train_model_simple loss: 0.9500246644020081
train_model_simple tokens_seen: 39424

train_model_simple global_step: 77
train_model_simple loss: 0.619135320186615
train_model_simple tokens_seen: 39936

train_model_simple global_step: 78
train_model_simple loss: 1.048189401626587
train_model_simple tokens_seen: 40448

train_model_simple global_step: 79
train_model_simple loss: 0.9421599507331848
train_model_simple tokens_seen: 40960

train_model_simple global_step: 80
train_model_simple loss: 0.9377689957618713
train_model_simple tokens_seen: 41472
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652, 4.3731770515441895, 2.823535203933716, 3.9686191082000732, 2.6375129222869873, 3.1558034420013428, 2.675602674484253, 1.7275314331054688, 0.9325865507125854, 1.3708552122116089, 0.7620922327041626, 0.5632918477058411]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957, 6.327761173248291, 6.258398532867432, 6.221040725708008, 6.221099376678467, 6.19390344619751, 6.161670207977295, 6.27439022064209, 6.259196758270264, 6.272136688232422, 6.329254627227783, 6.461617946624756]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752, 13312, 15872, 18432, 20992, 23552, 26112, 28672, 31232, 33792, 36352, 38912, 41472]
Ep 9 (Step 000080): Train loss 0.563, Val loss 6.462

Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back the window-curtains, I saw that, and down the room, my eyes

------------------train_model_simple epoch: 9------------------

train_model_simple global_step: 81
train_model_simple loss: 0.5822769403457642
train_model_simple tokens_seen: 41984

train_model_simple global_step: 82
train_model_simple loss: 0.49742797017097473
train_model_simple tokens_seen: 42496

train_model_simple global_step: 83
train_model_simple loss: 0.643966019153595
train_model_simple tokens_seen: 43008

train_model_simple global_step: 84
train_model_simple loss: 0.5977292656898499
train_model_simple tokens_seen: 43520

train_model_simple global_step: 85
train_model_simple loss: 0.6415424346923828
train_model_simple tokens_seen: 44032
train_model_simple train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652, 4.3731770515441895, 2.823535203933716, 3.9686191082000732, 2.6375129222869873, 3.1558034420013428, 2.675602674484253, 1.7275314331054688, 0.9325865507125854, 1.3708552122116089, 0.7620922327041626, 0.5632918477058411, 0.35756030678749084]
train_model_simple val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957, 6.327761173248291, 6.258398532867432, 6.221040725708008, 6.221099376678467, 6.19390344619751, 6.161670207977295, 6.27439022064209, 6.259196758270264, 6.272136688232422, 6.329254627227783, 6.461617946624756, 6.542330265045166]
train_model_simple track_tokens_seen: [512, 3072, 5632, 8192, 10752, 13312, 15872, 18432, 20992, 23552, 26112, 28672, 31232, 33792, 36352, 38912, 41472, 44032]
Ep 10 (Step 000085): Train loss 0.358, Val loss 6.542

train_model_simple global_step: 86
train_model_simple loss: 0.4804990291595459
train_model_simple tokens_seen: 44544

train_model_simple global_step: 87
train_model_simple loss: 0.5150243043899536
train_model_simple tokens_seen: 45056

train_model_simple global_step: 88
train_model_simple loss: 0.5719655156135559
train_model_simple tokens_seen: 45568

train_model_simple global_step: 89
train_model_simple loss: 0.5610840320587158
train_model_simple tokens_seen: 46080

Every effort moves you?"  "Yes--quite insensible to the irony. She wanted him vindicated--and by me!"  He laughed again, and threw back his head to look up at the sketch of the donkey. "There were days when I

train_losses: [9.812252044677734, 7.587954521179199, 6.582011699676514, 5.919936180114746, 15.767138481140137, 5.494009971618652, 4.3731770515441895, 2.823535203933716, 3.9686191082000732, 2.6375129222869873, 3.1558034420013428, 2.675602674484253, 1.7275314331054688, 0.9325865507125854, 1.3708552122116089, 0.7620922327041626, 0.5632918477058411, 0.35756030678749084]
val_losses: [9.846261024475098, 8.04432201385498, 6.800333023071289, 6.591868877410889, 16.15375518798828, 6.38377571105957, 6.327761173248291, 6.258398532867432, 6.221040725708008, 6.221099376678467, 6.19390344619751, 6.161670207977295, 6.27439022064209, 6.259196758270264, 6.272136688232422, 6.329254627227783, 6.461617946624756, 6.542330265045166]
tokens_seen: [512, 3072, 5632, 8192, 10752, 13312, 15872, 18432, 20992, 23552, 26112, 28672, 31232, 33792, 36352, 38912, 41472, 44032]
$
$ du -sh model.pth
623M    model.pth
$ du -sh loss.pdf
16K     loss.pdf
$
"""
