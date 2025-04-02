import os
import numpy as np
import tiktoken

# Define paths
data_dir = "data/moviescript"
os.makedirs(data_dir, exist_ok=True)
input_file = os.path.join(data_dir, "moviescript.txt")
train_file = os.path.join(data_dir, "train.bin")
val_file = os.path.join(data_dir, "val.bin")

# Load text data
with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

# Use GPT-2 BPE tokenizer
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)

# Train-validation split
n = len(tokens)
n_train = int(n * 0.9)
train_data = np.array(tokens[:n_train], dtype=np.uint16)
val_data = np.array(tokens[n_train:], dtype=np.uint16)

# Save tokenized data
train_data.tofile(train_file)
val_data.tofile(val_file)

print(f"Word-level tokenization complete: {n} tokens processed.")

