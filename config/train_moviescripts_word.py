# Train a miniature word-level movie scripts model
# Optimized for a GTX 1660 Ti (6GB VRAM)
# Start with batch_size = 16; adjust as needed for memory constraints

import os

out_dir = 'out-moviescripts-word'
eval_interval = 250        # Evaluate frequently to monitor overfitting
eval_iters = 200
log_interval = 10          # Log every 10 iterations

# Save checkpoints only when validation improves
always_save_checkpoint = False

wandb_log = False          # Disable wandb logging unless needed
wandb_project = 'moviescripts-word'
wandb_run_name = 'mini-gpt-moviescripts-word'

dataset = 'moviescripts'
data_dir = 'data/moviescripts'  # Ensure this directory contains train.bin and val.bin

# Training parameters tuned for a GTX 1660 Ti
gradient_accumulation_steps = 1  # No gradient accumulation initially
batch_size = 16                  # Safe starting batch size (adjust if needed)
block_size = 256                 # Context length in words (instead of characters)

# Define a small GPT model architecture
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 3e-4           # Moderate learning rate for stability
max_iters = 5000
lr_decay_iters = 5000          # Typically set equal to max_iters
min_lr = 1e-4                  # About one-tenth of the initial learning rate
beta2 = 0.99

warmup_iters = 100

# Device settings
device = 'cuda'                # Use GPU if available; otherwise, set to 'cpu'
compile = False                # Enable torch.compile for potential speedups (if supported)

