# Train a miniature word-level movie scripts model
# Optimized for GTX 1660 Ti (6GB VRAM)

out_dir = 'out-moviescripts-word'
eval_interval = 250
eval_iters    = 200
log_interval  = 10

always_save_checkpoint = False
wandb_log              = False

wandb_project   = 'moviescripts-word'
wandb_run_name  = 'mini-gpt-word'

dataset                = 'moviescripts'
data_dir               = 'data/moviescripts'   # must match your prepare output

gradient_accumulation_steps = 1
batch_size                = 16    # safe start for 6GB VRAM
block_size                = 256   # subword context length

# small GPT architecture
n_layer = 6
n_head  = 6
n_embd  = 384
dropout = 0.2

learning_rate   = 3e-4
max_iters       = 5000
lr_decay_iters  = 5000
min_lr          = 1e-4
beta2           = 0.99
warmup_iters    = 100

device  = 'cuda'    # switch back to 'cpu' if CUDA unavailable
compile = False     # disable torch.compile if Triton isn’t installed
