# evaluate the base gpt2
# n_layer=12, n_heads=12, n_embd=768
# 124M parameters
batch_size = 8
eval_iters = 500 # use more iterations to get good estimate
eval_only = True
wandb_log = True
init_from = 'gpt2'
