# ADA Project on LLM Pruning 

## Dependencies

This project requires the following core dependencies:

- `Python`: tested on v3.10.14
- `PyTorch`: tested on v2.4.1 with CUDA 12.2 
- `Transformers`: tested on v4.45.1
- `Datasets`: tested on v3.0.1
- `numpy`: tested on v2.1.1
- `pandas`: tested on v2.2.3
- `huggingface_hub`: tested on v0.25.1
- `wandb`: tested on v0.18.2 (for experiment tracking)

## Usage

### Example for Pruning OPT:

Below is an example command for pruning the OPT-125M model using SparseLLM, to achieve 70% sparsity.

```
python opt_main.py \
    --model facebook/opt-125m \
    --dataset c4 \
    --sparsity 0.7 \
```

We provide a quick overview of the key arguments:

- `--model`: The identifier for the model on the Hugging Face model hub.
- `--dataset`: The dataset to use for evaluation. We support datasets like `c4`, `wikitext2`, and `ptb`.
- `--sparsity`: The desired sparsity level (percentage of weights to be pruned).

### Available Sparsity Methods

We support the following pruning methods for the **OPT** model:

- **Unstructured**: Pruning individual weights without any specific pattern.
- **Semi-Structured N:M Sparsity**: For semi-structured pruning, use the following sparsity types:
  - `--sparsity_type 2:4`: Prune 2 out of every 4 weights.
  - `--sparsity_type 4:8`: Prune 4 out of every 8 weights.

```
python opt_main.py \
    --model facebook/opt-125m \
    --dataset c4 \
    --prunen 2 \
    --prunem 4 \
```
## We tried to do the same for LLaMa but we were not able to go forward with it due to LLaMa being a gated repo for use with HuggingFace.
