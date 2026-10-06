import torch
import numpy as np
import pandas as pd
import os
import json
from sklearn.decomposition import PCA
from functools import partial
from tqdm import tqdm
import transformer_lens.utils as utils

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "data/grep_bias_ir_mechir_format.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA_VALUES = np.arange(-5, 5.25, 0.25).tolist()
SAVE_DIR = "results_steering_aggregation_with_scores"

# ============================================================
# MODEL CONFIGS — final-layer heads only
# ============================================================
MODEL_CONFIGS = {
    "sentence-transformers/msmarco-distilbert-dot-v5": {
        "pool": "mean",
        "sim_func": "dot",
        "heads": [(5, 3), (5, 11)],
        "last_layer": 5,
        "n_heads": 12,
    },
    "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco": {
        "pool": "cls",
        "sim_func": "dot",
        "heads": [(5, 3), (5, 11)],
        "last_layer": 5,
        "n_heads": 12,
    },
    "sentence-transformers/multi-qa-distilbert-dot-v1": {
        "pool": "cls",
        "sim_func": "dot",
        "heads": [(5, 3), (5, 11)],
        "last_layer": 5,
        "n_heads": 12,
    },
    "sentence-transformers/multi-qa-MiniLM-L6-dot-v1": {
        "pool": "cls",
        "sim_func": "dot",
        "heads": [(5, 1), (5, 4), (5, 6), (5, 11)],
        "last_layer": 5,
        "n_heads": 12,
    },
    "sentence-transformers/msmarco-bert-base-dot-v5": {
        "pool": "mean",
        "sim_func": "dot",
        "heads": [(11, 3), (11, 11)],
        "last_layer": 11,
        "n_heads": 12,
    },
}

# ============================================================
# LOAD MODEL
# ============================================================
from mechir import Dot

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)

# Subsample for quick testing
N_SAMPLES = 100
df = df.drop_duplicates(subset="doc_group_id").sample(n=N_SAMPLES, random_state=42)

# ============================================================
# Import shared functions from embedding-level script
# ============================================================
from steering_embed import (
    get_gendered_token_indices,
    summarize_results,
)

# ============================================================
# Collect activation differences at identified heads
# ============================================================
def collect_head_activation_differences(df, tl_model, tokenizer, heads):
    """Extract activations at identified attention heads for gendered token positions
    and compute pairwise differences across all document groups."""

    mf_diffs = []
    gn_diffs = []
    mn_diffs = []
    fn_diffs = []
    all_diffs = []
    doc_group_info = []
    skipped = 0

    layers = sorted(set(layer for layer, _ in heads))
    names_filter = [
        utils.get_act_name("z", layer)
        for layer in layers
    ]

    unique_groups = df.drop_duplicates(subset="doc_group_id")

    for _, row in tqdm(unique_groups.iterrows(), total=len(unique_groups), desc="Collecting head activations"):
        doc_m = row["m_doc"]
        doc_f = row["f_doc"]
        doc_n = row["n_doc"]

        tok_m, tok_f, tok_n, idx_diff = get_gendered_token_indices(
            doc_m, doc_f, doc_n, tokenizer
        )

        if idx_diff is None or len(idx_diff) == 0:
            skipped += 1
            continue

        with torch.no_grad():
            input_m = torch.tensor(tok_m).unsqueeze(0).to(DEVICE)
            input_f = torch.tensor(tok_f).unsqueeze(0).to(DEVICE)
            input_n = torch.tensor(tok_n).unsqueeze(0).to(DEVICE)

            _, cache_m = tl_model.run_with_cache(input_m, names_filter=lambda name: name in names_filter)
            _, cache_f = tl_model.run_with_cache(input_f, names_filter=lambda name: name in names_filter)
            _, cache_n = tl_model.run_with_cache(input_n, names_filter=lambda name: name in names_filter)

        head_acts_m = []
        head_acts_f = []
        head_acts_n = []

        for layer, head_idx in heads:
            hook_name = "_model." + utils.get_act_name("z", layer)
            act_m = cache_m[hook_name][0, idx_diff, head_idx, :].cpu().mean(dim=0)
            act_f = cache_f[hook_name][0, idx_diff, head_idx, :].cpu().mean(dim=0)
            act_n = cache_n[hook_name][0, idx_diff, head_idx, :].cpu().mean(dim=0)

            head_acts_m.append(act_m)
            head_acts_f.append(act_f)
            head_acts_n.append(act_n)

        emb_m_avg = torch.cat(head_acts_m)
        emb_f_avg = torch.cat(head_acts_f)
        emb_n_avg = torch.cat(head_acts_n)

        diff_mf = (emb_m_avg - emb_f_avg).numpy()
        diff_mn = (emb_m_avg - emb_n_avg).numpy()
        diff_fn = (emb_f_avg - emb_n_avg).numpy()

        mf_diffs.append(diff_mf)
        mn_diffs.append(diff_mn)
        fn_diffs.append(diff_fn)
        gn_diffs.append(diff_mn)
        gn_diffs.append(diff_fn)
        all_diffs.append(diff_mf)
        all_diffs.append(diff_mn)
        all_diffs.append(diff_fn)

        doc_group_info.append(row["doc_group_id"])

    print(f"Skipped {skipped} groups due to tokenization length mismatch")
    return {
        "all": np.stack(all_diffs),
        "mf": np.stack(mf_diffs),
        "gn": np.stack(gn_diffs),
        "mn": np.stack(mn_diffs),
        "fn": np.stack(fn_diffs),
    }, doc_group_info


# ============================================================
# Steering hooks — head level
# ============================================================
def aggr_actadd_hook(value, hook, head_idx, steering_vector, alpha):
    """Add a scaled steering vector to a specific head's output.
    value shape: (batch, seq_len, n_heads, d_head)
    """
    value[:, :, head_idx, :] += alpha * steering_vector.unsqueeze(0).unsqueeze(0)
    return value


# ============================================================
# Get steered document embedding — head level
# ============================================================
def get_steered_doc_embedding_heads(model, tl_model, input_ids, attention_mask, heads, directions, alpha, method="actadd"):
    """Run the model with steering hooks on identified heads."""
    fwd_hooks = []

    for layer, head_idx in heads:
        hook_name = utils.get_act_name("z", layer)
        direction = directions[(layer, head_idx)]

        if method == "actadd":
            hook_fn = partial(aggr_actadd_hook, head_idx=head_idx, steering_vector=direction, alpha=alpha)
        fwd_hooks.append((hook_name, hook_fn))

    with torch.no_grad():
        encoder_output = tl_model.run_with_hooks(
            input_ids, fwd_hooks=fwd_hooks, return_type="embeddings",
        )

    pooled = model._pooling(encoder_output, attention_mask)
    if model._normalize:
        from mechir.util import normalize_outputs
        pooled = normalize_outputs(pooled)
    return pooled


# ============================================================
# Evaluate steering — generic
# ============================================================
def evaluate_steering_aggr(df, model, tl_model, tokenizer, alpha_values, steer_fn):
    """Evaluate score differences at different steering strengths.
    
    Args:
        steer_fn: callable(input_ids, attention_mask, alpha) -> pooled embedding
    """

    results = {alpha: {"MvN": [], "FvN": [], "MvF": [], "m_scores": [], "f_scores": [], "n_scores": []} for alpha in alpha_values}
    unique_groups = df.drop_duplicates(subset="doc_group_id")

    for _, row in tqdm(unique_groups.iterrows(), total=len(unique_groups), desc="Evaluating"):
        query = row["query"]
        doc_m = row["m_doc"]
        doc_f = row["f_doc"]
        doc_n = row["n_doc"]

        tok_m, tok_f, tok_n, idx_diff = get_gendered_token_indices(
            doc_m, doc_f, doc_n, tokenizer
        )

        if idx_diff is None or len(idx_diff) == 0:
            continue

        q_encoded = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        q_input_ids = q_encoded["input_ids"].to(DEVICE)
        q_attention_mask = q_encoded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            reps_q = model.forward(q_input_ids, q_attention_mask)

        for alpha in alpha_values:
            scores = {}

            for variant_name, tok_variant in [("m", tok_m), ("f", tok_f), ("n", tok_n)]:
                input_ids = torch.tensor(tok_variant).unsqueeze(0).to(DEVICE)
                attention_mask = torch.ones_like(input_ids).to(DEVICE)

                doc_embed = steer_fn(input_ids, attention_mask, alpha)
                score = model._score_func(reps_q, doc_embed)
                scores[variant_name] = score.item()

            results[alpha]["MvN"].append(scores["m"] - scores["n"])
            results[alpha]["FvN"].append(scores["f"] - scores["n"])
            results[alpha]["MvF"].append(scores["m"] - scores["f"])
            results[alpha]["m_scores"].append(scores["m"])
            results[alpha]["f_scores"].append(scores["f"])
            results[alpha]["n_scores"].append(scores["n"])

    return results


# ============================================================
# Split concatenated PCA direction into per-head directions
# ============================================================
def split_direction_to_heads(direction, heads, d_head):
    """Split a concatenated direction vector back into per-head directions."""
    directions = {}
    for i, (layer, head_idx) in enumerate(heads):
        head_dir = direction[i * d_head : (i + 1) * d_head]
        head_dir = head_dir / head_dir.norm()
        directions[(layer, head_idx)] = head_dir.to(DEVICE)
    return directions

def plot_all_models_grid_aggr(save_dir, model_names, direction_names, alpha_values, steering_level, method, no_steering_alpha=1.0):
    """Plot steering results for all models and directions in a grid."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 9})

    comparisons = ["MvN", "FvN", "MvF"]
    colors = {'MvN': '#fc8d62', 'FvN': '#66c2a5', 'MvF': '#8da0cb'}

    short_names = [name.split("/")[-1] for name in model_names]
    n_models = len(short_names)
    n_directions = len(direction_names)

    fig, axes = plt.subplots(
        nrows=n_models,
        ncols=n_directions,
        figsize=(5 * n_directions, 3 * n_models),
        sharex=True,
        squeeze=False,
    )

    for i, short_name in enumerate(short_names):
        for j, (direction_key, direction_label) in enumerate(direction_names.items()):
            ax = axes[i, j]

            results_path = os.path.join(save_dir, short_name, steering_level, direction_key, method, 'steering_results.json')
            if not os.path.exists(results_path):
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue

            with open(results_path, 'r') as f:
                results = json.load(f)

            for comp in comparisons:
                mean_abs_diffs = []
                valid_alphas = []
                for a in alpha_values:
                    key = str(a)
                    if key in results and comp in results[key]:
                        mean_abs_diffs.append(np.mean(np.abs(results[key][comp])))
                        valid_alphas.append(a)
                ax.plot(valid_alphas, mean_abs_diffs, 'o-', color=colors[comp],
                        linewidth=1.5, markersize=3, label=comp if i == 0 and j == 0 else None)

            ax.axvline(x=no_steering_alpha, color='gray', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if j == 0:
                ax.set_ylabel(short_name, fontsize=9)
            if i == 0:
                ax.set_title(direction_label, fontsize=10)
            if i == n_models - 1:
                ax.set_xlabel('α')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.02), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(save_dir, f'all_models_grid_{steering_level}_{method}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved grid plot to {save_dir}/all_models_grid_{steering_level}_{method}.png")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Save directory: {SAVE_DIR}")

    direction_names = {
        # "all": "All pairs (M-F, M-N, F-N)",
        "mf": "Male-Female only",
        # "gn": "Gendered-Neutral (M-N + F-N)",
        "mn": "Male-Neutral",
        "fn": "Female-Neutral",
    }

    method_configs = {
        "actadd": {"no_steering_alpha": 0.0},
    }

    # Two steering levels: identified heads vs full last layer
    steering_levels = ["heads", "resid"]

    for model_name, model_cfg in MODEL_CONFIGS.items():
        print(f"\n{'#'*60}")
        print(f"MODEL: {model_name}")
        print(f"{'#'*60}")

        model = Dot(model_name, pooling_type=model_cfg["pool"], sim_func_type=model_cfg["sim_func"])
        tl_model = model._model
        tokenizer = model.tokenizer
        heads = model_cfg["heads"]
        last_layer = model_cfg["last_layer"]
        d_head = tl_model.cfg.d_head

        short_name = model_name.split("/")[-1]
        model_save_dir = os.path.join(SAVE_DIR, short_name)
        os.makedirs(model_save_dir, exist_ok=True)

        print(f"Identified heads: {heads}")
        print(f"Last layer: {last_layer}")
        print(f"d_head: {d_head}")

        # Collect or load activations for both levels
        head_diff_sets_path = os.path.join(model_save_dir, "head_diff_sets.pt")
        if not os.path.exists(head_diff_sets_path):
            print("\nCollecting head activation differences...")
            head_diff_sets, _ = collect_head_activation_differences(df, tl_model, tokenizer, heads)
            torch.save({k: torch.tensor(v) for k, v in head_diff_sets.items()}, head_diff_sets_path)
        else:
            print("\nLoading cached head activation differences...")
            head_loaded = torch.load(head_diff_sets_path)
            head_diff_sets = {k: v.numpy() for k, v in head_loaded.items()}

        for steering_level in steering_levels:
            diff_sets = head_diff_sets
            level_label = f"Identified heads {heads}" if steering_level == "heads" else f"Full last layer (L{last_layer})"

            for direction_key, direction_label in direction_names.items():
                diffs = diff_sets[direction_key]

                # Compute mean difference as the steering direction
                mean_diff = torch.tensor(diffs.mean(axis=0), dtype=torch.float32).to(DEVICE)
                mean_diff_normalized = mean_diff / mean_diff.norm()
                print(f"\n{direction_label} mean diff norm: {mean_diff.norm().item():.4f}")

                for method_name, method_cfg in method_configs.items():
                    print(f"\n{'='*60}")
                    print(f"Level: {level_label} | Direction: {direction_label} | Method: {method_name}")
                    print(f"{'='*60}")

                    method_save_dir = os.path.join(model_save_dir, steering_level, direction_key, method_name)
                    os.makedirs(method_save_dir, exist_ok=True)

                    # Build the steering function
                    if steering_level == "heads":
                        if method_name == "projection":
                            directions = split_direction_to_heads(mean_diff_normalized, heads, d_head)
                        else:
                            direction = mean_diff
                            directions = split_direction_to_heads(direction, heads, d_head)

                        steer_fn = lambda input_ids, attention_mask, alpha, _dirs=directions, _method=method_name: \
                            get_steered_doc_embedding_heads(
                                model, tl_model, input_ids, attention_mask,
                                heads, _dirs, alpha, method=_method
                            )

                    print("\nEvaluating steering...")
                    results = evaluate_steering_aggr(
                        df, model, tl_model, tokenizer, ALPHA_VALUES, steer_fn
                    )
                    summarize_results(
                        results, ALPHA_VALUES, save_dir=method_save_dir,
                        no_steering_alpha=method_cfg["no_steering_alpha"]
                    )

                    # Save results
                    serializable_results = {
                        str(alpha): {comp: d for comp, d in comp_data.items()}
                        for alpha, comp_data in results.items()
                    }
                    with open(os.path.join(method_save_dir, 'steering_results.json'), 'w') as f:
                        json.dump(serializable_results, f)

                    # Save direction
                    torch.save(direction.cpu(), os.path.join(method_save_dir, 'direction.pt'))

        # Free memory
        del model, tl_model, tokenizer
        torch.cuda.empty_cache()

    # Plot grids for each level-method combination
    for steering_level in steering_levels:
        for method_name, method_cfg in method_configs.items():
            plot_all_models_grid_aggr(
                SAVE_DIR, list(MODEL_CONFIGS.keys()), direction_names, ALPHA_VALUES,
                steering_level=steering_level, method=method_name,
                no_steering_alpha=method_cfg["no_steering_alpha"]
            )

    print(f"\nAll results saved to {SAVE_DIR}/")