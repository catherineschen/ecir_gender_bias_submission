import torch
import numpy as np
import pandas as pd
import os
import json
from sklearn.decomposition import PCA
from functools import partial
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
DATA_PATH = "data/grep_bias_ir_mechir_format.csv"
MODEL_NAME = "sentence-transformers/msmarco-distilbert-dot-v5"  # mean-pooled, dot product
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA_VALUES = np.arange(-5, 5.25, 0.25).tolist()
SAVE_DIR = "results_steering_embed_with_scores"  # change to your preferred path

# ============================================================
# LOAD MODEL
# ============================================================
from mechir import Dot

MODEL_NAMES = {
    "sentence-transformers/msmarco-distilbert-dot-v5": {
        "pool": "mean",
        "sim_func": "dot",
    },
    "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco": {
        "pool": "cls",
        "sim_func": "dot",
    },
    "sentence-transformers/multi-qa-distilbert-dot-v1": {
        "pool": "cls",
        "sim_func": "dot",
    },
    "sentence-transformers/multi-qa-MiniLM-L6-dot-v1": {
        "pool": "cls",
        "sim_func": "dot",
    },
    "sentence-transformers/msmarco-bert-base-dot-v5": {
        "pool": "mean",
        "sim_func": "dot",
    },
}

def load_bi(model_name_or_path):
    pool = MODEL_NAMES[model_name_or_path]["pool"]
    sim_func = MODEL_NAMES[model_name_or_path]["sim_func"]
    return Dot(model_name_or_path, pooling_type=pool, sim_func_type=sim_func)

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(DATA_PATH)

# Subsample for quick testing
N_SAMPLES = 100  # adjust as needed
df = df.drop_duplicates(subset="doc_group_id").sample(n=N_SAMPLES, random_state=42)


# ============================================================
# STEP 1: Find gendered token indices
# ============================================================
def get_gendered_token_indices(doc_m, doc_f, doc_n, tokenizer):
    """Find token positions that differ across the three document variants."""
    tok_m = np.array(tokenizer.encode(doc_m))
    tok_f = np.array(tokenizer.encode(doc_f))
    tok_n = np.array(tokenizer.encode(doc_n))

    if not (tok_m.shape == tok_f.shape == tok_n.shape):
        return None, None, None, None

    diff_mf = tok_m != tok_f
    diff_mn = tok_m != tok_n
    diff_fn = tok_f != tok_n
    idx_diff = np.where(diff_mf | diff_mn | diff_fn)[0]

    return tok_m, tok_f, tok_n, idx_diff


# ============================================================
# STEP 2: Collect embedding differences for PCA
# ============================================================
def collect_embedding_differences(df, tl_model, tokenizer):
    """Extract embeddings at gendered token positions and compute
    pairwise differences across all document groups.
    
    Returns a dict with three sets of differences:
        - 'all': all three pairwise differences (M-F, M-N, F-N)
        - 'mf': male-female differences only
        - 'gn': gendered-neutral differences (M-N and F-N)
    """

    mf_diffs = []
    gn_diffs = []
    mn_diffs = []
    fn_diffs = []
    all_diffs = []
    doc_group_info = []
    skipped = 0

    unique_groups = df.drop_duplicates(subset="doc_group_id")

    for _, row in tqdm(unique_groups.iterrows(), total=len(unique_groups), desc="Collecting embeddings"):
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

            _, cache_m = tl_model.run_with_cache(input_m, names_filter="hook_full_embed")
            _, cache_f = tl_model.run_with_cache(input_f, names_filter="hook_full_embed")
            _, cache_n = tl_model.run_with_cache(input_n, names_filter="hook_full_embed")

            emb_m = cache_m["_model.hook_full_embed"][0, idx_diff, :].cpu()
            emb_f = cache_f["_model.hook_full_embed"][0, idx_diff, :].cpu()
            emb_n = cache_n["_model.hook_full_embed"][0, idx_diff, :].cpu()

        emb_m_avg = emb_m.mean(dim=0)
        emb_f_avg = emb_f.mean(dim=0)
        emb_n_avg = emb_n.mean(dim=0)

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
# STEP 3: Compute steering direction
# ============================================================
def compute_actadd_vectors(diff_sets):
    """Compute mean difference vectors for ActAdd steering."""
    vectors = {}
    for key, diffs in diff_sets.items():
        mean_diff = torch.tensor(diffs.mean(axis=0), dtype=torch.float32).to(DEVICE)
        vectors[key] = mean_diff
        print(f"ActAdd vector '{key}': norm = {mean_diff.norm().item():.4f}")
    return vectors



# ============================================================
# STEP 4: Steering hooks
# ============================================================
def steering_hook(value, hook, idx_diff, gender_direction, alpha):
    """Scale the projection of gendered token embeddings onto the gender direction.
    alpha=1.0: no change (keep original)
    alpha=0.0: fully project out the gender direction
    alpha<0.0: reverse the gender direction
    """
    gendered_embeds = value[:, idx_diff, :]
    projection = torch.einsum('bnd,d->bn', gendered_embeds, gender_direction)
    value[:, idx_diff, :] -= (1 - alpha) * projection.unsqueeze(-1) * gender_direction.unsqueeze(0).unsqueeze(0)
    return value

def actadd_hook(value, hook, idx_diff, steering_vector, alpha):
    """Add a scaled steering vector to gendered token embeddings.
    alpha=0.0: no change (keep original)
    alpha=1.0: add the full steering vector
    alpha=-1.0: subtract the full steering vector
    """
    value[:, idx_diff, :] += alpha * steering_vector.unsqueeze(0).unsqueeze(0)
    return value


# ============================================================
# STEP 5: Evaluate steering
# ============================================================
def get_steered_doc_embedding(model, tl_model, input_ids, attention_mask, idx_diff, direction, alpha, method="actadd"):
    """Run the model with steering hook and return the pooled document embedding."""
    if method == "actadd":
        hook_fn = partial(actadd_hook, idx_diff=idx_diff, steering_vector=direction, alpha=alpha)

    with torch.no_grad():
        encoder_output = tl_model.run_with_hooks(
            input_ids,
            fwd_hooks=[("hook_full_embed", hook_fn)],
            return_type="embeddings",
        )

    pooled = model._pooling(encoder_output, attention_mask)
    if model._normalize:
        from mechir.util import normalize_outputs
        pooled = normalize_outputs(pooled)

    return pooled


def evaluate_steering(df, model, tl_model, tokenizer, direction, alpha_values, method="actadd"):
    """Evaluate score differences at different steering strengths."""

    results = {alpha: {"MvN": [], "FvN": [], "MvF": [], "m_scores": [], "f_scores": [], "n_scores": []} for alpha in alpha_values}
    unique_groups = df.drop_duplicates(subset="doc_group_id")

    for _, row in tqdm(unique_groups.iterrows(), total=len(unique_groups), desc="Evaluating steering"):
        query = row["query"]
        doc_m = row["m_doc"]
        doc_f = row["f_doc"]
        doc_n = row["n_doc"]

        # Find gendered token positions
        tok_m, tok_f, tok_n, idx_diff = get_gendered_token_indices(
            doc_m, doc_f, doc_n, tokenizer
        )

        if idx_diff is None or len(idx_diff) == 0:
            continue

        # Tokenize query and get query embedding (no steering needed)
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

                doc_embed = get_steered_doc_embedding(
                    model, tl_model, input_ids, attention_mask,
                    idx_diff, direction, alpha, method=method
                )

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
# STEP 6: Summarize and plot
# ============================================================
def summarize_results(results, alpha_values, save_dir, no_steering_alpha=1.0):
    """Print summary statistics and plot score differences vs alpha."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'font.family': 'serif', 'font.size': 10})

    print("\n" + "=" * 60)
    print("STEERING RESULTS")
    print("=" * 60)

    comparisons = ["MvN", "FvN", "MvF"]

    for comp in comparisons:
        print(f"\n{comp}:")
        print(f"  {'alpha':>8s}  {'mean_diff':>10s}  {'std_diff':>10s}  {'mean_abs':>10s}")
        for alpha in alpha_values:
            diffs = results[alpha][comp]
            print(f"  {alpha:>8.2f}  {np.mean(diffs):>10.6f}  {np.std(diffs):>10.6f}  {np.mean(np.abs(diffs)):>10.6f}")

    # Plot mean absolute score difference vs alpha
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    colors = {'MvN': '#fc8d62', 'FvN': '#66c2a5', 'MvF': '#8da0cb'}

    for i, comp in enumerate(comparisons):
        ax = axes[i]
        mean_abs_diffs = [np.mean(np.abs(results[a][comp])) for a in alpha_values]
        ax.plot(alpha_values, mean_abs_diffs, 'o-', color=colors[comp], linewidth=2, markersize=6)
        ax.set_xlabel('α (steering strength)')
        if i == 0:
            ax.set_ylabel('Mean |ΔScore|')
        ax.set_title(comp)
        ax.grid(True, alpha=0.3)
        ax.axvline(x=no_steering_alpha, color='gray', linestyle='--', alpha=0.5, label='no steering')
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'steering_proof_of_concept.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nSaved plot to {save_dir}/steering_proof_of_concept.png")

def plot_all_models_grid(save_dir, model_names, direction_names, alpha_values, method="projection", no_steering_alpha=1.0):
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
        squeeze=False,  # add this line
    )

    for i, short_name in enumerate(short_names):
        for j, (direction_key, direction_label) in enumerate(direction_names.items()):
            ax = axes[i, j]

            # Load results
            results_path = os.path.join(save_dir, short_name, direction_key, method, 'steering_results.json')
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

            # Row labels (model names)
            if j == 0:
                ax.set_ylabel(short_name, fontsize=9)

            # Column titles (direction names)
            if i == 0:
                ax.set_title(direction_label, fontsize=10)

            # X labels on bottom row only
            if i == n_models - 1:
                ax.set_xlabel('α')

    # Single legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
              bbox_to_anchor=(0.5, 1.02), fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(os.path.join(save_dir, f'all_models_grid_{method}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved grid plot to {save_dir}/all_models_grid.png")

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

    for model_name, model_cfg in MODEL_NAMES.items():
        print(f"\n{'#'*60}")
        print(f"MODEL: {model_name}")
        print(f"{'#'*60}")

        model = Dot(model_name, pooling_type=model_cfg["pool"], sim_func_type=model_cfg["sim_func"])
        tl_model = model._model
        tokenizer = model.tokenizer

        short_name = model_name.split("/")[-1]
        model_save_dir = os.path.join(SAVE_DIR, short_name)
        os.makedirs(model_save_dir, exist_ok=True)

        # Save diff sets for reuse
        diff_sets_path = os.path.join(model_save_dir, 'diff_sets.pt')
        if not os.path.exists(diff_sets_path):
            print("\nCollecting embedding differences...")
            diff_sets, doc_group_info = collect_embedding_differences(df, tl_model, tokenizer)
            torch.save({k: torch.tensor(v) for k, v in diff_sets.items()}, diff_sets_path)
        else:
            print("\nLoading cached embedding differences...")
            loaded = torch.load(diff_sets_path)
            diff_sets = {k: v.numpy() for k, v in loaded.items()}

        for direction_key, direction_label in direction_names.items():
            diffs = diff_sets[direction_key]

            # Compute ActAdd vector (mean difference)
            actadd_vector = torch.tensor(diffs.mean(axis=0), dtype=torch.float32).to(DEVICE)
            print(f"ActAdd vector norm: {actadd_vector.norm().item():.4f}")

            for method_name, method_cfg in method_configs.items():
                print(f"\n{'='*60}")
                print(f"Direction: {direction_label} | Method: {method_name}")
                print(f"{'='*60}")

                if method_name == "actadd":
                    direction = actadd_vector

                method_save_dir = os.path.join(model_save_dir, direction_key, method_name)
                os.makedirs(method_save_dir, exist_ok=True)

                print("\nEvaluating steering...")
                results = evaluate_steering(
                    df, model, tl_model, tokenizer, direction, ALPHA_VALUES, method=method_name
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
                torch.save(direction.cpu(), os.path.join(method_save_dir, 'direction.pt'))

        # Free memory
        del model, tl_model, tokenizer
        torch.cuda.empty_cache()

    # Plot all models grid — one per method
    for method_name in method_configs:
        plot_all_models_grid(
            SAVE_DIR, list(MODEL_NAMES.keys()), direction_names, ALPHA_VALUES,
            method=method_name, no_steering_alpha=method_configs[method_name]["no_steering_alpha"]
        )

    print(f"\nAll results saved to {SAVE_DIR}/")