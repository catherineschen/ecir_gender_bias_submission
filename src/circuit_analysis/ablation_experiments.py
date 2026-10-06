import torch
import numpy as np
import pandas as pd

import argparse
from tqdm import tqdm
import os
import json

from mechir import Dot
from mechir.data import DotDataCollator, MechDataset
from transformer_lens import utils

from baseline import process_frame, _split_docno

MODEL_NAMES = {
    "sentence-transformers/msmarco-distilbert-base-tas-b": {
        "pool": "cls",
        "sim_func": "dot",
        "heads_to_ablate": [(4,1), (4,5), (4,10), (5,3), (5,11)],
        "mlps_to_ablate": [0, 1, 2, 3, 4, 5],
    },
    "sentence-transformers/msmarco-distilbert-dot-v5": {
        "pool": "mean",
        "sim_func": "dot",
        "heads_to_ablate": [(4,1), (4,5), (4,10), (5,3), (5,11)],
        "mlps_to_ablate": [0, 1, 2, 3, 4, 5],
    },
    "sentence-transformers/multi-qa-distilbert-dot-v1": {
        "pool": "cls",
        "sim_func": "dot",
        "heads_to_ablate": [(4,1), (4,5), (4,10), (5,3), (5,11)],
        "mlps_to_ablate": [0, 1, 2, 3, 4, 5],
    },
    "sentence-transformers/msmarco-bert-base-dot-v5": {
        "pool": "mean",
        "sim_func": "dot",
        "heads_to_ablate": [(10,6), (11,3), (11,11)],
        "mlps_to_ablate": [6, 7, 8, 9, 10, 11],
    },
    "sentence-transformers/multi-qa-MiniLM-L6-dot-v1": {
        "pool": "cls",
        "sim_func": "dot",
        "heads_to_ablate": [(4,3), (4,8), (4,9), (5,1), (5,4), (5,6), (5,11)],
        "mlps_to_ablate": [0, 1, 2, 3, 4, 5],
    },
}


def load_bi(model_name_or_path: str):
    pool = MODEL_NAMES[model_name_or_path]["pool"]
    sim_func = MODEL_NAMES[model_name_or_path]["sim_func"]
    return Dot(model_name_or_path,pooling_type=pool, sim_func_type=sim_func), DotDataCollator


def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model_type", "-mt", choices=["bi", "cross"], required=True,
        help="What type of model to use for the experiment.",
    )
    parser.add_argument(
        "--ablation_type", default="zero",
    )
    parser.add_argument(
        "--include_mlps", default=False, action="store_true",
    )
    parser.add_argument(
        "--leave_one_out", default=False, action="store_true",
    )
    parser.add_argument(
        "--batch_size", "-bs", default="32",
    )
    parser.add_argument(
        "--save_dir", "-sd", required=True,
    )
    parser.add_argument(
        "--means_dir", "-md", default="head_means",
    )
    parser.add_argument(
        "--use_reduced_dataset", "-reduce", action="store_true",
        help="Boolean value of whether to use a smaller dataset (n=6) for testing purposes."
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = utils.get_device()

    # Load data
    df = pd.read_csv("data/grep_bias_ir_mechir_format.csv")
    if args.use_reduced_dataset:
        df = df.head(6)

    processed_frame = process_frame(df)

    dataset = MechDataset(
        processed_frame, pre_perturbed=True, additional_cols=["qid", "docno"]
    )
    # Loop through models
    for model_name in MODEL_NAMES:

        # Load model
        if args.model_type == "bi":
            model, collator = load_bi(model_name)
        else:
            raise ValueError("model_type must be either 'bi' or 'cross'")
        
        model.to(device)
        
        # Collate data
        collator = collator(
            model.tokenizer, pre_perturbed=True, additional_cols=["qid", "docno"]
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=int(args.batch_size), collate_fn=collator
        )

        heads_to_ablate = MODEL_NAMES[model_name]["heads_to_ablate"]
        mlps_to_ablate = MODEL_NAMES[model_name]["mlps_to_ablate"]

        if not args.leave_one_out:
            head_settings = [("none", heads_to_ablate)]
        else:
            head_settings = [
                (leave_out, [h for h in heads_to_ablate if h != leave_out])
                for leave_out in heads_to_ablate
            ]

        # Ablate heads (with or without mlps) and save
        if not args.include_mlps:
            mlp_settings = [("none", None)]  # (label, mlp_layers_to_ablate)
        else:
            # loop through your provided list; each run ablates ONE mlp layer at a time
            mlp_settings = [(str(layer), [layer]) for layer in mlps_to_ablate]


        head_means = None
        if args.ablation_type == "mean":
            safe_model_dir = model_name.replace("/", "-")
            means_path = os.path.join(args.means_dir, f"head_means_{safe_model_dir}.pt")
            raw = torch.load(means_path, map_location=device)
            head_means = {
                tuple(int(x) for x in k.split("_")): v.to(device)
                for k, v in raw.items()
            }

        
        rows = []
        for left_out_head, rest_head_to_ablate in head_settings:
            for mlp_label, mlp_layers_to_ablate in mlp_settings:
                for batch in tqdm(dataloader, desc=f"{model_name} | mlp={mlp_label}"):
                    if args.model_type != "bi":
                        raise NotImplementedError("This snippet assumes bi-encoder batches.")

                    docnos = batch["docno"]
                    qids = batch["qid"]

                    queries = {k: v.to(device) for k, v in batch["queries"].items()}
                    docs_a = {k: v.to(device) for k, v in batch["documents"].items()}
                    docs_b = {k: v.to(device) for k, v in batch["perturbed_documents"].items()}

                    ablated_scores_a, ablated_scores_b, _, _ = model.ablate_attn_and_mlp(
                        queries=queries,
                        documents=docs_a,
                        documents_p=docs_b,
                        heads_to_ablate=rest_head_to_ablate,
                        mlps_to_ablate=mlp_layers_to_ablate,  # None or [layer]
                        ablation_type=args.ablation_type,
                        head_means=head_means,
                    )

                    sa_cpu = ablated_scores_a.detach().cpu().float().numpy()
                    sb_cpu = ablated_scores_b.detach().cpu().float().numpy()
                    diffs = sa_cpu - sb_cpu

                    for docno, qid, sa, sb, diff in zip(docnos, qids, sa_cpu, sb_cpu, diffs):
                        doc_group_id, comparison = _split_docno(docno)
                        rows.append({
                            "model": model_name,
                            "model_type": args.model_type,
                            "doc_group_id": doc_group_id,
                            "comparison": comparison,
                            "docno": docno,
                            "qid": qid,
                            "score_a": float(sa),
                            "score_b": float(sb),
                            "score_diff": float(diff),
                            "ablated_heads": json.dumps(rest_head_to_ablate),
                            "left_out_head": json.dumps(left_out_head),
                            "mlp_layer": mlp_label,
                        })

        out_df = pd.DataFrame(rows)
        out_df["gap_abs"] = out_df["score_diff"].abs()
        summary = (
            out_df.groupby(["mlp_layer", "comparison"])["gap_abs"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
        )
        print(summary)

        # Save one CSV per model
        safe_model_dir = model_name.replace("/", "-")
        os.makedirs(args.save_dir, exist_ok=True)
        out_path = os.path.join(
            args.save_dir,
            args.ablation_type,
            f"ablated_score_diffs_heads_{safe_model_dir}_{args.model_type}.csv"
        )
        out_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


# python ablation_experiments.py -mt bi -sd . --include_mlps
# python ablation_experiments.py -mt bi -sd ./results_ablation/ --leave_one_out