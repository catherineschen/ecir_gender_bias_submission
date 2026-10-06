import torch
import numpy as np
import pandas as pd

import argparse
from tqdm import tqdm
import os

import mechir
from mechir.util import linear_rank_function, robust_rank_function
from mechir import Dot
from mechir.data import DotDataCollator, MechDataset

from transformer_lens import utils


mechir.config("ignore-official", True)


MODEL_NAMES = {
    "sentence-transformers/msmarco-distilbert-base-tas-b": {
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
    "sentence-transformers/msmarco-distilbert-dot-v5": {
        "pool": "mean",
        "sim_func": "dot",
    },
}


def load_bi(model_name_or_path: str):
    pool = MODEL_NAMES[model_name_or_path]["pool"]
    sim_func = MODEL_NAMES[model_name_or_path]["sim_func"]
    return Dot(model_name_or_path,pooling_type=pool, sim_func_type=sim_func), DotDataCollator

def process_frame(frame):
    """
    Transforms a reshaped DataFrame (with m_doc, f_doc, n_doc columns)
    into a "comparison pair" DataFrame suitable for the MechDataset.
    
    Creates three comparisons: MvF, MvN, and FvN.
    """
    
    # Use lists to build the new columns efficiently
    qid_list = []
    query_list = []
    docno_list = []
    text_list = []       # This will be the "corrupted" text
    perturbed_list = [] # This will be the "clean" text
    
    # Iterate through the reshaped frame (one row per doc_group_id)
    for row in frame.itertuples():
        
        # --- 1. Male vs. Female (M as corrupted, F as clean) ---
        if pd.notna(row.m_doc) and pd.notna(row.f_doc):
            qid_list.append(row.q_id)
            query_list.append(row.query)
            # Create a unique docno for this specific comparison pair
            docno_list.append(f"{row.doc_group_id}_MvF") 
            text_list.append(row.m_doc)      # Corrupted/Base text
            perturbed_list.append(row.f_doc) # Clean/Perturbed text
            
        # --- 2. Male vs. Neutral (M as corrupted, N as clean) ---
        if pd.notna(row.m_doc) and pd.notna(row.n_doc):
            qid_list.append(row.q_id)
            query_list.append(row.query)
            docno_list.append(f"{row.doc_group_id}_MvN")
            text_list.append(row.m_doc)
            perturbed_list.append(row.n_doc)
            
        # --- 3. Female vs. Neutral (F as corrupted, N as clean) ---
        if pd.notna(row.f_doc) and pd.notna(row.n_doc):
            qid_list.append(row.q_id)
            query_list.append(row.query)
            docno_list.append(f"{row.doc_group_id}_FvN")
            text_list.append(row.f_doc)
            perturbed_list.append(row.n_doc)

    # Build the final DataFrame from the lists
    output_df = pd.DataFrame({
        "qid": qid_list,
        "query": query_list,
        "docno": docno_list,
        "text": text_list,
        "perturbed": perturbed_list,
    })
    
    return output_df

def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model_type", "-mt", choices=["bi", "cross"], required=True,
        help="What type of model to use for the experiment.",
    )
    parser.add_argument(
        "--model_name_or_path", "-mn", 
        help="Model to perform experiments on."
    )
    parser.add_argument(
        "--patch_type", "-pt", choices=["activation", "path"],
        help="Type of patching experiment to run."
    )
    parser.add_argument(
        "--batch_size", "-bs", required=True,
    )
    parser.add_argument(
        "--result_dir", "-rd", required=True,
    )
    parser.add_argument(
        "--use_reduced_dataset", "-reduce", action="store_true",
        help="Boolean value of whether to use a smaller dataset (n=6) for testing purposes."
    )
    parser.add_argument(
        "--metric", default="metric25"
    )

    # Activation patching args
    parser.add_argument(
        "--activation_patch_component", "-apc",
        choices=["block_all", "head_all", "head_by_pos"],
        help="[Activation Patching] Which components to patch."
    )

    # Path patching args
    parser.add_argument(
        "--sender_type", "-send", choices=["head", "mlp"], #required=True,
        help="[Path Patching] Sender component type."
    )
    parser.add_argument(
        "--receiver_type", "-receive", choices=["head", "mlp", "resid_post"], #required=True,
        help="[Path Patching] Receiver component type."
    )

    parser.add_argument(
        "--direct_includes_mlps", action="store_true",
        help="If true, allows MLPs to be recomputed."
    )
    parser.add_argument(
        "--receiver_node", "-rnode",
        help="Specific component to patch to."
    )
    parser.add_argument(
        "--sender_attn_component", "-sac", choices=["q", "k", "v", "attn", "pattern", "z"], default="z",
        help="Upstream attention component to patch"
    )
    parser.add_argument(
        "--receiver_attn_component", "-rac", choices=["q", "k", "v", "attn", "pattern"], default="v",
        help="Downstream attention component"
    )
    # parser.add_argument(
    #     "--toks_to_patch", nargs="+",
    #     help="Specific tokens to patch",
    # )
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = utils.get_device()

    if args.model_type == "bi":
        # pre_trained_model_name = "sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco"
        model, collator = load_bi(args.model_name_or_path)
    elif args.model_type == "cross":
        model, collator = load_cross(args.model_name_or_path)
    else:
        raise ValueError("model_type must be either 'bi' or 'cross'")

    # Construct patching parameters
    if args.patch_type == "activation":
        patch_kwargs = {
            "patch_type": args.activation_patch_component,    
        }
    elif args.patch_type == "path":
        patch_kwargs = {
            "sender_type": args.sender_type,
            "receiver_type": args.receiver_type,
            "sender_attn_component": args.sender_attn_component,
            "receiver_attn_component": args.receiver_attn_component,    
        }

        if args.direct_includes_mlps:
            patch_kwargs["recompute_mlps"] = args.direct_includes_mlps
        if args.receiver_node:
            if args.receiver_type == "head":
                receiver_node = tuple(map(int, args.receiver_node.split(',')))
            elif args.receiver_type == "mlp":
                receiver_node = int(args.receiver_node)
            # else: # resid_post
            #     receiver_layer = None
            patch_kwargs["receiver_node"] = receiver_node

    patch_kwargs["patching_metric"] = linear_rank_function if args.metric == "metric24" else robust_rank_function


    # Load data
    df = pd.read_csv("data/grep_bias_ir_mechir_format.csv")
    if args.use_reduced_dataset:
        df = df.head(6)

    processed_frame = process_frame(df)
    dataset = MechDataset(
        processed_frame, pre_perturbed=True, additional_cols=["qid", "docno"]
    )

    collator = collator(
        model.tokenizer, pre_perturbed=True, additional_cols=["qid", "docno"]
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=int(args.batch_size), collate_fn=collator
    )

    # Path patching
    patching_outputs = []
    docnos = []
    for batch in tqdm(dataloader):
        if args.model_type == "bi":
            queries = {key: value.to(device) for key, value in batch["queries"].items()}
            documents = {key: value.to(device) for key, value in batch["documents"].items()}
            perturbed_documents = {key: value.to(device) for key, value in batch["perturbed_documents"].items()}

            patching_func = model.activation_patch if args.patch_type == "activation" else model.path_patch
            
            patch_out, _ = patching_func(
                queries,
                documents,
                perturbed_documents,
                **patch_kwargs,
            )

        patching_outputs.append(patch_out)
        docnos.append(batch["docno"])


    # convert to numpy and dump
    output = torch.cat(patching_outputs, dim=0).cpu().detach().numpy()

    # Format save dir/filename
    formatted_model_name = args.model_name_or_path.replace("/", "-")
    dir_path = f"{args.result_dir}/{args.patch_type}_patch/{formatted_model_name}_{args.model_type}"

    if args.patch_type == "activation":
        output_fname = f"{args.activation_patch_component}.npy"
    elif args.patch_type == "path":
        mlp_setting = "recomputed_mlps" if args.direct_includes_mlps else "frozen_mlps"
        dir_path += f"/{mlp_setting}/{args.sender_type}_to_{args.receiver_type}"

        if args.receiver_type == "resid_post":
            rnode_name = "final_resid_post"
        elif isinstance(args.receiver_node, tuple):
            rnode_name = f"head_{args.receiver_node[0]}-{args.receiver_node[1]}"
        else: # mlp
            rnode_name = f"{args.receiver_type}_{str(args.receiver_node)}"
        
        if args.sender_type == "head" and args.receiver_type == "head":
            output_fname = f"{rnode_name}_send_{args.sender_attn_component}_recv_{args.receiver_attn_component}.npy"
        else:
            output_fname = f"{rnode_name}.npy"

    os.makedirs(dir_path, exist_ok=True)
    
    # save results by gender comparison groups
    base_filename = output_fname.replace(".npy", "")

    all_docnos = [docno for batch_tuple in docnos for docno in batch_tuple]

    groups = ["MvN", "FvN", "MvF"]
    for group_name in groups:
        group_docnos = []
        group_indices = []
        # Find all docnos and their indices for this group
        for i, docno in enumerate(all_docnos):
            if docno.endswith(group_name):
                group_docnos.append(docno)
                group_indices.append(i)

        if not group_indices:
            print(f"  Warning: No docnos found for group {group_name}")
            continue

        # --- Save the Docno List ---
        # Creates a file like ".../final_resid_post_MvN_docnos.txt"
        docno_file_path = os.path.join(dir_path, f"{base_filename}_{group_name}_docnos.txt")
        print(f"  Saving {len(group_docnos)} docnos to {docno_file_path}")
        with open(docno_file_path, 'w') as f:
            for docno in group_docnos:
                f.write(f"{docno}\n")
        
        # --- Save the Corresponding Results ---
        # Select the results from the main 'output' array using the indices
        group_results = output[group_indices]

        # Creates a file like ".../final_resid_post_MvN.npy"
        results_file_path = os.path.join(dir_path, f"{base_filename}_{group_name}.npy")
        print(f"  Saving {group_results.shape} results to {results_file_path}")
        np.save(results_file_path, group_results)


    return



if __name__ == "__main__":
    main()