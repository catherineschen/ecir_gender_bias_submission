from __future__ import annotations

import torch
import numpy as np
import pandas as pd

import argparse
from tqdm import tqdm
import os

import mechir
from mechir import Cat, Dot
from mechir.data import CatDataCollator, DotDataCollator, MechDataset

from transformer_lens import utils



from typing import Literal, Optional
from sentence_transformers import SentenceTransformer


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


def score_with_sentence_transformers(
    processed_frame: pd.DataFrame,
    model_name: str,
    pool: Literal["cls", "mean"] = "mean",
    sim_func: Literal["dot", "cos"] = "dot",
    batch_size: int = 64,
    device: Optional[str] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Scores each row (qid, query, docno, text, perturbed) using Sentence-Transformers.

    Assumptions:
      - processed_frame has columns: ["qid", "query", "docno", "text", "perturbed"]
      - docno looks like "{doc_group_id}_{MvF|MvN|FvN}" (we parse doc_group_id + comparison)

    Notes:
      - Sentence-Transformers handles pooling internally; `pool` is included only for consistency
        with your metadata, but ST models already bake pooling strategy into the architecture.
      - For cosine models: set sim_func="cos" which uses normalize_embeddings=True.
        For dot models: sim_func="dot" which uses normalize_embeddings=False.

    Returns:
      DataFrame with:
        model, qid, docno, doc_group_id, comparison, score_a, score_b, score_diff
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # cos <-> unit-normalized inner product
    normalize_embeddings = (sim_func == "cos")

    st = SentenceTransformer(model_name, device=device)

    # Pull columns as lists
    qids = processed_frame["qid"].tolist()
    queries = processed_frame["query"].tolist()
    docnos = processed_frame["docno"].tolist()
    docs_a = processed_frame["text"].tolist()
    docs_b = processed_frame["perturbed"].tolist()

    def _split_docno(docno: str):
        base, comp = docno.rsplit("_", 1)
        return int(base), comp

    rows = []

    # Batch over rows; encode queries/docs in reasonably large batches.
    # We encode query, doc_a, doc_b separately because they are different texts.
    # If you want faster: cache query embeddings by qid (can add later).
    n = len(processed_frame)
    it = range(0, n, batch_size)
    if show_progress:
        it = tqdm(it, desc=f"ST scoring {model_name}", total=(n + batch_size - 1) // batch_size)

    for start in it:
        end = min(start + batch_size, n)

        q_batch = queries[start:end]
        a_batch = docs_a[start:end]
        b_batch = docs_b[start:end]

        # Encode: returns np.ndarray [B, D] by default
        q_emb = st.encode(
            q_batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
        a_emb = st.encode(
            a_batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )
        b_emb = st.encode(
            b_batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False,
        )

        # Inner product (dot). If normalized, this equals cosine similarity.
        score_a = np.sum(q_emb * a_emb, axis=1)
        score_b = np.sum(q_emb * b_emb, axis=1)
        score_diff = score_a - score_b

        for i, (sa, sb, sd) in enumerate(zip(score_a, score_b, score_diff)):
            idx = start + i
            doc_group_id, comparison = _split_docno(docnos[idx])
            rows.append({
                "model": model_name,
                "pool": pool,
                "sim_func": sim_func,
                "qid": qids[idx],
                "docno": docnos[idx],
                "doc_group_id": doc_group_id,
                "comparison": comparison,
                "score_a": float(sa),
                "score_b": float(sb),
                "score_diff": float(sd),
            })

    return pd.DataFrame(rows)


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

def _split_docno(docno: str):
    """
    docno format: "{doc_group_id}_{MvF|MvN|FvN}"
    returns (doc_group_id: int, comparison: str)
    """
    base, comp = docno.rsplit("_", 1)
    return int(base), comp


def main():

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--model_type", "-mt", choices=["bi"], required=True,
        help="What type of model to use for the experiment.",
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
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = utils.get_device()

   

    # Load data
    df = pd.read_csv("data/grep_bias_ir_mechir_format.csv")
    if args.use_reduced_dataset:
        df = df.head(6)

    processed_frame = process_frame(df)

    # Sentence Transformers scoring
    for model_name, model_args in MODEL_NAMES.items():
        st_df = score_with_sentence_transformers(
            processed_frame=processed_frame,
            model_name=model_name,
            pool=model_args["pool"],
            sim_func=model_args["sim_func"],
            batch_size=int(args.batch_size),
        )

        safe_model_dir = model_name.replace("/", "-")
        out_path = os.path.join(
            args.result_dir,
            f"baseline_score_diffs_original_{safe_model_dir}_{args.model_type}.csv"
        )
        st_df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")


    return



if __name__ == "__main__":
    main()

