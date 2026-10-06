import torch
import numpy as np
import pandas as pd

import os
import argparse
from tqdm import tqdm
from collections import defaultdict


from transformer_lens import utils
from torch.utils.data import Dataset, DataLoader
from mechir.data import DotDataCollator, MechDataset

from ablation_experiments import MODEL_NAMES, load_bi
from baseline import process_frame

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, Optional
from transformers import PreTrainedTokenizerBase



DOC_GROUP_IDS_NEUTRAL_OVERLAP = [12,30,31,34,35,38,39,50,51,62,63,68,69,184,185,188,189,192,193,194,195]

def make_long_df(df: pd.DataFrame) -> pd.DataFrame:
    m = df[["doc_group_id","q_id","relevant","query","m_d_id","m_doc"]].copy()
    m = m.rename(columns={"m_d_id":"d_id","m_doc":"doc"})
    m["variant"] = "m"

    f = df[["doc_group_id","q_id","relevant","query","f_d_id","f_doc"]].copy()
    f = f.rename(columns={"f_d_id":"d_id","f_doc":"doc"})
    f["variant"] = "f"

    n = df[["doc_group_id","q_id","relevant","query","n_d_id","n_doc"]].copy()
    n = n.rename(columns={"n_d_id":"d_id","n_doc":"doc"})
    n["variant"] = "n"

    long_df = pd.concat([m, f, n], ignore_index=True)
    long_df["variant"] = pd.Categorical(long_df["variant"], ["m","f","n"], ordered=True)
    return long_df.sort_values(["doc_group_id","variant"]).reset_index(drop=True)

class SingleDocTupleDataset(Dataset):
    """
    Returns tuples in exactly the layout BaseCollator.get_data expects:
      (query, doc, [perturbed_doc if pre_perturbed], *additional_cols_values)
    """
    def __init__(self, long_df: pd.DataFrame, additional_cols=None, pre_perturbed: bool = False):
        self.df = long_df.reset_index(drop=True)
        self.additional_cols = additional_cols or []
        self.pre_perturbed = pre_perturbed

        if self.pre_perturbed and "perturbed_doc" not in self.df.columns:
            raise ValueError("pre_perturbed=True but long_df has no 'perturbed_doc' column.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        query = str(r["query"])
        doc = str(r["doc"])

        extra = [r[col] for col in self.additional_cols]

        if self.pre_perturbed:
            pert = str(r["perturbed_doc"])
            return (query, doc, pert, *extra)

        return (query, doc, *extra)
    
@dataclass
class SingleInstanceDotStyleCollator:
    tokenizer: PreTrainedTokenizerBase
    q_max_length: int = 30
    d_max_length: int = 300
    special_mask: bool = False
    additional_cols: Optional[Sequence[str]] = None

    def __call__(self, batch: List[Tuple[Any, ...]]) -> Dict[str, Any]:
        # batch items: (query, doc, *extras)
        queries = [b[0] for b in batch]
        docs = [b[1] for b in batch]

        tokenized_queries = self.tokenizer(
            queries,
            padding="max_length",
            truncation=False,
            max_length=self.q_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask,
        )
        tokenized_docs = self.tokenizer(
            docs,
            padding="max_length",
            truncation=False,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=self.special_mask,
        )

        out: Dict[str, Any] = {
            "queries": dict(tokenized_queries),
            "documents": dict(tokenized_docs),
        }

        # Unpack and attach metadata columns (doc_group_id, d_id, etc.)
        if self.additional_cols:
            unpacked = list(zip(*batch))
            extras_start = 2
            for i, name in enumerate(self.additional_cols):
                vals = list(unpacked[extras_start + i])

                # ints/bools -> tensor; everything else -> list
                if vals and isinstance(vals[0], (int, bool)):
                    out[name] = torch.tensor(vals, dtype=torch.long)
                else:
                    out[name] = vals

        return out
    

def save_model_tokens(tokenizer, input_ids, attention_mask, out_path, *, keep_special_tokens=True):
    """
    Save exactly the model token strings for a single sequence.
    input_ids: (S,) tensor
    attention_mask: (S,) tensor
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ids = input_ids.tolist()
    mask = attention_mask.tolist()

    toks = tokenizer.convert_ids_to_tokens(ids)

    labels = []
    for i, (tok, m) in enumerate(zip(toks, mask)):
        if m != 1:
            continue  # drop padding so labels match attention over "real" tokens
        if (not keep_special_tokens) and (tok in tokenizer.all_special_tokens):
            continue
        labels.append(f"{i}\t{tok}")  # keep original position index + token string

    with open(out_path, "w") as f:
        f.write("\n".join(labels))
        f.write("\n")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", "-mt", choices=["bi"], default="bi",
        help="What type of model to use for the experiment.",
    )
    parser.add_argument("--batch_size", default=8)
    parser.add_argument("--save_dir", "-sd")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = utils.get_device()

    # Load data
    df = pd.read_csv("data/grep_bias_ir_mechir_format.csv")

    # Filter for target documents

    # Neutral term lexical overlap
    df = df[df["doc_group_id"].isin(DOC_GROUP_IDS_NEUTRAL_OVERLAP)]
    # df = df[~df["doc_group_id"].isin(DOC_GROUP_IDS_NEUTRAL_OVERLAP)]
    long_df = make_long_df(df)

    additional_cols = ["doc_group_id", "d_id", "q_id", "relevant", "variant"]
    dataset = SingleDocTupleDataset(long_df, additional_cols=additional_cols, pre_perturbed=False)

    os.makedirs(args.save_dir, exist_ok=True)


    for model_name in MODEL_NAMES:

        if model_name != "sentence-transformers/msmarco-distilbert-base-tas-b":
            continue

        # Load model
        if args.model_type == "bi":
            model, _ = load_bi(model_name)
        else:
            raise ValueError("model_type must be either 'bi' or 'cross'")
        
        model.to(device)

        # Model-specific output dir
        safe_model_name = model_name.replace("/", "_")
        model_dir = os.path.join(args.save_dir, safe_model_name)
        os.makedirs(model_dir, exist_ok=True)

        # Subfolders
        attn_dir = os.path.join(model_dir, "attn")
        labels_dir = os.path.join(model_dir, "labels")
        os.makedirs(attn_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        # Collate data
        collate_fn = SingleInstanceDotStyleCollator(
            tokenizer=model.tokenizer,
            q_max_length=30,
            d_max_length=300,
            special_mask=False,
            additional_cols=additional_cols,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=int(args.batch_size), collate_fn=collate_fn, shuffle=False,
        )

        # ----------------------------
        # Initialize storage
        # ----------------------------
        q_store = defaultdict(list)  # (L,H) -> [Tensor(B,Tq,Tq), ...]
        d_store = defaultdict(list)  # (L,H) -> [Tensor(B,Td,Td), ...]

        meta_rows = []
        row_idx = 0

        # For labels: store as one file per row_idx to keep lookup easy
        # We'll write them batch-by-batch to avoid holding lots of python strings in memory.
        # (Still "exactly what the model sees": includes padding + special tokens.)
        q_labels_path = os.path.join(labels_dir, "query_tokens_by_row.txt")
        d_labels_path = os.path.join(labels_dir, "doc_tokens_by_row.txt")

        # Truncate existing label files if rerunning
        open(q_labels_path, "w").close()
        open(d_labels_path, "w").close()

        heads_to_cache = MODEL_NAMES[model_name]["heads_to_ablate"]
        attn_pattern_name_filter = lambda name: name.endswith("pattern")

        # Get attention patterns for each document

        for batch in tqdm(dataloader, desc=f"{model_name}"):

            # batch_doc_group_ids = batch["doc_group_id"]
            # batch_qids = batch["q_id"]
            # batch_variants = batch["variant"]

            B = len(batch["variant"])

            # ---- metadata (aligned to batch order) ----
            doc_group_ids = batch["doc_group_id"].tolist() if torch.is_tensor(batch["doc_group_id"]) else list(batch["doc_group_id"])
            q_ids = batch["q_id"].tolist() if torch.is_tensor(batch["q_id"]) else list(batch["q_id"])
            d_ids = batch["d_id"].tolist() if torch.is_tensor(batch["d_id"]) else list(batch["d_id"])
            variants = list(batch["variant"])

            # ---- token labels (exact tokens incl pad + specials) ----
            # Note: convert_ids_to_tokens works on lists; do it batched.
            q_input_ids = batch["queries"]["input_ids"].cpu()
            d_input_ids = batch["documents"]["input_ids"].cpu()

            q_tokens_batch = [model.tokenizer.convert_ids_to_tokens(ids) for ids in q_input_ids.tolist()]  # list[list[str]]
            d_tokens_batch = [model.tokenizer.convert_ids_to_tokens(ids) for ids in d_input_ids.tolist()]

            # append to label files in a row-aligned way:
            # Format: row_idx \t tok0 tok1 tok2 ...
            with open(q_labels_path, "a") as fq, open(d_labels_path, "a") as fd:
                for i in range(B):
                    cur_row = row_idx + i
                    fq.write(f"{cur_row}\t" + "\t".join(q_tokens_batch[i]) + "\n")
                    fd.write(f"{cur_row}\t" + "\t".join(d_tokens_batch[i]) + "\n")

           

            # Cache queries and docs
            queries = {k: v.to(device) for k, v in batch["queries"].items()}
            docs = {k: v.to(device) for k, v in batch["documents"].items()}

            _, query_cache = model.run_with_cache(
                **queries,
                return_cache_object=True,
                names_filter=attn_pattern_name_filter,
            )
            _, doc_cache = model.run_with_cache(
                **docs,
                return_cache_object=True,
                names_filter=attn_pattern_name_filter,
            )

            scores, _, _, _ = model.score(queries, docs)
            scores_cpu = scores.detach().cpu().tolist()


            for layer, head in heads_to_cache:
                q = query_cache[f"_model.blocks.{layer}.attn.hook_pattern"][:, head].detach().cpu()
                d = doc_cache[f"_model.blocks.{layer}.attn.hook_pattern"][:, head].detach().cpu()
                q_store[(layer, head)].append(q)  # (B,Tq,Tq)
                d_store[(layer, head)].append(d)  # (B,Td,Td)
            
            # store metadata rows
            for i in range(B):
                meta_rows.append({
                    "row_idx": row_idx + i,
                    "doc_group_id": int(doc_group_ids[i]),
                    "q_id": int(q_ids[i]),
                    "d_id": int(d_ids[i]),
                    "variant": variants[i],
                    "baseline_score": float(scores_cpu[i]),
                })
            row_idx += B

        # ----------------------------
        # Save metadata
        # ----------------------------
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_parquet(os.path.join(model_dir, "meta.parquet"), index=False)

        # Also save raw token ids + masks (handy for debugging / alternate label generation)
        # These are optional; comment out if you don't want them.
        # If you want them, collect them like q_store/d_store; simplest for N~63 is to just
        # re-run through the dataloader once more. But you can also collect during the loop.
        # (Keeping minimal here; labels already saved above.)

        # ----------------------------
        # Save attention arrays (one file per (layer, head))
        # ----------------------------
        for (layer, head), chunks in q_store.items():
            q_all = torch.cat(chunks, dim=0).numpy()  # (N,Tq,Tq)
            d_all = torch.cat(d_store[(layer, head)], dim=0).numpy()  # (N,Td,Td)

            np.save(os.path.join(attn_dir, f"q_attn_L{layer}_H{head}.npy"), q_all)
            np.save(os.path.join(attn_dir, f"d_attn_L{layer}_H{head}.npy"), d_all)

        # Optional: save a quick manifest of which heads were cached
        with open(os.path.join(model_dir, "heads_cached.txt"), "w") as f:
            for layer, head in heads_to_cache:
                f.write(f"{layer}\t{head}\n")


if __name__ == "__main__":
    main()