import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from torch.testing import assert_close

def batched_dot_product(a: Tensor, b: Tensor):
    """
    Calculating the dot product between two tensors a and b.

    Parameters
    ----------
    a: torch.Tensor
        size: batch_size x vector_dim
    b: torch.Tensor
        size: batch_size x vector_dim
    Returns
    -------
    torch.Tensor: size of (batch_size)
        dot product
    """
    return (a * b).sum(dim=-1)

def batched_cos_sim(a: Tensor, b: Tensor):
    # a = F.normalize(a, dim=-1)
    # b = F.normalize(b, dim=-1)
    return (a * b).sum(dim=-1)


def linear_rank_function(patch_score: Tensor, score: Tensor, score_p: Tensor):
    return (patch_score - score) / (score_p - score)

def robust_rank_function(patch_score: Tensor, score: Tensor, score_p: Tensor):
    return (patch_score - score) / torch.sqrt(1 + (score_p - score))

def normalize_outputs(outputs: Tensor):
    return F.normalize(outputs, p=2, dim=-1)

def sort_docs_by_score_batch(
    docs_A: Dict[str, torch.Tensor],
    docs_B: Dict[str, torch.Tensor],
    scores_A: torch.Tensor,
    scores_B: torch.Tensor
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Sorts a batch of document pairs and scores into 'higher_score'
    and 'lower_score' groups, preserving the batch order.

    Args:
        docs_A: A dictionary of tokenized documents.
        docs_B: A dictionary of tokenized documents (batch-aligned with docs_A).
        scores_A: A 1D tensor of scores for docs_A.
        scores_B: A 1D tensor of scores for docs_B.

    Returns:
        A tuple containing:
        - higher_score_docs (TokenizedDict): Docs with the higher scores.
        - lower_score_docs (TokenizedDict): Docs with the lower scores.
        - higher_scores (ScoreTensor): The higher scores.
        - lower_scores (ScoreTensor): The lower scores.
    """
    
    # 1. Determine the higher and lower scores
    higher_scores = torch.maximum(scores_A, scores_B)
    lower_scores = torch.minimum(scores_A, scores_B)
    
    # 2. Create a boolean mask where True means A's score is higher
    is_A_higher_mask = scores_A > scores_B
    
    higher_score_docs = {}
    lower_score_docs = {}
    
    batch_size = is_A_higher_mask.shape[0]
    
    # 3. Iterate over the keys in the tokenized dict (e.g., "input_ids")
    for key in docs_A.keys():
        tensor_A = docs_A[key]
        tensor_B = docs_B[key]
        
        # Reshape the mask to be broadcastable
        num_extra_dims = tensor_A.ndim - 1
        reshape_dims = (batch_size,) + (1,) * num_extra_dims
        broadcastable_mask = is_A_higher_mask.view(reshape_dims)
        
        # 4. Use torch.where to select the correct tensor rows
        # Where mask is True (A is higher), pick tensor_A
        higher_score_docs[key] = torch.where(broadcastable_mask, tensor_A, tensor_B)
        
        # Where mask is True (A is higher), pick tensor_B
        lower_score_docs[key] = torch.where(broadcastable_mask, tensor_B, tensor_A)
        
    return higher_score_docs, lower_score_docs, higher_scores, lower_scores

def get_hook_layer(hook_name: str):
    # Returns the layer index if the name has the form '_model.blocks.{layer}.{...}'
    # Helper function that's mainly useful on HookedTransformer
    # If it doesn't have this form, raises an error -
    if hook_name is None:
        raise ValueError("Name cannot be None")
    split_name = hook_name.split(".")
    return int(split_name[2])

def create_attn_mask(
    docs: Dict[str, torch.Tensor],
    mask_type: str,
    neg_inf: float = -1e9,
) -> torch.Tensor:
    """
    Create an additive attention-logit mask for attn_scores.

    Returns:
      masks: FloatTensor[batch, input_len, input_len] where values are 0 except
             neg_inf at masked (q_pos, k_pos) entries.

    Assumptions (for SEP position inference):
      - docs["attention_mask"] is 1 for real tokens, 0 for padding
      - SEP is the last real token (common for [CLS] ... [SEP] padded batches)
    """
    input_ids = docs["input_ids"]          # [B, L]
    attn_mask = docs["attention_mask"]     # [B, L]

    if input_ids.ndim != 2 or attn_mask.ndim != 2:
        raise ValueError(
            f"Expected input_ids and attention_mask to be [B,L]. "
            f"Got input_ids {tuple(input_ids.shape)}, attention_mask {tuple(attn_mask.shape)}"
        )

    B, L = input_ids.shape
    device = input_ids.device

    # Infer SEP position per example as the last non-pad token index:
    # length[b] = number of real tokens; last index = length[b] - 1
    lengths = attn_mask.long().sum(dim=1)  # [B]
    if torch.any(lengths <= 0):
        bad = (lengths <= 0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(f"Some sequences have no valid tokens (attention_mask sum=0). Bad batch idx: {bad}")

    sep_pos = (lengths - 1).clamp(min=0)   # [B]


    if mask_type == "all_to_sep":
        # masks: [B, Q, K] = [B, L, L]
        masks = torch.zeros((B, L, L), device=device, dtype=torch.float32)

        b_idx = torch.arange(B, device=device)

        # Block all destination tokens (all q) from attending to SEP key (k = sep_pos[b])
        masks[b_idx, :, sep_pos] = neg_inf

        return masks

    raise NotImplementedError(f"Unknown mask_type={mask_type}")

def seed_everything(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def is_pyterrier_availible():
    try:
        import pyterrier as pt

        return True
    except ImportError:
        return False


def is_ir_axioms_availible():
    try:
        import ir_axioms

        return True
    except ImportError:
        return False


def is_ir_datasets_availible():
    try:
        import ir_datasets

        return True
    except ImportError:
        return False


def is_sae_lens_availible():
    try:
        import sae_lens

        return True
    except ImportError:
        return False


def load_json(file: str):
    import json
    import gzip

    """
    Load a JSON or JSONL (optionally compressed with gzip) file.

    Parameters:
    file (str): The path to the file to load.

    Returns:
    dict or list: The loaded JSON content. Returns a list for JSONL files, 
                  and a dict for JSON files.

    Raises:
    ValueError: If the file extension is not recognized.
    """
    if file.endswith(".json"):
        with open(file, "r") as f:
            return json.load(f)
    elif file.endswith(".jsonl"):
        with open(file, "r") as f:
            return [json.loads(line) for line in f]
    elif file.endswith(".json.gz"):
        with gzip.open(file, "rt") as f:
            return json.load(f)
    elif file.endswith(".jsonl.gz"):
        with gzip.open(file, "rt") as f:
            return [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unknown file type for {file}")


def save_json(data, file: str):
    import json
    import gzip

    """
    Save data to a JSON or JSONL file (optionally compressed with gzip).

    Parameters:
    data (dict or list): The data to save. Must be a list for JSONL files.
    file (str): The path to the file to save.

    Raises:
    ValueError: If the file extension is not recognized.
    """
    if file.endswith(".json"):
        with open(file, "w") as f:
            json.dump(data, f)
    elif file.endswith(".jsonl"):
        with open(file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    elif file.endswith(".json.gz"):
        with gzip.open(file, "wt") as f:
            json.dump(data, f)
    elif file.endswith(".jsonl.gz"):
        with gzip.open(file, "wt") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    else:
        raise ValueError(f"Unknown file type for {file}")


def activation_cache_to_disk(activation_cache, path):
    cache_dict = activation_cache.cache_dict
    has_batch_dim = activation_cache.has_batch_dim

    cache_dict = {k: v.cpu().numpy().tolist() for k, v in cache_dict.items()}
    out = {
        "cache_dict": cache_dict,
        "has_batch_dim": has_batch_dim,
    }
    save_json(out, path)


def disk_to_activation_cache(path, model):
    from transformer_lens import ActivationCache

    data = load_json(path)
    cache_dict = {k: torch.tensor(v) for k, v in data["cache_dict"].items()}
    has_batch_dim = data["has_batch_dim"]
    return ActivationCache(cache_dict, model, has_batch_dim)

def load_sentence_transformer_model(model_name):
    model = SentenceTransformer(model_name)
    return model

def encode_sentence_transformer_input(model, input):
    return model.encode(input)



def test_tl_vs_hf_model_outputs(
        model, 
        queries, 
        documents,
        model_name
    ):

    hf_model = model._Dot__hf_model
    tl_model = model._model
    st_model = load_sentence_transformer_model(model_name)


    # Test queries
    queries_hf = hf_model.forward(queries["input_ids"], attention_mask=queries["attention_mask"]).get("last_hidden_state")
    queries_tl = tl_model.forward(queries["input_ids"], attention_mask=queries["attention_mask"])
    assert_close(queries_hf, queries_tl, rtol=1.3e-6, atol=4e-5)
    
    queries_st = st_model[0].auto_model.forward(queries["input_ids"], attention_mask=queries["attention_mask"]).get("last_hidden_state")
    assert_close(queries_hf, queries_st, rtol=1.3e-6, atol=4e-5)

    # Test documents
    docs_hf = hf_model.forward(documents["input_ids"], attention_mask=documents["attention_mask"]).get("last_hidden_state")
    docs_tl = tl_model.forward(documents["input_ids"], attention_mask=documents["attention_mask"])
    assert_close(docs_hf, docs_tl, rtol=1.3e-6, atol=4e-5)

    docs_st = st_model[0].auto_model.forward(documents["input_ids"], attention_mask=documents["attention_mask"]).get("last_hidden_state")
    assert_close(docs_hf, docs_st, rtol=1.3e-6, atol=4e-5)

    # Test pooling
    pooling_func = model._pooling

    queries_hf_cls = pooling_func(queries_hf, attention_mask=queries["attention_mask"])
    queries_tl_cls = pooling_func(queries_tl, attention_mask=queries["attention_mask"])
    assert_close(queries_hf_cls, queries_tl_cls, rtol=1.3e-6, atol=4e-5)

    queries_st_cls = st_model[1]({"token_embeddings": queries_st, "attention_mask": queries["attention_mask"]})["sentence_embedding"]
    assert_close(queries_hf_cls, queries_st_cls)

    docs_hf_cls = pooling_func(docs_hf, attention_mask=documents["attention_mask"])
    docs_tl_cls = pooling_func(docs_tl, attention_mask=documents["attention_mask"])
    assert_close(docs_hf_cls, docs_tl_cls, rtol=1.3e-6, atol=4e-5)

    docs_st_cls = st_model[1]({"token_embeddings": docs_st, "attention_mask": documents["attention_mask"]})["sentence_embedding"]
    assert_close(docs_hf_cls, docs_st_cls, rtol=1.3e-6, atol=4e-5)


    # Test scoring
    # score_func = model._score_func

    # scores_hf = score_func(queries_hf_cls, docs_hf_cls)
    # scores_tl = score_func(queries_tl_cls, docs_tl_cls)
    # assert_close(scores_hf, scores_tl, rtol=1.3e-6, atol=4e-5)


def test_tl_vs_st_model_outputs(
    tl_model,
    model_name,
    queries,
    documents,
):
    st_model = SentenceTransformer(model_name)

    queries_tl = tl_model.forward(**queries)
    queries_st = st_model(queries)["sentence_embedding"]
    assert_close(queries_tl, queries_st, rtol=1.3e-6, atol=4e-5)

    docs_tl = tl_model.forward(**documents)
    docs_st = st_model(documents)["sentence_embedding"]
    assert_close(docs_tl, docs_st, rtol=1.3e-6, atol=4e-5)

    score_func = tl_model._score_func
    
    scores_tl = score_func(queries_tl, docs_tl)
    scores_st = util.cos_sim(queries_st, docs_st)
    assert_close(docs_tl, docs_st, rtol=1.3e-6, atol=4e-5)
