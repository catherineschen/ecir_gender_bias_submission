import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import utils
from mechir.util import get_hook_layer

from ablation_experiments import MODEL_NAMES, load_bi


def collate_docs(batch, tokenizer, max_length=300):
    return tokenizer(
        batch, padding="max_length", truncation=False, max_length=max_length, return_tensors="pt"
    )


def compute_and_save_means(model_name, csv_path, save_dir, batch_size=32):
    device = utils.get_device()
    model, _ = load_bi(model_name)
    model.to(device)
    heads_to_cache = MODEL_NAMES[model_name]["heads_to_ablate"]

    df = pd.read_csv(csv_path)
    docs = pd.concat([df["m_doc"], df["f_doc"], df["n_doc"]], ignore_index=True).tolist()

    dataloader = DataLoader(
        docs,
        batch_size=batch_size,
        collate_fn=lambda b: collate_docs(b, model.tokenizer),
    )

    accumulator = {(l, h): {"sum": None, "count": None} for l, h in heads_to_cache}
    attn_name_filter = lambda name: name.endswith("z")

    for batch in tqdm(dataloader, desc=f"caching {model_name}"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        def hook_fn(value, hook):
            hook_layer = get_hook_layer(hook.name)
            mask = attention_mask.to(value.dtype)
            for layer, head in heads_to_cache:
                if layer != hook_layer:
                    continue
                head_out = value[:, :, head]  # [batch, pos, d_head]
                weighted_sum = (head_out * mask[..., None]).sum(dim=(0, 1))  # [d_head]
                token_count = mask.sum()
                acc = accumulator[(layer, head)]
                if acc["sum"] is None:
                    acc["sum"] = weighted_sum.detach().cpu()
                    acc["count"] = token_count.detach().cpu()
                else:
                    acc["sum"] += weighted_sum.detach().cpu()
                    acc["count"] += token_count.detach().cpu()
            return value

        model.run_with_hooks(
            input_ids,
            attention_mask=attention_mask,
            fwd_hooks=[(attn_name_filter, hook_fn)],
        )

    means = {key: (acc["sum"] / acc["count"]) for key, acc in accumulator.items()}

    safe_model_dir = model_name.replace("/", "-")
    out_path = os.path.join(save_dir, f"head_means_{safe_model_dir}.pt")
    torch.save({f"{l}_{h}": v for (l, h), v in means.items()}, out_path)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/grep_bias_ir_mechir_format.csv")
    parser.add_argument("--save_dir", "-sd", required=True)
    parser.add_argument("--batch_size", "-bs", type=int, default=16)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    os.makedirs(args.save_dir, exist_ok=True)

    for model_name in MODEL_NAMES:
        compute_and_save_means(
            model_name, args.csv_path, args.save_dir, args.batch_size
        )


if __name__ == "__main__":
    main()