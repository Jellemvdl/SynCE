#!/usr/bin/env python3
"""
evaluate_siglip_zero_shot.py

Evaluate zero-shot SigLIP classification on a test split defined by:
  - class_mapping.csv: columns [id,name]
  - test.csv:          columns [filename,class_id]

Metrics reported:
  - Accuracy@1
  - Accuracy@5
  - Macro Precision
  - Macro Recall

Example:
  python evaluate_siglip_zero_shot.py \
    --class_mapping class_mapping.csv \
    --test_csv test.csv \
    --image_root /scratch/leejmvd/finegrained_text/data/parasite_egg \
    --generic   # (optional) use generic "An image of a {name}" prompts

Requires: torch, PIL, scikit-learn, open_clip, pandas, numpy
"""

import argparse
import json
import pathlib
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import precision_score, recall_score
from tqdm import tqdm
import open_clip  # SigLIP backbone via timm hub


SIGLIP_HUB = {
    "SigLIP": {
        "model_name": "hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "pretrained": "",
    },
}


@torch.no_grad()
def load_siglip(model_key: str):
    cfg = SIGLIP_HUB[model_key]
    model_name, pretrained = cfg["model_name"], cfg["pretrained"] or None
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name=model_name, pretrained=pretrained
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    return model, preprocess, tokenizer, device


@torch.no_grad()
def encode_text(model, tokenizer, texts: List[str], device: torch.device) -> torch.Tensor:
    tok = tokenizer(texts).to(device)
    feat = model.encode_text(tok)
    return F.normalize(feat, dim=-1)  # [N,D]


@torch.no_grad()
def encode_images(
    model, preprocess, device, img_paths: List[pathlib.Path], batch_size: int = 32
) -> torch.Tensor:
    """Return [N,D] L2-normalized image features; missing files are skipped (masked separately)."""
    feats = []
    batch, keep_mask = [], []
    for p in img_paths:
        if p.is_file():
            img = preprocess(Image.open(p).convert("RGB"))
            batch.append(img)
            keep_mask.append(True)
        else:
            keep_mask.append(False)
        if len(batch) == batch_size:
            imgs = torch.stack(batch).to(device)
            with torch.no_grad():
                f = model.encode_image(imgs)
                f = F.normalize(f, dim=-1)
            feats.append(f.cpu())
            batch.clear()
    if batch:
        imgs = torch.stack(batch).to(device)
        with torch.no_grad():
            f = model.encode_image(imgs)
            f = F.normalize(f, dim=-1)
        feats.append(f.cpu())
    if feats:
        feats = torch.cat(feats, dim=0)  # only for kept images
    else:
        feats = torch.empty((0, model.visual.output_dim), dtype=torch.float32)
    return feats, keep_mask


def topk_acc(scores: torch.Tensor, target: np.ndarray, k: int = 1) -> float:
    """scores: [N,C] tensor on CPU, target: [N] np.int"""
    target_t = torch.tensor(target, device=scores.device)
    return (
        scores.topk(k, 1).indices.eq(target_t[:, None]).any(1).float().mean().item()
    )


def main():
    DATASET = "military_vehicles"  # for default paths; change as needed
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--class_mapping", default=f"~/SynCE/artifacts/data/{DATASET}/class_mapping.csv", help="CSV with columns: id,name")
    ap.add_argument("--test_csv", default=f"~/SynCE/artifacts/data/{DATASET}/test.csv", help="CSV with columns: filename,class_id")
    ap.add_argument("--image_root", default=f"/scratch/leejmvd/finegrained_text/data/{DATASET}/images/", help="Root folder with images")
    ap.add_argument("--model_key", choices=SIGLIP_HUB.keys(), default="SigLIP")
    ap.add_argument("--generic", action="store_true",
                    help="Use generic prompts ('An image of a {name}') instead of domain-specific")
    ap.add_argument("--batch_size", type=int, default=64, help="Image batch size for encoding")
    ap.add_argument("--save_predictions", default=None,
                    help="Optional: path to write per-image predictions JSON")
    args = ap.parse_args()

    # 1) Load CSVs
    df_map = pd.read_csv(args.class_mapping)  # id,name
    df_test = pd.read_csv(args.test_csv)      # filename,class_id
    # basic checks
    assert set(df_map.columns) >= {"id", "name"}
    assert set(df_test.columns) >= {"filename", "class_id"}

    # 2) Build class prompts and encode
    df_map = df_map.sort_values("id")
    class_names = df_map["name"].tolist()
    num_cls = len(class_names)

    if args.generic:
        prompts = [f"An image of a {name}" for name in class_names]
    else:
        prompts = [f"An image of {name}, a military vehicle" for name in class_names]

    model, preprocess, tokenizer, device = load_siglip(args.model_key)
    with torch.no_grad():
        class_emb = encode_text(model, tokenizer, prompts, device)  # [C,D]
    class_emb_t = class_emb.T  # [D,C], CPU/GPU handled by torch automatically

    # 3) Encode images
    image_root = pathlib.Path(args.image_root)
    img_paths = [image_root / fn for fn in df_test["filename"].tolist()]
    img_feats, keep_mask = encode_images(model, preprocess, device, img_paths, args.batch_size)

    # Filter rows for which image exists
    keep_idx = [i for i, k in enumerate(keep_mask) if k]
    if len(keep_idx) != len(df_test):
        missing = len(df_test) - len(keep_idx)
        print(f"⚠️  Skipping {missing} missing files (not found under {image_root})")
    df_eval = df_test.iloc[keep_idx].reset_index(drop=True)
    y_true = df_eval["class_id"].astype(int).to_numpy()

    if len(df_eval) == 0:
        raise RuntimeError("No images to evaluate. Check paths and filenames.")

    # 4) Similarities and predictions
    with torch.no_grad():
        sims = img_feats.to(device) @ class_emb_t.to(device)   # [N,C], cosine similarity
        # move back to CPU for metrics
        sims = sims.cpu()

    y_pred_top1 = sims.argmax(dim=1).numpy()
    acc1 = (y_pred_top1 == y_true).mean().item()
    # acc5 = topk_acc(sims, y_true, k=5)

    macro_prec = precision_score(y_true, y_pred_top1, average="macro", zero_division=0)
    macro_rec  = recall_score(y_true, y_pred_top1, average="macro", zero_division=0)

    print("\n=== Zero-shot SigLIP evaluation ===")
    print(f"Images evaluated: {len(df_eval)} | Classes: {num_cls}")
    print(f"Accuracy@1       : {acc1:.6f}")
    print(f"Accuracy@5       : {acc5:.6f}")
    print(f"Macro Precision  : {macro_prec:.6f}")
    print(f"Macro Recall     : {macro_rec:.6f}")

    # 5) Optional per-image predictions JSON
    if args.save_predictions:
        preds_out = []
        cls_by_id = dict(zip(df_map["id"].astype(int), df_map["name"]))
        inv_names = {i: n for i, n in enumerate(class_names)}  # index-aligned names
        for (fn, true_id), row in zip(df_eval[["filename", "class_id"]].itertuples(index=False, name=None),
                                      sims):
            top5_idx = row.topk(5).indices.tolist()
            preds_out.append({
                "filename": fn,
                "predicted_class_1": inv_names[top5_idx[0]],
                "predicted_class_5": [inv_names[i] for i in top5_idx],
                "correct_class": cls_by_id[int(true_id)],
                "true_id": int(true_id),
                "pred_id": int(top5_idx[0]),
            })
        out_p = pathlib.Path(args.save_predictions)
        out_p.write_text(json.dumps({"predictions": preds_out}, indent=2))
        print(f"✅ Wrote per-image predictions to {out_p}")

if __name__ == "__main__":
    main()
