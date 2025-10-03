#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tsne_visualization.py

Create t-SNE visualizations to compare original and enriched embeddings
for both class prototypes and image representations, and optionally run
an evaluation sweep over α (class-prototype mix), τ (caption weighting),
and β (image-vs-description mix).
"""

from __future__ import annotations
import argparse, itertools, json, pathlib, sys, os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import open_clip
from sklearn.metrics import precision_score, recall_score
from typing import List, Tuple

# SigLIP helpers
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
def encode_image(model, preprocess, img_f: pathlib.Path, device):
    img = preprocess(Image.open(img_f).convert("RGB"))[None].to(device)
    feat = model.encode_image(img)
    return F.normalize(feat, dim=-1).squeeze(0)

@torch.no_grad()
def encode_text(model, tokenizer, txt, device):
    if isinstance(txt, str):
        txt = [txt]
    tok = tokenizer(txt).to(device)
    feat = model.encode_text(tok)
    return F.normalize(feat, dim=-1) if len(txt) > 1 else F.normalize(feat, dim=-1).squeeze(0)

def topk_acc(scores: torch.Tensor, target: np.ndarray, k: int = 1) -> float:
    target_t = torch.tensor(target, device=scores.device)
    return scores.topk(k, 1).indices.eq(target_t[:, None]).any(1).float().mean().item()

# -----------------------------------------------------------------------------
# 1) t-SNE helpers
# -----------------------------------------------------------------------------
def visualize_embeddings_tsne(args, alpha=0.5, temp=0.1, beta=0.0):
    """
    Creates t-SNE visualizations comparing original and enriched embeddings.

    feature = (1 - beta) * image_feature + beta * description_feature
    beta = 0 → only image; beta = 1 → only description.
    """
    print("Preparing t-SNE visualization...")
    model, preprocess, tokenizer, device = load_siglip(args.model_name)

    if args.class_mapping:
        cls_list = pd.read_csv(args.class_mapping).sort_values("id")["name"].tolist()
    else:
        with open(args.json) as f:
            recs = json.load(f)["predictions"]
        df = pd.DataFrame(recs)
        cls_list = sorted(df["correct_class"].unique().tolist())

    num_cls = len(cls_list)
    cls2idx = {c: i for i, c in enumerate(cls_list)}

    with torch.no_grad():
        domain_emb = encode_text(
            model, tokenizer, [f"An image of a {c}, a military vehicle" for c in cls_list], device
        )
        photo_emb = encode_text(
            model, tokenizer, [f"An image of a {c}" for c in cls_list], device
        )

        template = "A photo of a {}."
        all_cap_embs = []
        with open(args.caption_json) as f:
            gen = json.load(f)

        for cls in cls_list:
            if template in gen:
                caps = gen[template][cls]["captions"][: args.n_captions]
            else:
                caps = gen[cls]["captions"][: args.n_captions]

            if args.caption_json.endswith("gpt-50-sentences.json"):
                prompts = [f"An image of a {cls}. {cap}" for cap in caps]
            else:
                prompts = [f"An image of a {cap}" for cap in caps]

            embs = encode_text(model, tokenizer, prompts, device)
            all_cap_embs.append(embs)

        class_cap_emb = []
        for idx, embs in enumerate(all_cap_embs):
            ref = domain_emb[idx]
            sims = (embs @ ref) / max(temp, 1e-5)
            w = torch.softmax(sims, dim=0)
            pooled = (w[:, None] * embs).sum(0, keepdim=True)
            class_cap_emb.append(F.normalize(pooled, dim=-1))
        class_cap_emb = torch.cat(class_cap_emb, dim=0)

        template_emb = photo_emb if args.generic else domain_emb
        class_emb = F.normalize((1 - alpha) * template_emb + alpha * class_cap_emb, dim=-1)

    img_root = pathlib.Path(args.image_root)
    with open(args.json) as f:
        recs = json.load(f)["predictions"]
    df = pd.DataFrame(recs)

    if args.test_split:
        keep = pd.read_csv(args.test_split)["filename"].tolist()
        df = df[df.filename.isin(keep)]

    sample_img, sample_desc, sample_enriched, sample_labels, class_colors = [], [], [], [], []

    print("Computing sample embeddings...")
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        img_f = img_root / row.filename
        if not img_f.is_file():
            continue

        img_emb = encode_image(model, preprocess, img_f, device)
        sample_img.append(img_emb.cpu().numpy())

        desc_emb = None
        if hasattr(row, "description"):
            if isinstance(row.description, list):
                desc_embs = encode_text(model, tokenizer, row.description, device)
                desc_emb = desc_embs.mean(0)
            else:
                desc_emb = encode_text(model, tokenizer, row.description, device)

        if desc_emb is not None:
            sample_desc.append(desc_emb.cpu().numpy())
            enriched = F.normalize((1.0 - beta) * img_emb + beta * desc_emb, dim=-1)
            sample_enriched.append(enriched.cpu().numpy())

        class_idx = cls2idx[row.correct_class]
        sample_labels.append(class_idx)
        class_colors.append(row.correct_class)

    domain_emb_np = domain_emb.cpu().numpy()
    photo_emb_np = photo_emb.cpu().numpy()
    class_cap_emb_np = class_cap_emb.cpu().numpy()
    class_emb_np = class_emb.cpu().numpy()
    sample_img_np = np.array(sample_img)

    have_desc = len(sample_enriched) > 0
    if have_desc:
        sample_desc_np = np.array(sample_desc)
        sample_enriched_np = np.array(sample_enriched)

    sample_labels_np = np.array(sample_labels)

    print("Computing t-SNE embeddings...")

    tsne_classes = TSNE(n_components=2, random_state=42, perplexity=min(30, max(2, num_cls - 1)))
    combined_class_embs = np.vstack([domain_emb_np, photo_emb_np, class_cap_emb_np, class_emb_np])
    combined_class_types = np.array(
        ["Domain"] * num_cls + ["Photo"] * num_cls + ["Caption"] * num_cls + ["Combined"] * num_cls
    )
    combined_class_names = np.tile(cls_list, 4)
    tsne_result_classes = tsne_classes.fit_transform(combined_class_embs)

    max_samples = min(2000, len(sample_img_np))
    indices = np.random.choice(len(sample_img_np), max_samples, replace=False)
    tsne_images = TSNE(n_components=2, random_state=42)

    if have_desc:
        combined_image_embs = np.vstack([sample_img_np[indices], sample_enriched_np[indices]])
        combined_image_types = np.array(["Original"] * max_samples + ["Enriched"] * max_samples)
        combined_image_classes = np.tile([cls_list[i] for i in sample_labels_np[indices]], 2)
        tsne_result_images = tsne_images.fit_transform(combined_image_embs)
    else:
        tsne_result_images = tsne_images.fit_transform(sample_img_np[indices])
        combined_image_types = np.array(["Original"] * max_samples)
        combined_image_classes = np.array([cls_list[i] for i in sample_labels_np[indices]])

    print("Creating visualization plots...")

    plt.figure(figsize=(12, 10))
    palette = sns.color_palette("hsv", len(cls_list))
    class_palette = {cls: palette[i] for i, cls in enumerate(cls_list)}
    for embed_type in ["Domain", "Photo", "Caption", "Combined"]:
        mask = combined_class_types == embed_type
        sns.scatterplot(
            x=tsne_result_classes[mask, 0],
            y=tsne_result_classes[mask, 1],
            hue=combined_class_names[mask],
            palette=class_palette,
            style=combined_class_types[mask],
            s=100,
            alpha=0.7,
        )
    plt.title(f"t-SNE of Class Embeddings (alpha={alpha}, temp={temp})")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    os.makedirs("tsne_plots", exist_ok=True)
    plt.savefig(f"tsne_plots/tsne_class_embeddings_alpha{alpha}_temp{temp}.png", dpi=300)

    plt.figure(figsize=(12, 10))
    for embed_type in np.unique(combined_image_types):
        mask = combined_image_types == embed_type
        sns.scatterplot(
            x=tsne_result_images[mask, 0],
            y=tsne_result_images[mask, 1],
            hue=combined_image_classes[mask],
            palette=class_palette,
            style=combined_image_types[mask],
            s=50,
            alpha=0.7,
        )
    plt.title(f"t-SNE of Image Embeddings (beta={beta})")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"tsne_plots/tsne_image_embeddings_beta{beta}.png", dpi=300)

    print("t-SNE visualization completed!")

def create_original_vs_enriched_tsne(args, alpha=0.5, temp=0.1, beta=0.0):
    """
    Creates two t-SNE visualizations side by side:
    1) Original image embeddings
    2) Enriched embeddings with caption information using β-mix
    """
    print("Preparing side-by-side t-SNE visualization...")
    model, preprocess, tokenizer, device = load_siglip(args.model_name)

    if args.class_mapping:
        cls_list = pd.read_csv(args.class_mapping).sort_values("id")["name"].tolist()
    else:
        with open(args.json) as f:
            recs = json.load(f)["predictions"]
        df = pd.DataFrame(recs)
        cls_list = sorted(df["correct_class"].unique().tolist())

    num_cls = len(cls_list)
    cls2idx = {c: i for i, c in enumerate(cls_list)}

    with torch.no_grad():
        domain_emb = encode_text(
            model, tokenizer, [f"An image of a {c}, a military vehicle" for c in cls_list], device
        )

        template = "A photo of a {}."
        all_cap_embs = []
        with open(args.caption_json) as f:
            gen = json.load(f)

        for cls in cls_list:
            if template in gen:
                caps = gen[template][cls]["captions"][: args.n_captions]
            else:
                caps = gen[cls]["captions"][: args.n_captions]

            if args.caption_json.endswith("gpt-50-sentences.json"):
                prompts = [f"An image of a {cls}. {cap}" for cap in caps]
            else:
                prompts = [f"An image of a {cap}" for cap in caps]

            embs = encode_text(model, tokenizer, prompts, device)
            all_cap_embs.append(embs)

        class_cap_emb = []
        for idx, embs in enumerate(all_cap_embs):
            ref = domain_emb[idx]
            sims = (embs @ ref) / max(temp, 1e-5)
            w = torch.softmax(sims, dim=0)
            pooled = (w[:, None] * embs).sum(0, keepdim=True)
            class_cap_emb.append(F.normalize(pooled, dim=-1))
        class_cap_emb = torch.cat(class_cap_emb, dim=0)

        class_emb = F.normalize((1 - alpha) * domain_emb + alpha * class_cap_emb, dim=-1)

    img_root = pathlib.Path(args.image_root)
    with open(args.json) as f:
        recs = json.load(f)["predictions"]
    df = pd.DataFrame(recs)

    if args.test_split:
        keep = pd.read_csv(args.test_split)["filename"].tolist()
        df = df[df.filename.isin(keep)]

    sample_img, sample_enriched, sample_labels = [], [], []
    print("Computing sample embeddings...")
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        img_f = img_root / row.filename
        if not img_f.is_file():
            continue

        img_emb = encode_image(model, preprocess, img_f, device)
        sample_img.append(img_emb.cpu().numpy())

        desc_emb = None
        if hasattr(row, "description"):
            if isinstance(row.description, list):
                desc_embs = encode_text(model, tokenizer, row.description, device)
                desc_emb = desc_embs.mean(0)
            else:
                desc_emb = encode_text(model, tokenizer, row.description, device)

        if desc_emb is not None:
            enriched = F.normalize((1.0 - beta) * img_emb + beta * desc_emb, dim=-1)
            sample_enriched.append(enriched.cpu().numpy())

        class_idx = cls2idx[row.correct_class]
        sample_labels.append(class_idx)

    sample_img_np = np.array(sample_img)
    have_enriched = len(sample_enriched) > 0
    if have_enriched:
        sample_enriched_np = np.array(sample_enriched)
    sample_labels_np = np.array(sample_labels)

    n_classes = len(cls_list)
    palette = sns.color_palette("hsv", n_classes)
    class_palette = {cls: palette[i] for i, cls in enumerate(cls_list)}

    max_samples = min(2000, len(sample_img_np))
    indices = np.random.choice(len(sample_img_np), max_samples, replace=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    print("Computing t-SNE for original embeddings...")
    tsne_original = TSNE(n_components=2, random_state=42)
    tsne_result_original = tsne_original.fit_transform(sample_img_np[indices])
    original_classes = np.array([cls_list[i] for i in sample_labels_np[indices]])
    sns.scatterplot(
        x=tsne_result_original[:, 0],
        y=tsne_result_original[:, 1],
        hue=original_classes,
        palette=class_palette,
        s=50,
        alpha=0.7,
        ax=ax1,
    )
    ax1.set_title("Original Image Embeddings (No Enrichment)")
    ax1.set_xlabel("t-SNE dimension 1")
    ax1.set_ylabel("t-SNE dimension 2")

    if have_enriched:
        print("Computing t-SNE for enriched embeddings...")
        tsne_enriched = TSNE(n_components=2, random_state=42)
        tsne_result_enriched = tsne_enriched.fit_transform(sample_enriched_np[indices])
        sns.scatterplot(
            x=tsne_result_enriched[:, 0],
            y=tsne_result_enriched[:, 1],
            hue=original_classes,
            palette=class_palette,
            s=50,
            alpha=0.7,
            ax=ax2,
        )
        ax2.set_title(f"Enriched Image Embeddings (beta={beta})")
        ax2.set_xlabel("t-SNE dimension 1")
        ax2.set_ylabel("t-SNE dimension 2")

    handles, labels = ax2.get_legend_handles_labels() if have_enriched else ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.05), ncol=min(5, n_classes))
    ax1.get_legend().remove()
    if have_enriched:
        ax2.get_legend().remove()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    os.makedirs("tsne_plots", exist_ok=True)
    plt.savefig(f"tsne_plots/comparison_original_vs_enriched_beta{beta}.png", dpi=300, bbox_inches="tight")
    print("Side-by-side t-SNE visualization completed!")

# -----------------------------------------------------------------------------
# 2) Main evaluation sweep (α, τ, β)
# -----------------------------------------------------------------------------
def main(args):
    with open(args.json) as f:
        recs = json.load(f)["predictions"]
    df = pd.DataFrame(recs)

    if args.test_split:
        keep = pd.read_csv(args.test_split)["filename"].tolist()
        df = df[df.filename.isin(keep)]

    if args.class_mapping:
        cls_list = pd.read_csv(args.class_mapping).sort_values("id")["name"].tolist()
    else:
        cls_list = sorted(df["correct_class"].unique().tolist())

    cls2idx = {c: i for i, c in enumerate(cls_list)}
    num_cls = len(cls_list)

    model, preprocess, tokenizer, device = load_siglip(args.model_name)

    print(f"Encoding {num_cls} class prototypes …")
    with torch.no_grad():
        domain_emb = encode_text(
            model, tokenizer, [f"An image of a {c}, a military vehicle" for c in cls_list], device
        )
        photo_emb = encode_text(
            model, tokenizer, [f"An image of a {c}" for c in cls_list], device
        )

        template = "A photo of a {}."
        all_cap_embs = []
        with open(args.caption_json) as f:
            gen = json.load(f)

        for cls in cls_list:
            if template in gen:
                caps = gen[template][cls]["captions"][: args.n_captions]
            else:
                caps = gen[cls]["captions"][: args.n_captions]

            if args.caption_json.endswith("gpt-50-sentences.json"):
                prompts = [f"An image of a {cls}. {cap}" for cap in caps]
            else:
                prompts = [f"An image of a {cap}" for cap in caps]

            embs = encode_text(model, tokenizer, prompts, device)
            all_cap_embs.append(embs)

    uses_img = args.use_image
    uses_desc = args.use_description and "description" in df.columns
    if args.use_description and not uses_desc:
        print("⚠️  --use_description ignored (column 'description' missing).", file=sys.stderr)

    sample_img, sample_desc, y_true, valid_rows = [], [], [], []
    img_root = pathlib.Path(args.image_root)
    print("Pre-computing per-sample modality features …")
    for row in df.itertuples(index=False):
        img_f = img_root / row.filename
        if uses_img and not img_f.is_file():
            continue

        img_vec = encode_image(model, preprocess, img_f, device) if uses_img else None
        desc_vec = None
        if uses_desc:
            if args.json.endswith(f"{DATASET}.json"):
                desc_vec = encode_text(model, tokenizer, row.description, device)
            else:
                desc_vec = encode_text(model, tokenizer, row.description, device).mean(0)

        sample_img.append(img_vec)
        sample_desc.append(desc_vec)
        y_true.append(cls2idx[row.correct_class])
        valid_rows.append(row.filename)

    y_true = np.asarray(y_true)
    num_samples = len(y_true)

    if args.alpha is not None:
        alpha_vals = [max(0.0, min(1.0, args.alpha))]
    else:
        a_lo = max(0.0, min(1.0, args.alpha_min))
        a_hi = max(a_lo, min(1.0, args.alpha_max))
        a_step = args.alpha_step
        n_steps = int(round((a_hi - a_lo) / a_step)) + 1
        alpha_vals = [round(a_lo + i * a_step, 2) for i in range(n_steps)]
    print(f"Alpha sweep values: {alpha_vals}")

    if args.temp is not None:
        temp_vals = [max(1e-5, args.temp)]
    else:
        t_lo = max(1e-5, min(args.temp_min, args.temp_max))
        t_hi = max(t_lo, max(args.temp_min, args.temp_max))
        t_step = args.temp_step
        n_t = int(round((t_hi - t_lo) / t_step)) + 1
        temp_vals = [round(t_lo + i * t_step, 3) for i in range(n_t)]
    print(f"Temperature sweep values: {temp_vals}")

    if args.beta is not None:
        beta_vals = [max(0.0, min(1.0, args.beta))]
    else:
        b_lo = max(0.0, min(1.0, args.beta_min))
        b_hi = max(b_lo, min(1.0, args.beta_max))
        b_step = args.beta_step
        n_b = int(round((b_hi - b_lo) / b_step)) + 1
        beta_vals = [round(b_lo + i * b_step, 2) for i in range(n_b)]
    print(f"Beta sweep values: {beta_vals}")

    total_comb = len(alpha_vals) * len(temp_vals) * len(beta_vals)
    print(f"Evaluating {total_comb:,} weight / alpha / temp combinations …")

    results, sims_last = [], None
    from tqdm import tqdm as _tqdm
    pbar = _tqdm(total=total_comb, desc="Evaluating")

    for temp in temp_vals:
        class_cap_emb = []
        for idx, embs in enumerate(all_cap_embs):
            ref = domain_emb[idx]
            sims = (embs @ ref) / temp
            w = torch.softmax(sims, dim=0)
            pooled = (w[:, None] * embs).sum(0, keepdim=True)
            class_cap_emb.append(F.normalize(pooled, dim=-1))
        class_cap_emb = torch.cat(class_cap_emb, dim=0)

        for alpha in alpha_vals:
            template_emb = photo_emb if args.generic else domain_emb
            class_emb = F.normalize((1 - alpha) * template_emb + alpha * class_cap_emb, dim=-1)
            class_emb_t = class_emb.T

            for beta in beta_vals:
                sims_all, y_pred = [], []

                for i in range(num_samples):
                    img_vec = sample_img[i]
                    desc_vec = sample_desc[i]

                    if img_vec is None and desc_vec is None:
                        continue
                    elif img_vec is not None and desc_vec is not None:
                        x = F.normalize((1.0 - beta) * img_vec + beta * desc_vec, dim=-1)
                    elif img_vec is not None:
                        x = img_vec
                    else:
                        x = desc_vec

                    sim = x @ class_emb_t
                    sims_all.append(sim)
                    y_pred.append(int(sim.argmax()))

                sims_all = torch.stack(sims_all)
                y_pred = np.asarray(y_pred)

                acc1 = (y_pred == y_true[: len(y_pred)]).mean()
                acc5 = topk_acc(sims_all, y_true[: len(y_pred)], 5)
                prec = precision_score(y_true[: len(y_pred)], y_pred, average="macro", zero_division=0)
                rec = recall_score(y_true[: len(y_pred)], y_pred, average="macro", zero_division=0)

                results.append(
                    {
                        "alpha": alpha,
                        "temp": temp,
                        "beta": beta,
                        "acc1": acc1,
                        "acc5": acc5,
                        "macro_precision": prec,
                        "macro_recall": rec,
                    }
                )

                if sims_last is None or acc1 > sims_last[0]:
                    sims_last = (acc1, sims_all, (alpha, temp, beta))

                pbar.update(1)

    pbar.close()

    with open(args.results_json, "w") as f:
        json.dump(
            {"num_samples": num_samples, "num_combinations": len(results), "results": results},
            f,
            indent=2,
        )
    print(f"✅  Saved full sweep results to '{args.results_json}'")

    top5 = sorted(
        results,
        key=lambda r: (r["acc1"], r["acc5"], r["macro_precision"], r["macro_recall"]),
        reverse=True,
    )[:5]

    print(f"\n=== Top-5 (temp, alpha, beta) n = {args.n_captions} ===")
    for rk, r in enumerate(top5, 1):
        print(
            f"{rk:>2}. τ={r['temp']:.3f} α={r['alpha']:.2f} β={r['beta']:.2f}  "
            f"| acc1={r['acc1']:.4f}, acc5={r['acc5']:.4f}, "
            f"prec={r['macro_precision']:.4f}, rec={r['macro_recall']:.4f}"
        )

    _, sims_best, best_params = sims_last
    alpha_b, temp_b, beta_b = best_params
    print(
        f"\n📄  Writing per-image top-5 predictions for best combo "
        f"(alpha={alpha_b}, temp={temp_b}, beta={beta_b})"
    )

    predictions_out = []
    for fname, cls_true_idx, sims in zip(valid_rows, y_true, sims_best):
        top5_idx = sims.topk(5).indices.tolist()
        predictions_out.append(
            {
                "filename": fname,
                "predicted_class_1": cls_list[top5_idx[0]],
                "predicted_class_5": [cls_list[i] for i in top5_idx],
                "correct_class": cls_list[cls_true_idx],
            }
        )

    with open(args.pred_json, "w") as f:
        json.dump({"predictions": predictions_out}, f, indent=2)
    print(f"✅  Wrote {len(predictions_out):,} predictions to {args.pred_json}")

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    DATASET = "military_vehicles"

    # Data & model
    p.add_argument("--json", default=f"generated_captions/image-side/Qwen2.5-VL-7B/{DATASET}.json",
                   help="Prediction/description JSON")
    p.add_argument("--image_root", default=f"/path/to/images/{DATASET}",
                   help="Directory with the images")
    p.add_argument("--model_name", choices=SIGLIP_HUB.keys(), default="SigLIP",
                   help="SigLIP checkpoint key")
    p.add_argument("--class_mapping", default=f"artifacts/data/{DATASET}/class_mapping.csv",
                   help="CSV with `id,name` (optional)")
    p.add_argument("--test_split", default=f"artifacts/data/{DATASET}/test.csv",
                   help="CSV listing filenames in test fold")
    p.add_argument("--caption_json", type=str, default=f"generated_captions/class-side/{DATASET}.json",
                   help="JSON with generated captions")
    p.add_argument("--n_captions", type=int, default=5,
                   help="How many generated captions to use per class")
    p.add_argument("--generic", action="store_true",
                   help="Use generic class embeddings (photo prompts)")

    # Alpha sweep
    p.add_argument("--alpha", type=float, default=None,
                   help="FIXED alpha value (0-1). If set, disables alpha sweep")
    p.add_argument("--alpha_step", type=float, default=0.1,
                   help="Alpha-grid step size (ignored if --alpha given)")
    p.add_argument("--alpha_min", type=float, default=0.1,
                   help="Lower bound for alpha sweep (ignored if --alpha is given)")
    p.add_argument("--alpha_max", type=float, default=0.6,
                   help="Upper bound for alpha sweep (ignored if --alpha is given)")

    # Temperature sweep (caption weighting)
    p.add_argument("--temp", type=float, default=None,
                   help="FIXED soft-max temperature (>0). If set, disables temp sweep")
    p.add_argument("--temp_step", type=float, default=0.02,
                   help="Temperature sweep step size")
    p.add_argument("--temp_min", type=float, default=0.05,
                   help="Lower bound for temperature sweep")
    p.add_argument("--temp_max", type=float, default=0.30,
                   help="Upper bound for temperature sweep")

    # Beta sweep (image vs description)
    p.add_argument("--beta", type=float, default=None,
                   help="FIXED beta value (0-1). If set, disables beta sweep")
    p.add_argument("--beta_step", type=float, default=0.1,
                   help="Beta-grid step size (ignored if --beta given)")
    p.add_argument("--beta_min", type=float, default=0.0,
                   help="Lower bound for beta sweep (ignored if --beta is given)")
    p.add_argument("--beta_max", type=float, default=1.0,
                   help="Upper bound for beta sweep (ignored if --beta is given)")

    # Outputs
    p.add_argument("--results_json", default="sweep_results/siglip_alpha_temp_beta_sweep_results.json",
                   help="Where to write sweep results")
    p.add_argument("--pred_json", default="sweep_results/image_predictions_with_top5.json",
                   help="Where to write per-image prediction JSON for the best combo")

    # Visualization
    p.add_argument("--visualize_tsne", action="store_true",
                   help="Generate t-SNE visualizations")
    p.add_argument("--side_by_side", action="store_true",
                   help="Create side-by-side comparison of original vs enriched embeddings")
    p.add_argument("--vis_alpha", type=float, default=0.1,
                   help="Alpha value for t-SNE visualization")
    p.add_argument("--vis_temp", type=float, default=0.1,
                   help="Temperature for t-SNE visualization")
    p.add_argument("--vis_beta", type=float, default=0.2,
                   help="Beta value for t-SNE visualization (image vs description)")

    # Modality toggles
    p.add_argument("--use_image", action="store_true",
                   help="Include image features")
    p.add_argument("--use_description", action="store_true",
                   help="Include description-text features")

    args = p.parse_args()

    if args.visualize_tsne:
        visualize_embeddings_tsne(
            args,
            alpha=args.vis_alpha,
            temp=args.vis_temp,
            beta=args.vis_beta,
        )
        if args.side_by_side:
            create_original_vs_enriched_tsne(
                args,
                alpha=args.vis_alpha,
                temp=args.vis_temp,
                beta=args.vis_beta,
            )
    else:
        main(args)
