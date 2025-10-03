#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json, pathlib
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import precision_score, recall_score
import open_clip  # SigLIP backbone

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
    return (
        F.normalize(feat, dim=-1)
        if len(txt) > 1
        else F.normalize(feat, dim=-1).squeeze(0)
    )

# Metrics
def topk_acc(scores: torch.Tensor, target: np.ndarray, k: int = 1) -> float:
    target_t = torch.tensor(target, device=scores.device)
    return scores.topk(k, 1).indices.eq(target_t[:, None]).any(1).float().mean().item()

# Main
def main(args):
    # Data
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

    if "description" not in df.columns:
        raise ValueError("Input JSON must include a 'description' field per sample.")

    cls2idx = {c: i for i, c in enumerate(cls_list)}
    num_cls = len(cls_list)

    # Model
    model, preprocess, tokenizer, device = load_siglip(args.model_name)

    # Class embeddings
    print(f"Encoding {num_cls} class prototypes …")
    with torch.no_grad():
        domain_emb = encode_text(
            model,
            tokenizer,
            [f"An image of a {c}, a military vehicle." for c in cls_list],
            device,
        )
        photo_emb = encode_text(
            model, tokenizer, [f"An image of a {c}" for c in cls_list], device
        )

        template = "A photo of a {}."
        class_cap_emb = []
        with open(args.caption_json) as f:
            gen = json.load(f)

        for cls in cls_list:
            if template in gen:
                caps = gen[template][cls]["captions"][: args.n_captions]
            else:
                caps = gen[cls]["captions"][: args.n_captions]

            if args.caption_json.startswith(cls):
                prompts = [f"An image of a {cap}" for cap in caps]
            else:
                prompts = [f"An image of a {cls}. {cap}" for cap in caps]                

            emb = encode_text(model, tokenizer, prompts, device).mean(0, keepdim=True)
            class_cap_emb.append(emb)
        class_cap_emb = torch.cat(class_cap_emb, dim=0)

    # Feature cache (assume image and description are always used)
    sample_img, sample_desc, y_true, valid_rows = [], [], [], []
    img_root = pathlib.Path(args.image_root)
    print("Pre-computing per-sample image/description features …")
    for row in df.itertuples(index=False):
        img_f = img_root / row.filename
        if not img_f.is_file():
            continue  # skip if image file missing

        sample_img.append(encode_image(model, preprocess, img_f, device))
        sample_desc.append(encode_text(model, tokenizer, row.description, device))

        y_true.append(cls2idx[row.correct_class])
        valid_rows.append(row.filename)

    y_true = np.asarray(y_true)
    num_samples = len(y_true)
    assert len(sample_img) == num_samples == len(sample_desc)

    # Alpha grid (template vs. caption embeddings)
    if args.alpha is not None:
        alpha_vals = [max(0.0, min(1.0, args.alpha))]
    else:
        a_lo = max(0.0, min(1.0, args.alpha_min))
        a_hi = max(a_lo, min(1.0, args.alpha_max))
        a_step = args.alpha_step
        n_steps = int(round((a_hi - a_lo) / a_step)) + 1
        alpha_vals = [round(a_lo + i * a_step, 2) for i in range(n_steps)]

    # Beta grid (image vs. description features)
    if args.beta is not None:
        beta_vals = [max(0.0, min(1.0, args.beta))]
    else:
        b_lo = max(0.0, min(1.0, args.beta_min))
        b_hi = max(b_lo, min(1.0, args.beta_max))
        b_step = args.beta_step
        n_b = int(round((b_hi - b_lo) / b_step)) + 1
        beta_vals = [round(b_lo + i * b_step, 2) for i in range(n_b)]

    print(f"Alpha sweep values: {alpha_vals}")
    print(f"Beta  sweep values: {beta_vals}")

    total_comb = len(alpha_vals) * len(beta_vals)
    print(f"Evaluating {total_comb:,} (alpha, beta) combinations …")

    # Evaluation
    results, sims_last = [], None
    for alpha in alpha_vals:
        template_emb = photo_emb if args.generic else domain_emb
        class_emb = F.normalize((1 - alpha) * template_emb + alpha * class_cap_emb, dim=-1)
        class_emb_t = class_emb.T

        for beta in beta_vals:
            sims_all, y_pred = [], []

            # Combine modalities per sample:
            # resulting_feature = (1 - beta) * image_feature + beta * description_feature
            for i in range(num_samples):
                x = F.normalize((1.0 - beta) * sample_img[i] + beta * sample_desc[i], dim=-1)
                sim = x @ class_emb_t
                sims_all.append(sim)
                y_pred.append(int(sim.argmax()))

            sims_all = torch.stack(sims_all)
            y_pred = np.asarray(y_pred)

            acc1 = (y_pred == y_true).mean()
            acc5 = topk_acc(sims_all, y_true, 5)
            prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_true, y_pred, average="macro", zero_division=0)

            results.append(
                {
                    "alpha": alpha,
                    "beta": beta,
                    "acc1": acc1,
                    "acc5": acc5,
                    "macro_precision": prec,
                    "macro_recall": rec,
                }
            )

            if sims_last is None or acc1 > sims_last[0]:
                sims_last = (acc1, sims_all, (alpha, beta))

    # Save grid results
    with open(args.results_json, "w") as f:
        json.dump(
            {
                "num_samples": num_samples,
                "num_combinations": len(results),
                "results": results,
            },
            f,
            indent=2,
        )
    print(f"✅  Saved full sweep results to '{args.results_json}'")

    # Report top-5
    top5 = sorted(
        results,
        key=lambda r: (
            r["acc1"],
            r["acc5"],
            r["macro_precision"],
            r["macro_recall"],
        ),
        reverse=True,
    )[:5]

    print("\n=== Top-5 (alpha, beta) combinations ===")
    for rk, r in enumerate(top5, 1):
        print(
            f"{rk:>2}. alpha={r['alpha']:.2f} | beta={r['beta']:.2f}  "
            f"| acc1={r['acc1']:.4f}, acc5={r['acc5']:.4f}, "
            f"prec={r['macro_precision']:.4f}, rec={r['macro_recall']:.4f}"
        )

    # Per-image prediction dump (best config)
    _, sims_best, best_params = sims_last
    alpha_b, beta_b = best_params
    print(
        f"\n📄  Writing per-image top-5 predictions for best combo "
        f"(alpha={alpha_b}, beta={beta_b})"
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

if __name__ == "__main__":
    DATASET = "YOUR_DATASET"

    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Data & model
    p.add_argument("--dataset", default=DATASET,
                   help="Dataset name token used to form default paths.")
    p.add_argument("--json", default=f"generated_captions/image-side/Qwen2.5-VL-7B/{DATASET}.json",
                   help="Prediction/description JSON")
    p.add_argument("--image_root", default="/path/to/images",
                   help="Directory with the images")
    p.add_argument("--model_name", choices=SIGLIP_HUB.keys(), default="SigLIP",
                   help="SigLIP checkpoint key (see SIGLIP_HUB)")
    p.add_argument("--class_mapping", default=f"artifacts/data/{DATASET}/class_mapping.csv",
                   help="CSV with `id,name` (optional)")
    p.add_argument("--test_split", default=f"artifacts/data/{DATASET}/test.csv",
                   help="CSV listing filenames in test fold")
    p.add_argument("--caption_json", type=str, default=f"generated_captions/class-side/{DATASET}.json",
                   help="JSON with generated captions")
    p.add_argument("--n_captions", type=int, default=5,
                   help="How many generated captions to use per class")
    p.add_argument("--generic", action="store_true",
                   help="Use generic class embeddings (template-only)")

    # Sweep config (alpha, beta)
    p.add_argument("--alpha", type=float, default=None,
                   help="FIXED alpha value in [0,1]. If set, disables alpha sweep")
    p.add_argument("--alpha_step", type=float, default=0.1,
                   help="Alpha-grid step size (ignored if --alpha given)")
    p.add_argument("--alpha_min", type=float, default=0.1,
                   help="Lower bound for alpha sweep (ignored if --alpha is given)")
    p.add_argument("--alpha_max", type=float, default=0.6,
                   help="Upper bound for alpha sweep (ignored if --alpha is given)")

    p.add_argument("--beta", type=float, default=None,
                   help="FIXED beta value in [0,1]. If set, disables beta sweep")
    p.add_argument("--beta_step", type=float, default=0.1,
                   help="Beta-grid step size (ignored if --beta given)")
    p.add_argument("--beta_min", type=float, default=0.1,
                   help="Lower bound for beta sweep (ignored if --beta is given)")
    p.add_argument("--beta_max", type=float, default=0.6,
                   help="Upper bound for beta sweep (ignored if --beta is given)")

    # Outputs
    p.add_argument("--results_json", default="sweep_results/siglip_alpha_beta_sweep_results.json",
                   help="Where to write the exhaustive grid results")
    p.add_argument("--pred_json", default="sweep_results/image_predictions_with_top5.json",
                   help="Where to write per-image prediction JSON for the best combo")

    args = p.parse_args()
    main(args)
