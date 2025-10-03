#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
zero_shot_wdys_pe.py

Pure-Python re-implementation of the WDYS zero-shot fusion idea using the
PerceptionEncoder (PE) family instead of CLIP.

It fuses class text embeddings (template/domain/caption-based) and evaluates
image-text similarity against a labeled test split.
"""

from __future__ import annotations
import argparse, json, pathlib, sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import precision_score, recall_score

import core.vision_encoder.pe as pe
import core.vision_encoder.transforms as transforms

# PerceptionEncoder model hub
PE_HUB = {
    "PE-L": "PE-Core-L14-336",
    "PE-B": "PE-Core-B16-224",
    "PE-G": "PE-Core-G14-448",
}

@torch.no_grad()
def load_pe(model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = pe.CLIP.from_config(model_name, pretrained=True).to(device).eval()
    preprocess = transforms.get_image_transform(model.image_size)
    _tok = transforms.get_text_tokenizer(model.context_length)

    def tokenizer(text):
        return _tok([text] if isinstance(text, str) else text)

    return model, preprocess, tokenizer, device

@torch.no_grad()
def encode_image(m, preprocess, img_f, device):
    img = preprocess(Image.open(img_f).convert("RGB"))[None].to(device)
    feat = m.encode_image(img)
    return F.normalize(feat, dim=-1)

@torch.no_grad()
def encode_text(m, tokenizer, txt, device):
    tok = tokenizer(txt).to(device)
    feat = m.encode_text(tok)
    return F.normalize(feat, dim=-1)

def topk_acc(scores: torch.Tensor, target: np.ndarray, k: int = 1) -> float:
    return scores.topk(k, 1).indices.eq(torch.tensor(target)[:, None]).any(1).float().mean().item()

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

    mname = PE_HUB[args.model_name]
    model, preprocess, tokenizer, device = load_pe(mname)

    LONG_DESCRIPTIONS = {
        "boxer": "Boxer. German 8x8 armored vehicle for transport and combat. Crew of 3-8. Size: 8x2x2m.",
        "BRDM-2": "BRDM-2. Russian 4x4 armored reconnaissance vehicle. Crew of 3. Size: 6x2x2m.",
        "BTR-80": "BTR-80. Russian 8x8 armored vehicle for transport and combat. Crew of 3-10. Size: 8x3x2m.",
        "fennek": "Fennek. German 4x4 armored reconnaissance vehicle. Crew of 3. Size: 6x2x2m.",
        "Fuchs": "Fuchs. German 6x6 armored vehicle for transport and reconnaissance. Crew of 9. Size: 6x2x2m.",
        "howitzer": "PhZ 2000. German self-propelled howitzer with 155mm gun. Crew of 5. Size: 11x4x3m.",
        "Leopard": "Leopard. German main battle tank with a 120mm gun. Crew of 4. Size: 10x4x3m.",
        "M1A2": "M1A2 Abrams. American main battle tank with a 120mm smoothbore gun. Crew of 4. Size: 9.77x3.66x2.44m.",
        "M109": "M109. American self-propelled howitzer with 155mm gun. Crew of 6. Size: 10x3x3m.",
        "MSTA": "MSTA. Russian self-propelled howitzer with 152mm gun. Crew of 5. Size: 12x3x3m.",
        "Patria": "Patria. Finnish 8x8 armored vehicle for transport and combat. Crew of 3-9. Size: 8x3x2m.",
        "T90": "T-90. Russian main battle tank with a 125mm gun. Crew of 3. Size: 9.63x3.78x2.22m.",
        "truck": "Truck. American 6x6 vehicle for transport and logistics. Crew of 2-6. Size: 7x2x2m.",
        "2S1": "2S1 Gvozdika. Soviet 122mm self-propelled howitzer. Crew of 4. Size: 7.26x2.85x2.73m.",
        "2S3": "2S3 Akatsiya. Soviet 152mm self-propelled howitzer. Crew of 4. Size: 8.4x3.25x3.0m.",
        "BMP-1": "BMP-1. Soviet amphibious tracked infantry fighting vehicle. Crew of 3 + 8 passengers. Size: 6.74x2.94x2.06m.",
        "T-62": "T-62. Soviet main battle tank with a 115mm smoothbore gun. Crew of 4. Size: 9.34x3.30x2.40m.",
        "T-72": "T-72. Soviet main battle tank with a 125mm smoothbore gun. Crew of 3. Size: 9.53x3.59x2.23m.",
    }

    print(f"Encoding {num_cls} class prototypes …")
    name_emb        = encode_text(model, tokenizer, cls_list, device)
    photo_emb       = encode_text(model, tokenizer, [f"A photo of a {c}" for c in cls_list], device)
    domain_emb      = encode_text(model, tokenizer, [f"An image of a {c}, a military vehicle" for c in cls_list], device)
    description_emb = encode_text(model, tokenizer, [LONG_DESCRIPTIONS.get(c, f"An image of a {c}") for c in cls_list], device)

    with open(args.caption_json, "r") as f:
        gen = json.load(f)

    template = "A photo of a {}."
    class_cap_emb = []
    for cls in cls_list:
        caps = gen[template][cls]["captions"][: args.n_captions]
        prompts = [f"An image of a {cls}. {c}" for c in caps]
        E = encode_text(model, tokenizer, prompts, device)
        class_cap_emb.append(E.mean(dim=0, keepdim=True))
    class_cap_emb = torch.cat(class_cap_emb, dim=0)

    class_emb = F.normalize(domain_emb * 0.6 + class_cap_emb * 0.4, dim=-1)

    y_true, y_pred, sims_all = [], [], []

    uses_desc = args.use_description and "description" in df.columns
    uses_pred = args.use_prediction and "predicted_class" in df.columns
    if args.use_description and not uses_desc:
        print("⚠️  --use_description ignored (column 'description' missing).", file=sys.stderr)
    if args.use_prediction and not uses_pred:
        print("⚠️  --use_prediction ignored (column 'predicted_class' missing).", file=sys.stderr)

    w_img, w_desc, w_pred = args.w_img, args.w_desc, args.w_pred

    missing_images = 0
    for row in df.itertuples(index=False):
        img_f = pathlib.Path(args.image_root) / row.filename
        if not img_f.exists():
            missing_images += 1
            continue

        feats = []
        if args.use_image:
            feats.append(w_img * encode_image(model, preprocess, img_f, device))
        if uses_desc:
            feats.append(w_desc * encode_text(model, tokenizer, row.description, device))
        if uses_pred:
            feats.append(w_pred * encode_text(model, tokenizer, row.predicted_class, device))

        if not feats:
            raise ValueError("No modality chosen! Enable at least one of --use_* flags.")

        x = F.normalize(sum(feats), dim=-1)
        sim = x @ class_emb.T

        sims_all.append(sim.squeeze(0).cpu())
        y_true.append(cls2idx[row.correct_class])
        y_pred.append(int(sim.argmax(1)))

    sims_all = torch.stack(sims_all)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc1 = (y_pred == y_true).mean()
    acc5 = topk_acc(sims_all, y_true, 5)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)

    print("\n=== Zero-shot WDYS-style results (PerceptionEncoder) ===")
    print(f"Samples           : {(len(df)-missing_images):,}")
    print(f"Classes           : {num_cls}")
    print(f"acc@1             : {acc1:.4f}")
    print(f"acc@5             : {acc5:.4f}")
    print(f"macro precision   : {prec:.4f}")
    print(f"macro recall      : {rec:.4f}")

if __name__ == "__main__":
    DATASET = "YOUR_DATASET"

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--json",         default="data/image_predictions/top1_desc.json",
                   help="Prediction/description JSON")
    p.add_argument("--image_root",   default="/path/to/images",
                   help="Directory with the images")
    p.add_argument("--model_name",   choices=PE_HUB.keys(), default="PE-L",
                   help="PerceptionEncoder key")
    p.add_argument("--class_mapping",default=f"artifacts/data/{DATASET}/class_mapping.csv",
                   help="CSV with `id,name` (optional)")
    p.add_argument("--test_split",   default=f"artifacts/data/{DATASET}/test.csv",
                   help="CSV listing filenames in test fold")
    p.add_argument("--caption_json", type=str,
                   default=f"generated_captions/class-side/{DATASET}.json",
                   help="JSON with generated captions")
    p.add_argument("--n_captions", type=int, default=5,
                   help="How many generated captions to use per class")

    # modality switches
    p.add_argument("--use_image",       action="store_true",
                   help="Include image features")
    p.add_argument("--use_description", action="store_true",
                   help="Include LLM description features")
    p.add_argument("--use_prediction",  action="store_true",
                   help="Include initial LLM-prediction features")

    # modality weights
    p.add_argument("--w_img",  type=float, default=1.0,
                   help="Weight for image feature")
    p.add_argument("--w_desc", type=float, default=1.0,
                   help="Weight for description feature")
    p.add_argument("--w_pred", type=float, default=1.0,
                   help="Weight for LLM prediction feature")

    args = p.parse_args()
    main(args)
