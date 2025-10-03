#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
img2txt_fewshot_generation.py

Generate one or multiple captions per test image using Qwen-2.5-VL and class-specific
few-shot examples (drawn from a captions JSON).

Example
-------
python gen_captions_with_qwen.py \
       --image_root /path/to/images \
       --test_csv  artifacts/data/YOUR_DATASET/test.csv \
       --class_mapping artifacts/data/YOUR_DATASET/class_mapping.csv \
       --captions_json generated_captions/class-side/YOUR_DATASET.json \
       --out_json     generated_captions/image-side/Qwen2.5-VL-7B/YOUR_DATASET.json \
       --n_examples   3
"""

from __future__ import annotations
import argparse, json, pathlib, re, random
from typing import List

import torch
from PIL import Image
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# 1.  Load / wrap Qwen-2.5-VL                                                 #
# --------------------------------------------------------------------------- #
def load_vlm(device: torch.device):
    """
    Returns a callable: generate(image: PIL.Image, examples: List[str]) -> str
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print("🔄  Loading Qwen-2.5-VL-7B-Instruct …")
    model = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("✅  Qwen-VL ready")

    @torch.no_grad()
    def _generate(image: Image.Image, fewshot: List[str]) -> str:
        prompt = (
            "You are a military vehicle expert.\n\n"
            "We have the following example captions:\n"
            + "\n".join(f"- {ex}" for ex in fewshot)
            + "\n\nUsing the same style and level of detail, "
              "describe the vehicle in the image below. "
              "Return ONLY the generated description."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            gen_ids = model.generate(**inputs, max_new_tokens=256)
        decoded = processor.batch_decode(
            [g[len(i):] for g, i in zip(gen_ids, inputs.input_ids)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return decoded.strip(" \"'\n")

    return _generate


def load_vlm_extended(device: torch.device, num_captions: int = 10, max_words: int = 20):
    """
    Returns a callable: generate(image: PIL.Image, examples: List[str]) -> List[str]

    The callable produces `num_captions` (default: 10) distinct captions, each
    no longer than `max_words` words.
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    from typing import List
    import torch

    print("🔄  Loading Qwen-2.5-VL-7B-Instruct …")
    model = (
        Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("✅  Qwen-VL ready")

    @torch.no_grad()
    def _generate(image: Image.Image, fewshot: List[str]) -> List[str]:
        prompt = (
            "You are a an expert in describing histopathologic scans of lymph node sections, helping in identifying whether it contains a tumor or not.\n\n"
            "Here are example captions:\n"
            + "\n".join(f"- {ex}" for ex in fewshot)
            + "\n\nUsing the same style and level of detail, "
            "describe the section in the image below. Return one line."
            "Focus on distinctive features that could help conclude whether this "
            "section contains a tumor or not, helping to differentiate this "
            "section from *similar* ones **and** from "
            "dissimilar sections."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": prompt},
                ],
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            gen_ids = model.generate(**inputs, max_new_tokens=256)
        decoded = processor.batch_decode(
            [g[len(i):] for g, i in zip(gen_ids, inputs.input_ids)],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        lines = [ln.lstrip("- •*").strip() for ln in decoded.splitlines()]
        if len(lines) <= 1:
            cap = decoded.lstrip("- •*").strip()
            return [" ".join(cap.split())]

        uniq_caps = []
        for cap in lines:
            if not cap or cap in uniq_caps:
                continue
            words = cap.split()
            uniq_caps.append(" ".join(words))
            if len(uniq_caps) == num_captions:
                break

        return uniq_caps

    return _generate

# --------------------------------------------------------------------------- #
# 2.  Utilities                                                               #
# --------------------------------------------------------------------------- #
def strip_class_prefix(caption: str, cls_name: str) -> str:
    """
    Remove 'CLASS.' or 'CLASS :' prefix, case-insensitive.
    """
    pattern = rf"^{re.escape(cls_name)}\s*[\.:]\s*"
    return re.sub(pattern, "", caption.strip(), flags=re.IGNORECASE)


def pick_examples(caps: List[str], n: int, cls_name: str) -> List[str]:
    cleaned = [strip_class_prefix(c, cls_name) for c in caps]
    random.shuffle(cleaned)
    return cleaned[:n]

# --------------------------------------------------------------------------- #
# 3.  Main                                                                    #
# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generate_caption = load_vlm_extended(device, num_captions=args.num_captions)

    # -- data -----------------------------------------------------------------
    df_test = pd.read_csv(args.test_csv)               # filename, class_id
    df_map  = pd.read_csv(args.class_mapping)          # id, name
    id2name = dict(zip(df_map.id, df_map.name))

    with open(args.captions_json) as f:
        caps_nested = json.load(f)

    class_caps_raw = {}
    for cls, data in caps_nested.items():
        class_caps_raw.setdefault(cls, {"captions": []})
        class_caps_raw[cls]["captions"].extend(data["captions"])

    all_captions = []
    for cls, info in class_caps_raw.items():
        cleaned = [strip_class_prefix(c, cls) for c in info["captions"]]
        all_captions.extend(cleaned)

    random.shuffle(all_captions)
    fewshot_all = all_captions[:args.n_examples]

    # -- iterate --------------------------------------------------------------
    predictions = []
    predictions_split = []
    img_root = pathlib.Path(args.image_root)

    df_exist = df_test[df_test["filename"].apply(lambda fn: (img_root / fn).is_file())]

    for filename, cls_id in tqdm(
        df_exist.itertuples(index=False, name=None),
        total=len(df_exist),
        desc="Captioning"
    ):
        cls_name = id2name[cls_id]
        img_path = img_root / filename
        image    = Image.open(img_path).convert("RGB")

        caption = generate_caption(image, fewshot_all)

        predictions.append(
            {
                "filename": filename,
                "correct_class": cls_name,
                "predicted_class": cls_name,
                "description": caption,
            }
        )

        predictions_split.append(
            {
                "filename": filename,
                "correct_class": cls_name,
                "predicted_class": cls_name,
                "description": caption,
            }
        )

    # -- save -----------------------------------------------------------------
    out = {"predictions": predictions}
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅  Wrote {len(predictions):,} captions to {args.out_json}")

    split_out_json = f"generated_captions/image-side/Qwen2.5-VL-7B/{args.dataset}-multiple.json"
    out_split = {"predictions": predictions_split}
    with open(split_out_json, "w") as f:
        json.dump(out_split, f, indent=2)
    print(f"✅  Wrote {len(predictions_split):,} split captions to {split_out_json}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    DATASET = "YOUR_DATASET"

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        default=DATASET,
        help="Dataset name token used for default paths and filenames."
    )
    parser.add_argument(
        "--image_root",
        default="/path/to/images",
        help="Directory containing test images"
    )
    parser.add_argument(
        "--test_csv",
        default=f"artifacts/data/{DATASET}/test.csv",
        help="CSV with columns filename,class_id"
    )
    parser.add_argument(
        "--class_mapping",
        default=f"artifacts/data/{DATASET}/class_mapping.csv",
        help="CSV with columns id,name"
    )
    parser.add_argument(
        "--captions_json",
        default=f"generated_captions/class-side/{DATASET}.json",
        help="JSON with class-wise example captions"
    )
    parser.add_argument(
        "--out_json",
        default=f"generated_captions/image-side/Qwen2.5-VL-7B/{DATASET}.json",
        help="Where to write output JSON"
    )
    parser.add_argument(
        "--n_examples",
        type=int,
        default=3,
        help="How many few-shot captions in total (randomly picked from all classes)"
    )
    parser.add_argument(
        "--num_captions",
        type=int,
        default=5,
        help="How many captions to generate per image"
    )
    args = parser.parse_args()
    main(args)
