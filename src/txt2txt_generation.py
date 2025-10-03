#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate captions for each class in the 'military_vehicles' dataset using an LLM
and (optionally) score them with a selected text encoder.

Pipeline:
1) Query the LLM for a comprehensive attribute list useful for identifying images
   of military vehicles (tanks, APCs, armored fighting vehicles).
2) For each vehicle class, ask the LLM to produce short, diverse sentences that
   assign values to those attributes.

Defaults are set so users can simply run the script and fill in arguments as needed.

Outputs:
- Captions are saved to JSON under: generated_captions/class-side

Notes:
- The default LLM is GPT-4o (OpenAI).
- The encoder backend can be selected via --encoder_backend {perception,siglip,none}
  • perception -> loads PerceptionEncoder from utils.perceptionEncoder.core.vision_encoder
  • siglip -> (optional) user-provided SigLIP loader; if unavailable, falls back to default scores
  • none -> skip encoder scoring (every caption gets default score 0.7)

Environment:
- Requires OPENAI_API_KEY to be set (e.g., in a .env file or environment).

Example usage:
---------------
$ python txt2txt_generation.py
$ python txt2txt_generation.py --num_captions 30 --temperature 0.5 --encoder_backend none
$ python txt2txt_generation.py --model gpt-4o --encoder_backend perception
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import pandas as pd
from tqdm import tqdm
import torch
from dotenv import load_dotenv
from openai import OpenAI

# ────────────────────────────────────────────────────────────────────────────
# Dataset registry (only 'military_vehicles' retained as requested)
# ────────────────────────────────────────────────────────────────────────────

DATASET_CFG = {
    "military_vehicles": {
        "classname_file": "artifacts/data/military_vehicles/class_mapping.csv",
    },
}

# Default score when no encoder similarity is available
DEFAULT_SIM_SCORE = 0.7


# ────────────────────────────────────────────────────────────────────────────
# LLM interface
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    model_name: str = "gpt-4o"
    temperature: float = 0.7
    max_retries: int = 3


def _require_env_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found – set it in your environment or .env")
    return api_key


def build_llm_client() -> OpenAI:
    api_key = _require_env_key()
    return OpenAI(api_key=api_key)


def llm_generate(client: OpenAI, cfg: LLMConfig, prompt: str) -> str:
    """
    Call the chat model with a single-user prompt, return the raw text content.
    Retries a few times on transient failures.
    """
    last_err: Optional[Exception] = None
    for _ in range(cfg.max_retries):
        try:
            resp = client.chat.completions.create(
                model=cfg.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=cfg.temperature,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 – opaque SDK exceptions possible
            last_err = e
    raise RuntimeError(f"LLM generation failed after {cfg.max_retries} retries: {last_err}")


# ────────────────────────────────────────────────────────────────────────────
# Encoder loading and similarity computation
# ────────────────────────────────────────────────────────────────────────────

@dataclass
class EncoderBundle:
    encoder: object | None
    tok_fn: Callable[[str | List[str]], torch.Tensor] | None


def load_perception_encoder(device: torch.device) -> EncoderBundle:
    """
    Load PerceptionEncoder from the new path:
    utils.perceptionEncoder.core.vision_encoder
    Returns a simple bundle with encoder + tokenizer function.

    If the import fails, we gracefully fall back to no encoder (None).
    """
    try:
        # Adjust these imports to your package’s actual API if needed.
        from utils.perceptionEncoder.core import vision_encoder as ve  # type: ignore
        from utils.perceptionEncoder.core import transforms as ptrans  # type: ignore

        print("⏬ Loading PerceptionEncoder PE-Core-L14-336 …")
        enc = ve.CLIP.from_config("PE-Core-L14-336", pretrained=True).to(device).eval()
        base_tok = ptrans.get_text_tokenizer(enc.context_length)

        def tok_fn(text: str | List[str]):
            return base_tok([text] if isinstance(text, str) else text)

        return EncoderBundle(encoder=enc, tok_fn=tok_fn)

    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Could not load PerceptionEncoder. Falling back to default scores. Reason: {e}")
        return EncoderBundle(encoder=None, tok_fn=None)


def load_siglip_encoder(_device: torch.device) -> EncoderBundle:
    """
    Optional: Load SigLIP if your environment provides it.
    For now, this function returns a no-op bundle and gracefully falls back.
    You can implement your SigLIP loader here later.
    """
    try:
        # Placeholder hook for user implementation.
        # If you have a SigLIP text encoder, load it here and return EncoderBundle.
        raise NotImplementedError("SigLIP loader not implemented in this template.")
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ SigLIP not available. Falling back to default scores. Reason: {e}")
        return EncoderBundle(encoder=None, tok_fn=None)


def compute_caption_similarities(
    captions: List[str],
    classname: str,
    bundle: EncoderBundle,
    device: torch.device,
) -> List[float]:
    """
    If an encoder is available, compute cosine similarity between each caption and the classname.
    Otherwise, return DEFAULT_SIM_SCORE for each caption.
    """
    if bundle.encoder is None or bundle.tok_fn is None:
        return [DEFAULT_SIM_SCORE for _ in captions]

    with torch.no_grad():
        cls_tok = bundle.tok_fn([classname]).to(device)
        cap_tok = bundle.tok_fn(captions).to(device)
        enc = bundle.encoder

        cls_emb = enc.encode_text(cls_tok)
        cap_emb = enc.encode_text(cap_tok)

        cls_emb = torch.nn.functional.normalize(cls_emb, dim=-1)
        cap_emb = torch.nn.functional.normalize(cap_emb, dim=-1)

        sims = (cap_emb @ cls_emb.T).squeeze(1).cpu().tolist()
        return sims


# ────────────────────────────────────────────────────────────────────────────
# Prompt builders and parsing helpers
# ────────────────────────────────────────────────────────────────────────────

ATTRIBUTE_PROMPT = (
    "What are the attributes useful for identifying images of military vehicles "
    "(tanks, APC, armoured fighting vehicle)? Provide a list of all attributes."
)

def build_caption_prompt(
    attributes_text: str,
    classname: str,
    num_captions: int,
) -> str:
    """
    Builds the second prompt that asks for short sentences providing attribute values
    for a given vehicle class.
    """
    return (
        f"Based on the following attribute list:\n"
        f"{attributes_text}\n\n"
        f"Provide short sentences with values of these attributes for the following vehicle class:\n"
        f"- {classname}\n\n"
        f"Give {num_captions} short sentences, making sure each sentence is different.\n"
        f"Return exactly {num_captions} distinct sentences, one per line, with no numbering or extra text."
    )


def parse_lines(s: str) -> List[str]:
    """
    Splits a block of text into non-empty, stripped lines.
    """
    lines = [ln.strip("-• \t") for ln in (s or "").splitlines()]
    return [ln for ln in lines if ln]


def ensure_n_items(items: List[str], n: int) -> List[str]:
    """
    Ensures a list has exactly n items by padding with empty strings or trimming.
    """
    items = items[:n]
    if len(items) < n:
        items += [""] * (n - len(items))
    return items


# ────────────────────────────────────────────────────────────────────────────
# Core generation routine
# ────────────────────────────────────────────────────────────────────────────

def generate_for_dataset(
    dataset_name: str,
    output_dir: str,
    llm_cfg: LLMConfig,
    num_captions: int,
    encoder_backend: str,
) -> str:
    """
    Runs the two-step LLM flow:
      1) Get attribute list
      2) For each class, produce short sentences with attribute values

    Writes a JSON file under output_dir and returns the JSON path.
    """
    if dataset_name not in DATASET_CFG:
        raise KeyError(
            f"Unknown dataset: {dataset_name!r}. Only 'military_vehicles' is supported."
        )

    # Prepare output folder
    os.makedirs(output_dir, exist_ok=True)

    # Load class names
    class_csv = DATASET_CFG[dataset_name]["classname_file"]
    class_df = pd.read_csv(class_csv)
    classnames: List[str] = class_df["name"].tolist()

    # LLM client
    client = build_llm_client()

    # Step 1: Attribute discovery
    print("🔎 Querying LLM for attribute list …")
    attributes_raw = llm_generate(client, llm_cfg, ATTRIBUTE_PROMPT)
    attributes_list = parse_lines(attributes_raw)
    if not attributes_list:
        print("⚠️ No attributes parsed; proceeding with raw attribute text.")
    attributes_text_for_prompt = attributes_raw if attributes_list else "attributes: (none parsed)"

    # Encoder setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if encoder_backend.lower() == "perception":
        bundle = load_perception_encoder(device)
        encoder_tag = "PerceptionEncoder"
    elif encoder_backend.lower() == "siglip":
        bundle = load_siglip_encoder(device)
        encoder_tag = "SigLIP"
    else:
        print("ℹ️ Encoder scoring disabled; using default similarity scores.")
        bundle = EncoderBundle(encoder=None, tok_fn=None)
        encoder_tag = "NoEncoder"

    # Output file name
    json_path = os.path.join(
        output_dir,
        f"{dataset_name}_{llm_cfg.model_name}_{encoder_tag}_captions.json",
    )

    results: dict[str, dict[str, List[str] | List[float] | str]] = {
        "_meta": {
            "dataset": dataset_name,
            "llm_model": llm_cfg.model_name,
            "encoder_backend": encoder_tag,
            "temperature": llm_cfg.temperature,
            "num_captions": num_captions,
            "attributes_prompt": ATTRIBUTE_PROMPT,
            "attributes_response_raw": attributes_raw,
        }  # keep raw for traceability
    }

    print(f"📝 Generating {num_captions} sentences for {len(classnames)} classes …")
    for classname in tqdm(classnames, desc="Classes"):
        prompt = build_caption_prompt(attributes_text_for_prompt, classname, num_captions)
        raw = llm_generate(client, llm_cfg, prompt)
        captions = ensure_n_items(parse_lines(raw), num_captions)

        sims = compute_caption_similarities(captions, classname, bundle, device)

        # sort by similarity (if available) descending
        pairs = sorted(zip(captions, sims), key=lambda x: x[1], reverse=True)
        captions_sorted, sims_sorted = zip(*pairs) if pairs else ([], [])

        results[classname] = {
            "captions": list(captions_sorted),
            "similarities": list(sims_sorted),
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved results to {json_path}")
    return json_path


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-step LLM caption generation."
    )
    p.add_argument(
        "--dataset",
        default="military_vehicles",
        choices=["military_vehicles"],
        help="Dataset name.",
    )
    p.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM identifier (default: gpt-4o).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM sampling temperature (default: 0.7).",
    )
    p.add_argument(
        "--num_captions",
        type=int,
        default=20,
        help="Number of captions per class for step 2 (default: 20).",
    )
    p.add_argument(
        "--encoder_backend",
        default="siglip",
        choices=["perception", "siglip", "none"],
        help="Similarity backend: 'perception', 'siglip', or 'none' (default: siglip).",
    )
    p.add_argument(
        "--output_dir",
        default=os.path.join("generated_captions", "class-side"),
        help="Destination directory for JSON output (default: generated_captions/class-side).",
    )
    p.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Max retries for each LLM call (default: 3).",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()
    llm_cfg = LLMConfig(
        model_name=args.model,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    generate_for_dataset(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        llm_cfg=llm_cfg,
        num_captions=args.num_captions,
        encoder_backend=args.encoder_backend,
    )


if __name__ == "__main__":
    main()
