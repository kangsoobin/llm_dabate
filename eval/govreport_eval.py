#!/usr/bin/env python3
"""Evaluate Kanana base / SFT / GRPO adapters on GovReport summarization.

The official GovReport tar has two schemas:
- CRS: {summary, reports}
- GAO: {highlight, report}

This script samples the official test split, generates summaries with the same
prompt/decoding settings for each variant, and reports lightweight metrics that
need no extra packages: ROUGE-1/2/L F1, output length, compression, and
bigram repetition.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DEFAULT_DATA_DIR = BASE_DIR / "eval_data" / "govreport" / "gov-report"
DEFAULT_OUTPUT_DIR = BASE_DIR / "eval_outputs" / "govreport"

VARIANTS: dict[str, str | None] = {
    "base": None,
    "sft_left": "adapters/left",
    "sft_right": "adapters/right",
    "grpo_left": "adapters/left_grpo_v3_left_long_g9/checkpoint-112",
    "grpo_right": "adapters/right_grpo_v3_right_long_g8/checkpoint-120",
}


@dataclass
class Example:
    source: str
    report_id: str
    title: str
    document: str
    reference: str
    path: str

    @property
    def key(self) -> str:
        return f"{self.source}:{self.report_id}"


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def flatten_sections(obj: Any) -> list[str]:
    """Flatten GovReport section trees/lists into title-marked paragraphs."""
    chunks: list[str] = []
    if obj is None:
        return chunks
    if isinstance(obj, str):
        text = normalize_space(obj)
        return [text] if text else []
    if isinstance(obj, list):
        for item in obj:
            chunks.extend(flatten_sections(item))
        return chunks
    if isinstance(obj, dict):
        title = normalize_space(str(obj.get("section_title") or obj.get("title") or ""))
        if title:
            chunks.append(f"## {title}")
        for p in obj.get("paragraphs") or []:
            text = normalize_space(str(p))
            if text:
                chunks.append(text)
        for key in ("subsections", "children", "sections"):
            for child in obj.get(key) or []:
                chunks.extend(flatten_sections(child))
        return chunks
    return []


def load_json_example(path: Path) -> Example | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    source = path.parent.name
    report_id = str(data.get("id") or path.stem)
    title = normalize_space(str(data.get("title") or ""))

    if source == "crs":
        reference_parts = data.get("summary") or []
        doc_obj = data.get("reports")
    elif source == "gao":
        reference_parts = flatten_sections(data.get("highlight"))
        doc_obj = data.get("report")
    else:
        return None

    if isinstance(reference_parts, str):
        reference = normalize_space(reference_parts)
    else:
        reference = normalize_space("\n".join(str(x) for x in reference_parts))
    document = normalize_space("\n".join(flatten_sections(doc_obj)))

    if not reference or not document:
        return None
    return Example(source, report_id, title, document, reference, str(path.relative_to(BASE_DIR)))


def load_split_examples(data_dir: Path, split: str, limit: int | None, seed: int) -> list[Example]:
    split_dir = data_dir / "split_ids"
    examples: list[Example] = []
    for source in ("crs", "gao"):
        ids_path = split_dir / f"{source}_{split}.ids"
        if not ids_path.exists():
            raise FileNotFoundError(f"missing split file: {ids_path}")
        ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for report_id in ids:
            ex = load_json_example(data_dir / source / f"{report_id}.json")
            if ex is not None:
                examples.append(ex)

    rng = random.Random(seed)
    by_source: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_source[ex.source].append(ex)
    for bucket in by_source.values():
        rng.shuffle(bucket)

    if limit is None or limit >= len(examples):
        rng.shuffle(examples)
        return examples

    # Balanced CRS/GAO sample; this keeps small pilot runs from becoming CRS-only.
    sources = sorted(by_source)
    selected: list[Example] = []
    while len(selected) < limit and any(by_source.values()):
        for source in sources:
            if len(selected) >= limit:
                break
            if by_source[source]:
                selected.append(by_source[source].pop())
    return selected


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", (text or "").lower())


def rouge_n(pred_tokens: list[str], ref_tokens: list[str], n: int) -> dict[str, float]:
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return {"p": 0.0, "r": 0.0, "f": 0.0}
    pred = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
    ref = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
    overlap = sum((pred & ref).values())
    p = overlap / max(1, sum(pred.values()))
    r = overlap / max(1, sum(ref.values()))
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": p, "r": r, "f": f}


def lcs_len(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    # Keep memory O(min(n, m)). Reference/prediction summaries are short enough.
    if len(b) > len(a):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        left = 0
        for j, y in enumerate(b, 1):
            val = prev[j - 1] + 1 if x == y else max(prev[j], left)
            cur.append(val)
            left = val
        prev = cur
    return prev[-1]


def rouge_l(pred_tokens: list[str], ref_tokens: list[str]) -> dict[str, float]:
    lcs = lcs_len(pred_tokens, ref_tokens)
    p = lcs / max(1, len(pred_tokens))
    r = lcs / max(1, len(ref_tokens))
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"p": p, "r": r, "f": f}


def repetition_2(tokens: list[str]) -> float:
    if len(tokens) < 2:
        return 0.0
    bigrams = [tuple(tokens[i : i + 2]) for i in range(len(tokens) - 1)]
    return 1.0 - len(set(bigrams)) / max(1, len(bigrams))


def compute_metrics(prediction: str, reference: str, source_words: int) -> dict[str, float]:
    pred_tokens = word_tokens(prediction)
    ref_tokens = word_tokens(reference)
    r1 = rouge_n(pred_tokens, ref_tokens, 1)
    r2 = rouge_n(pred_tokens, ref_tokens, 2)
    rl = rouge_l(pred_tokens, ref_tokens)
    return {
        "rouge1_f": r1["f"],
        "rouge1_p": r1["p"],
        "rouge1_r": r1["r"],
        "rouge2_f": r2["f"],
        "rougeL_f": rl["f"],
        "pred_words": float(len(pred_tokens)),
        "ref_words": float(len(ref_tokens)),
        "compression": float(len(pred_tokens) / max(1, source_words)),
        "bigram_repetition": repetition_2(pred_tokens),
    }


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    numeric: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.get("metrics", {}).items():
            if isinstance(value, (int, float)):
                numeric[key].append(float(value))
    return {key: statistics.fmean(vals) for key, vals in sorted(numeric.items()) if vals}


def choose_document_tokens(tokenizer: Any, document: str, max_input_tokens: int) -> str:
    ids = tokenizer.encode(document, add_special_tokens=False)
    if len(ids) <= max_input_tokens:
        return document
    head = max_input_tokens // 2
    tail = max_input_tokens - head
    kept = ids[:head] + ids[-tail:]
    text = tokenizer.decode(kept, skip_special_tokens=True)
    return text + "\n\n[Note: The middle of the source report was omitted to fit the context budget.]"


def build_prompt(tokenizer: Any, ex: Example, max_input_tokens: int) -> str:
    doc = choose_document_tokens(tokenizer, ex.document, max_input_tokens)
    user = (
        "Summarize the following U.S. government report in English. "
        "Focus on the report purpose, major findings, evidence, and policy implications. "
        "Keep the summary factual, concise, and understandable to an informed citizen.\n\n"
        f"Title: {ex.title}\n"
        f"Report ID: {ex.report_id}\n\n"
        f"Report text:\n{doc}\n\nSummary:"
    )
    messages = [
        {"role": "system", "content": "You are a careful policy analyst and long-document summarization evaluator."},
        {"role": "user", "content": user},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return messages[0]["content"] + "\n\n" + messages[1]["content"]


def load_model_and_tokenizer(model_id: str, adapter_rel: str | None, precision: str, local_files_only: bool):
    from peft import PeftModel
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from runtime_utils import patch_deepseek_v3_moe_dtype_bug

    patch_deepseek_v3_moe_dtype_bug()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": {"": 0},
        "local_files_only": local_files_only,
    }
    if precision == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter_rel:
        adapter_path = Path(adapter_rel)
        if not adapter_path.is_absolute():
            adapter_path = BASE_DIR / adapter_path
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    model.eval()
    return model, tokenizer


def generate_one(model: Any, tokenizer: Any, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = output[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return normalize_space(text)


def load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        done.add((row["variant"], row["key"]))
    return done


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(run_dir: Path, predictions_path: Path, args: argparse.Namespace) -> None:
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    metrics = {variant: mean_metrics(vrows) for variant, vrows in sorted(by_variant.items())}
    macro: dict[str, dict[str, float]] = {}
    for label, members in {
        "sft_macro": ["sft_left", "sft_right"],
        "grpo_macro": ["grpo_left", "grpo_right"],
    }.items():
        member_rows = [row for variant in members for row in by_variant.get(variant, [])]
        if member_rows:
            macro[label] = mean_metrics(member_rows)
    metrics.update(macro)

    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = ["rouge1_f", "rouge2_f", "rougeL_f", "pred_words", "compression", "bigram_repetition"]
    lines = [
        "# GovReport Evaluation",
        "",
        f"- split: `{args.split}`",
        f"- limit: `{args.limit}`",
        f"- seed: `{args.seed}`",
        f"- max_input_tokens: `{args.max_input_tokens}`",
        f"- max_new_tokens: `{args.max_new_tokens}`",
        f"- variants: `{','.join(args.variants)}`",
        "",
        "| variant | n | " + " | ".join(cols) + " |",
        "|---|---:|" + "---:|" * len(cols),
    ]
    for variant, vals in metrics.items():
        n = len(by_variant.get(variant, []))
        if variant.endswith("_macro"):
            n = sum(len(by_variant.get(v, [])) for v in (["sft_left", "sft_right"] if variant.startswith("sft") else ["grpo_left", "grpo_right"]))
        lines.append(
            f"| {variant} | {n} | "
            + " | ".join(f"{vals.get(col, 0.0):.4f}" for col in cols)
            + " |"
        )
    lines.extend([
        "",
        "Notes:",
        "- ROUGE is a lightweight in-repo implementation, not the external rouge-score package.",
        "- GovReport documents are often longer than the model context; the script keeps the head and tail of each report.",
        "- For the final paper/table, rerun with a larger `--limit` or the full test split.",
    ])
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate base/SFT/GRPO Kanana variants on GovReport.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-id", default="kakaocorp/kanana-2-30b-a3b-instruct")
    parser.add_argument("--variants", default="base,sft_left,sft_right,grpo_left,grpo_right")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--precision", choices=["4bit", "bf16"], default="4bit")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    ns = parser.parse_args()
    ns.variants = [v.strip() for v in ns.variants.split(",") if v.strip()]
    unknown = [v for v in ns.variants if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}; choices={sorted(VARIANTS)}")
    return ns


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    run_name = args.run_name or f"{args.split}_n{args.limit}_seed{args.seed}_in{args.max_input_tokens}_out{args.max_new_tokens}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.resume:
        raise SystemExit(f"{predictions_path} exists; pass --resume or choose --run-name")

    examples = load_split_examples(args.data_dir, args.split, args.limit, args.seed)
    manifest = {
        "model_id": args.model_id,
        "variants": args.variants,
        "split": args.split,
        "limit": args.limit,
        "seed": args.seed,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "data_dir": str(args.data_dir),
        "examples": [{"key": ex.key, "title": ex.title, "path": ex.path} for ex in examples],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    done = load_done(predictions_path) if args.resume else set()
    print(f"GovReport examples: {len(examples)} | variants: {args.variants} | run_dir={run_dir}")

    for variant in args.variants:
        adapter = VARIANTS[variant]
        pending = [ex for ex in examples if (variant, ex.key) not in done]
        if not pending:
            print(f"[{variant}] all predictions already present")
            continue
        print(f"\n[{variant}] loading model adapter={adapter}", flush=True)
        start_load = time.time()
        model, tokenizer = load_model_and_tokenizer(args.model_id, adapter, args.precision, args.local_files_only)
        print(f"[{variant}] loaded in {time.time() - start_load:.1f}s", flush=True)

        for idx, ex in enumerate(pending, 1):
            source_words = len(word_tokens(ex.document))
            prompt = build_prompt(tokenizer, ex, args.max_input_tokens)
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            t0 = time.time()
            pred = generate_one(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p)
            elapsed = time.time() - t0
            metrics = compute_metrics(pred, ex.reference, source_words)
            row = {
                "variant": variant,
                "adapter": adapter,
                "key": ex.key,
                "source": ex.source,
                "report_id": ex.report_id,
                "title": ex.title,
                "path": ex.path,
                "prompt_tokens": prompt_tokens,
                "source_words": source_words,
                "reference": ex.reference,
                "prediction": pred,
                "elapsed_sec": elapsed,
                "metrics": metrics,
            }
            append_jsonl(predictions_path, row)
            print(
                f"[{variant}] {idx}/{len(pending)} {ex.key} "
                f"R1={metrics['rouge1_f']:.4f} R2={metrics['rouge2_f']:.4f} "
                f"RL={metrics['rougeL_f']:.4f} words={metrics['pred_words']:.0f} {elapsed:.1f}s",
                flush=True,
            )

        del model, tokenizer
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    write_report(run_dir, predictions_path, args)
    print(f"\nWrote: {run_dir / 'metrics.json'}")
    print(f"Wrote: {run_dir / 'report.md'}")


if __name__ == "__main__":
    main()
