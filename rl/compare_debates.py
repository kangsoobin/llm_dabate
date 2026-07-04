#!/usr/bin/env python3
"""
rl/compare_debates.py — SFT vs GRPO 어댑터 토론 정성/정량 비교
═══════════════════════════════════════════════════════════════

사용법:
    # 1) 같은 주제·같은 설정으로 두 구성의 토론을 생성 (simulate_vllm.py)
    python rl/simulate_vllm.py --gpu 0 --n-topics 4 --rounds-per-topic 8 --max-model-len 16384 \
        --left-adapter adapters/left_grpo --right-adapter adapters/right_grpo --out rl/data/eval_grpo.jsonl
    python rl/simulate_vllm.py --gpu 3 --n-topics 4 --rounds-per-topic 8 --max-model-len 16384 \
        --left-adapter adapters/left --right-adapter adapters/right --out rl/data/eval_sft.jsonl

    # 2) 비교
    python rl/compare_debates.py --a rl/data/eval_sft.jsonl --b rl/data/eval_grpo.jsonl \
        --label-a SFT --label-b GRPO

readme.md '검증 방법'의 지표를 자동화한 것:
    1. 중립·양시론 표현 빈도 (낮을수록 좋음 — Default Bias 억제 목표)
    2. 라운드별 페르소나 anchor 유사도 (라운드가 가도 유지되는지 — R_persona 목표)
    3. 라운드별 상대 논점 커버리지 (논점 교전도 — R_engagement 목표)
    4. 자기 반복도 (직전 자기 발언과의 Jaccard — R_novelty 목표)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import yaml  # noqa: E402

from rl.rewards.utils import SentenceEmbedder, jaccard_similarity, key_point_coverage  # noqa: E402
from rl.rewards.v2_redesign import _load_anchor_texts  # noqa: E402

# readme.md '정성 검증' 목록 + 변형
NEUTRAL_PATTERNS = [
    "양쪽 다 일리", "양측 모두 일리", "둘 다 일리",
    "균형 잡힌", "균형있", "균형 있",
    "복잡한 문제", "복잡한 사안", "단정하기 어렵",
    "일부 동의", "부분적으로 동의", "일리가 있습니다", "일리가 있다",
    "중립적", "신중하게 접근", "종합적으로 고려",
]


def load_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def neutral_hits(text: str) -> int:
    return sum(text.count(p) for p in NEUTRAL_PATTERNS)


def analyze(rows: list[dict], embedder: SentenceEmbedder, anchor_vecs: dict) -> dict:
    per_round: dict[int, dict] = defaultdict(lambda: {"drift": [], "coverage": [], "selfrep": [], "neutral": 0, "n": 0})
    total_neutral = 0
    for r in rows:
        rd = per_round[r["round_num"]]
        resp = r["response_ref"]
        rd["n"] += 1
        hits = neutral_hits(resp)
        rd["neutral"] += hits
        total_neutral += hits
        rd["drift"].append(embedder.cos_sim_vec(resp, anchor_vecs[r["side"]]))
        if r["opponent_response"]:
            rd["coverage"].append(key_point_coverage(r["opponent_response"], resp))
        if r["own_history"]:
            rd["selfrep"].append(jaccard_similarity(resp, [r["own_history"][-1]]))

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    summary = {
        "total_neutral": total_neutral,
        "n_turns": len(rows),
        "rounds": {
            rn: {
                "drift": mean(d["drift"]),
                "coverage": mean(d["coverage"]),
                "selfrep": mean(d["selfrep"]),
                "neutral": d["neutral"],
            }
            for rn, d in sorted(per_round.items())
        },
    }
    return summary


def print_comparison(sum_a: dict, sum_b: dict, label_a: str, label_b: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"중립·양시론 표현 총 출현: {label_a}={sum_a['total_neutral']}회 / {label_b}={sum_b['total_neutral']}회"
          f"  (각 {sum_a['n_turns']}턴)")
    print(f"{'=' * 78}")
    header = f"{'라운드':>4} │ {'anchor 유사도(페르소나 유지)':^30} │ {'논점 커버리지':^20} │ {'자기반복 Jaccard':^18}"
    print(header)
    print(f"{'':>4} │ {label_a:^14} {label_b:^14} │ {label_a:^9} {label_b:^9} │ {label_a:^8} {label_b:^8}")
    print("─" * 78)
    for rn in sorted(sum_a["rounds"]):
        a, b = sum_a["rounds"][rn], sum_b["rounds"].get(rn, {})
        print(
            f"{rn:>4} │ {a['drift']:^14.4f} {b.get('drift', float('nan')):^14.4f} │ "
            f"{a['coverage']:^9.3f} {b.get('coverage', float('nan')):^9.3f} │ "
            f"{a['selfrep']:^8.3f} {b.get('selfrep', float('nan')):^8.3f}"
        )
    print("─" * 78)
    print("해석: anchor 유사도 ↑=페르소나 유지 잘함 / 커버리지 ↑=상대 논점 교전 / 자기반복 ↓=좋음")


def print_excerpts(rows_a: list[dict], rows_b: list[dict], label_a: str, label_b: str, round_num: int) -> None:
    """같은 주제·같은 side·같은 라운드의 발언을 나란히 출력 (정성 비교용)."""
    def pick(rows, side):
        for r in rows:
            if r["round_num"] == round_num and r["side"] == side:
                return r
        return None

    for side in ("left", "right"):
        a, b = pick(rows_a, side), pick(rows_b, side)
        if not a or not b:
            continue
        print(f"\n{'─' * 78}\n[{side.upper()} / 라운드 {round_num} / 주제: {a['question'][:40]}]")
        print(f"\n◆ {label_a}:\n{a['response_ref'][:500]}{'...' if len(a['response_ref']) > 500 else ''}")
        print(f"\n◆ {label_b}:\n{b['response_ref'][:500]}{'...' if len(b['response_ref']) > 500 else ''}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT vs GRPO 토론 비교")
    parser.add_argument("--a", required=True, help="비교 기준 트랜스크립트 (예: eval_sft.jsonl)")
    parser.add_argument("--b", required=True, help="비교 대상 트랜스크립트 (예: eval_grpo.jsonl)")
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--excerpt-rounds", type=int, nargs="*", default=[1, 8],
                        help="원문 발췌를 출력할 라운드 번호들")
    args = parser.parse_args()

    with open(os.path.join(BASE_DIR, "config", "reward.yaml"), encoding="utf-8") as f:
        v2_cfg = yaml.safe_load(f).get("v2", {})
    embedder = SentenceEmbedder(model_id=v2_cfg.get("embedding_model", "BAAI/bge-m3"))
    anchors = _load_anchor_texts(v2_cfg, BASE_DIR)
    anchor_vecs = {side: embedder.encode_mean(texts) for side, texts in anchors.items()}

    rows_a = load_rows(args.a if os.path.isabs(args.a) else os.path.join(BASE_DIR, args.a))
    rows_b = load_rows(args.b if os.path.isabs(args.b) else os.path.join(BASE_DIR, args.b))

    sum_a = analyze(rows_a, embedder, anchor_vecs)
    sum_b = analyze(rows_b, embedder, anchor_vecs)
    print_comparison(sum_a, sum_b, args.label_a, args.label_b)

    for rn in args.excerpt_rounds:
        print_excerpts(rows_a, rows_b, args.label_a, args.label_b, rn)


if __name__ == "__main__":
    main()
