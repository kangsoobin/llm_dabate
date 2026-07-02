#!/usr/bin/env python3
"""
rl/build_anchor.py — R_persona용 페르소나 anchor 텍스트 생성
════════════════════════════════════════════════════════════

사용법:
    python rl/build_anchor.py --side left
    python rl/build_anchor.py --side right

동작:
    sft/data/{side}_train.jsonl(sft/generate_data.py 결과물)에서 assistant 응답을
    무작위로 몇 개 골라 이어붙여, R_persona(rl/rewards/v2_redesign.py)의 "기준 페르소나"
    anchor 텍스트를 만든다. config/reward.yaml을 자동으로 고치지 않고 출력만 하는 이유는
    sft/train.py가 학습 후 어댑터 경로를 안내만 하고 model.yaml을 직접 고치지 않는 것과
    같은 관례를 따르기 위함 — 값을 확인하고 직접 반영하도록 유도.
"""

from __future__ import annotations

import argparse
import json
import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_anchor(side: str, n_samples: int, seed: int) -> str:
    data_path = os.path.join(BASE_DIR, "sft", "data", f"{side}_train.jsonl")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path}가 없습니다. 먼저 `python sft/generate_data.py --side {side}`를 실행하세요."
        )

    responses = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            for msg in record["messages"]:
                if msg["role"] == "assistant":
                    responses.append(msg["content"])

    if not responses:
        raise ValueError(f"{data_path}에서 assistant 응답을 찾지 못했습니다.")

    random.Random(seed).shuffle(responses)
    picked = responses[:n_samples]
    return " ".join(picked)


def main() -> None:
    parser = argparse.ArgumentParser(description="R_persona anchor 텍스트 생성")
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    anchor = build_anchor(args.side, args.n_samples, args.seed)
    print(f"\n[{args.side.upper()}] anchor 텍스트 ({args.n_samples}개 샘플 결합, {len(anchor)}자)\n")
    print(f"config/reward.yaml 의 v2.anchor_texts.{args.side} 에 아래 값을 붙여넣으세요:\n")
    print(anchor)


if __name__ == "__main__":
    main()
