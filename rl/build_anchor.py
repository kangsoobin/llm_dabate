#!/usr/bin/env python3
"""
rl/build_anchor.py — R_persona용 페르소나 anchor 텍스트 생성
════════════════════════════════════════════════════════════

사용법:
    python rl/build_anchor.py --side left
    python rl/build_anchor.py --side right

동작:
    sft/data/{side}_train.jsonl(팀원 공유 SFT 데이터)에서 assistant 응답을 무작위로
    n개 골라 rl/data/anchors/{side}.json 에 **리스트로** 저장한다.

    R_persona(rl/rewards/v2_redesign.py)는 이 샘플들을 각각 임베딩한 뒤 평균내어
    페르소나 기준점(anchor) 벡터로 쓴다 — docs/reward_design_v2.md §3.1의
    "seed 응답 임베딩 평균" 원안 그대로. (예전 구현처럼 텍스트를 이어붙이면
    임베딩 모델 max_seq_length에서 앞부분만 잘려 anchor가 왜곡된다.)

    config/reward.yaml의 v2.anchor_files가 이 경로를 가리키고 있으면 자동 로드된다.
"""

from __future__ import annotations

import argparse
import json
import os
import random

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_anchor_samples(side: str, n_samples: int, seed: int) -> list[str]:
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
    return responses[:n_samples]


def main() -> None:
    parser = argparse.ArgumentParser(description="R_persona anchor 텍스트 생성")
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--n-samples", type=int, default=16,
                        help="anchor 샘플 수 (기본 16 — 임베딩 평균이므로 넉넉히)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = build_anchor_samples(args.side, args.n_samples, args.seed)
    out_path = os.path.join(BASE_DIR, "rl", "data", "anchors", f"{args.side}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    total_chars = sum(len(s) for s in samples)
    print(f"[{args.side.upper()}] anchor 샘플 {len(samples)}개 (총 {total_chars:,}자) 저장: {out_path}")
    print("config/reward.yaml의 v2.anchor_files가 이 경로를 가리키면 자동 사용됩니다.")


if __name__ == "__main__":
    main()
