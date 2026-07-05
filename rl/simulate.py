#!/usr/bin/env python3
"""
rl/simulate.py — GRPO 롤아웃용 멀티라운드 self-play 트랜스크립트 생성
════════════════════════════════════════════════════════════════════

사용법:
    conda activate debate
    python rl/simulate.py \
        --left-adapter adapters/left --right-adapter adapters/right \
        --n-topics 40 --rounds-per-topic 3 \
        --out rl/data/transcripts.jsonl

동작:
    1. sft/questions.yaml에서 주제를 뽑아 LEFT/RIGHT가 실제로 여러 라운드 토론하게 한다
       (core/session.py와 동일한 상대 발언 주입 로직 — DebateSession._build_message 재사용).
    2. 매 턴마다 (side, question, opponent_response, own_history, prompt_messages)를 기록한다.
       prompt_messages는 그 턴의 생성 직전 시점 [system] + 대화 히스토리 전체이며,
       rl/rollout.py가 이걸 그대로 GRPOTrainer의 conversational prompt로 사용해
       동일 컨텍스트에서 새로 G개 응답을 재샘플링한다 (turn-level GRPO).

주의: LEFT/RIGHT 모델 로딩이 필요하므로 GPU 서버에서만 실행 가능하다.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agents import create_left_agent, create_right_agent  # noqa: E402
from core.session import DebateSession  # noqa: E402


def load_topics(n: int | None = None) -> list[str]:
    q_path = os.path.join(BASE_DIR, "sft", "questions.yaml")
    with open(q_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    topics = list(data.get("seed_questions", [])) + list(data.get("generated_questions", []))
    topics = [t for t in topics if t]
    return topics[:n] if n is not None else topics


def own_history_texts(agent, exclude_last: bool) -> list[str]:
    """agent.history에서 assistant 발언만 순서대로 추출. exclude_last=True면 방금 생성분은 제외."""
    texts = [m["content"] for m in agent.history if m["role"] == "assistant"]
    return texts[:-1] if exclude_last and texts else texts


def load_configs() -> tuple[dict, dict, str, str]:
    cfg_dir = os.path.join(BASE_DIR, "config")
    with open(os.path.join(cfg_dir, "model.yaml"), encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(os.path.join(cfg_dir, "prompts.yaml"), encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    gen_cfg = {
        "max_new_tokens": model_cfg["max_new_tokens"],
        "temperature": model_cfg["temperature"],
        "top_p": model_cfg["top_p"],
        "repetition_penalty": model_cfg["repetition_penalty"],
    }
    return model_cfg, gen_cfg, prompts["left"], prompts["right"]


def simulate_debates(
    left_adapter: str | None,
    right_adapter: str | None,
    n_topics: int,
    rounds_per_topic: int,
    left_gpu: int,
    right_gpu: int,
    seed: int = 42,
) -> list[dict]:
    model_cfg, gen_cfg, left_prompt, right_prompt = load_configs()
    model_cfg_left = {**model_cfg, "left_gpu": left_gpu}
    model_cfg_right = {**model_cfg, "right_gpu": right_gpu}

    left = create_left_agent(model_cfg_left, gen_cfg, left_prompt, adapter_path=left_adapter)
    right = create_right_agent(model_cfg_right, gen_cfg, right_prompt, adapter_path=right_adapter)

    print(f"[LEFT] 모델 로딩 중 (GPU {left_gpu}, adapter={left_adapter})...")
    left.load()
    print(f"[RIGHT] 모델 로딩 중 (GPU {right_gpu}, adapter={right_adapter})...")
    right.load()

    # _build_message는 self를 참조하지 않는 순수 로직이므로 인스턴스 하나로 재사용한다.
    session = DebateSession(left, right)

    topics = load_topics(n_topics)
    random.Random(seed).shuffle(topics)

    rows: list[dict] = []
    for t_idx, topic in enumerate(topics, 1):
        left.reset_history()
        right.reset_history()

        for round_idx in range(1, rounds_per_topic + 1):
            # ── LEFT 턴 ──────────────────────────────────────
            right_prev = right.last_response
            left_own_hist = own_history_texts(left, exclude_last=False)
            left_msg = session._build_message(topic, right_prev, right.name, speaker_side="left")
            l_resp = left.generate(left_msg)
            left_prompt_messages = [{"role": "system", "content": left_prompt}] + left.history[:-1]

            rows.append(
                {
                    "side": "left",
                    "question": topic,
                    "opponent_response": right_prev,
                    "own_history": left_own_hist,
                    "round_num": round_idx,
                    "prompt_messages": left_prompt_messages,
                    "response_ref": l_resp,
                }
            )

            # ── RIGHT 턴 (방금 나온 LEFT 발언을 주입) ─────────
            right_own_hist = own_history_texts(right, exclude_last=False)
            right_msg = session._build_message(topic, l_resp, left.name, speaker_side="right")
            r_resp = right.generate(right_msg)
            right_prompt_messages = [{"role": "system", "content": right_prompt}] + right.history[:-1]

            rows.append(
                {
                    "side": "right",
                    "question": topic,
                    "opponent_response": l_resp,
                    "own_history": right_own_hist,
                    "round_num": round_idx,
                    "prompt_messages": right_prompt_messages,
                    "response_ref": r_resp,
                }
            )

        print(f"\r  주제 {t_idx}/{len(topics)} 완료", end="", flush=True)

    print()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO 롤아웃용 self-play 트랜스크립트 생성")
    parser.add_argument("--left-adapter", default="adapters/left")
    parser.add_argument("--right-adapter", default="adapters/right")
    parser.add_argument("--n-topics", type=int, default=40)
    parser.add_argument("--rounds-per-topic", type=int, default=3)
    parser.add_argument("--left-gpu", type=int, default=0)
    parser.add_argument("--right-gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join("rl", "data", "transcripts.jsonl"))
    args = parser.parse_args()

    left_adapter = os.path.join(BASE_DIR, args.left_adapter) if args.left_adapter else None
    right_adapter = os.path.join(BASE_DIR, args.right_adapter) if args.right_adapter else None
    if left_adapter and not os.path.isdir(left_adapter):
        print(f"[경고] {left_adapter}가 없습니다 — 어댑터 없이(base 모델) 진행합니다.")
        left_adapter = None
    if right_adapter and not os.path.isdir(right_adapter):
        print(f"[경고] {right_adapter}가 없습니다 — 어댑터 없이(base 모델) 진행합니다.")
        right_adapter = None

    rows = simulate_debates(
        left_adapter,
        right_adapter,
        args.n_topics,
        args.rounds_per_topic,
        args.left_gpu,
        args.right_gpu,
        args.seed,
    )

    out_path = os.path.join(BASE_DIR, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n완료! {len(rows)}개 턴 저장: {out_path}")


if __name__ == "__main__":
    main()
