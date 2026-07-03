#!/usr/bin/env python3
"""
rl/simulate_vllm.py — vLLM 기반 self-play 트랜스크립트 생성 (rl/simulate.py의 고속 버전)
═══════════════════════════════════════════════════════════════════════════════════

사용법:
    # vllm이 설치된 환경에서 (프로젝트 .venv 또는 별도 vllm 환경)
    python rl/simulate_vllm.py --gpu 0 --n-topics 40 --rounds-per-topic 3

rl/simulate.py 대비 차이:
    - HF transformers 대신 vLLM 오프라인 엔진 사용. transformers의 DeepSeek-V3 MoE 구현은
      128개 expert를 Python loop로 순회해 발언 하나에 수 분이 걸리지만(HANDOFF.md §4-8),
      vLLM은 fused MoE 커널로 수백 배 빠르다.
    - base 모델 하나(bf16)에 LEFT/RIGHT LoRA 어댑터를 멀티-LoRA로 얹어 GPU 한 장에서 처리
      (vLLM ≥0.15의 MoE 계열 multi-LoRA 지원 사용).
    - 라운드/side 단위로 모든 토픽을 배치 생성 — 토픽 수가 늘어도 처리 시간이 거의 선형 이하.

출력 스키마는 rl/simulate.py와 동일 (rl/rollout.py가 그대로 소비):
    {side, question, opponent_response, own_history, round_num, prompt_messages, response_ref}
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

from core.session import build_debate_message  # noqa: E402  (무거운 모델 로딩 없음)
from runtime_utils import restrict_visible_gpus  # noqa: E402

LEFT_NAME = "이진보"   # agents/left_agent.py와 동일
RIGHT_NAME = "김보수"  # agents/right_agent.py와 동일


def load_topics(n: int | None = None) -> list[str]:
    q_path = os.path.join(BASE_DIR, "sft", "questions.yaml")
    with open(q_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    topics = list(data.get("seed_questions", [])) + list(data.get("generated_questions", []))
    topics = [t for t in topics if t]
    return topics[:n] if n is not None else topics


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


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM 기반 self-play 트랜스크립트 생성")
    parser.add_argument("--left-adapter", default="adapters/left")
    parser.add_argument("--right-adapter", default="adapters/right")
    parser.add_argument("--n-topics", type=int, default=40)
    parser.add_argument("--rounds-per-topic", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0, help="사용할 물리 GPU 한 장 (bf16 기준 ~65GB 사용)")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join("rl", "data", "transcripts.jsonl"))
    args = parser.parse_args()

    # 공용 서버 GPU 침범 방지 — vllm/torch가 CUDA를 초기화하기 전에 호출
    restrict_visible_gpus([args.gpu])

    from vllm import LLM, SamplingParams  # noqa: E402  (CUDA_VISIBLE_DEVICES 설정 후 import)
    from vllm.lora.request import LoRARequest  # noqa: E402

    model_cfg, gen_cfg, left_prompt, right_prompt = load_configs()

    left_adapter = os.path.join(BASE_DIR, args.left_adapter)
    right_adapter = os.path.join(BASE_DIR, args.right_adapter)
    for path, side in ((left_adapter, "left"), (right_adapter, "right")):
        if not os.path.isdir(path):
            print(f"[오류] SFT 어댑터가 없습니다: {path} — sft/train.py --side {side}를 먼저 실행하세요.")
            sys.exit(1)

    print(f"vLLM 로딩: {model_cfg['model_id']} (bf16, multi-LoRA)")
    llm = LLM(
        model=model_cfg["model_id"],
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,          # sft/train.py의 LoRA r=16과 일치해야 함
        max_loras=2,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    left_lora = LoRARequest("left", 1, left_adapter)
    right_lora = LoRARequest("right", 2, right_adapter)

    sampling = SamplingParams(
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
        repetition_penalty=gen_cfg["repetition_penalty"],
        max_tokens=gen_cfg["max_new_tokens"],
        seed=args.seed,
    )

    topics = load_topics(args.n_topics)
    random.Random(args.seed).shuffle(topics)
    n = len(topics)
    print(f"토픽 {n}개 × {args.rounds_per_topic}라운드 × 2 side self-play 시작")

    # 토픽별 side별 대화 히스토리 (system 제외, user/assistant 교대)
    left_hist: list[list[dict]] = [[] for _ in range(n)]
    right_hist: list[list[dict]] = [[] for _ in range(n)]
    last_left: list[str] = [""] * n   # 각 토픽에서 LEFT의 직전 발언
    last_right: list[str] = [""] * n

    rows: list[dict] = []

    def run_side_turn(side: str, round_idx: int) -> None:
        """모든 토픽에 대해 한 side의 이번 턴을 배치 생성."""
        is_left = side == "left"
        system_prompt = left_prompt if is_left else right_prompt
        hists = left_hist if is_left else right_hist
        opp_last = last_right if is_left else last_left
        opp_name = RIGHT_NAME if is_left else LEFT_NAME
        lora = left_lora if is_left else right_lora

        conversations = []
        for t in range(n):
            user_msg = build_debate_message(topics[t], opp_last[t], opp_name, speaker_side=side)
            conversations.append(
                [{"role": "system", "content": system_prompt}] + hists[t] + [{"role": "user", "content": user_msg}]
            )

        outputs = llm.chat(conversations, sampling, lora_request=lora)

        for t, out in enumerate(outputs):
            response = out.outputs[0].text.strip()
            own_history = [m["content"] for m in hists[t] if m["role"] == "assistant"]
            rows.append(
                {
                    "side": side,
                    "question": topics[t],
                    "opponent_response": opp_last[t],
                    "own_history": own_history,
                    "round_num": round_idx,
                    "prompt_messages": conversations[t],
                    "response_ref": response,
                }
            )
            hists[t].append({"role": "user", "content": conversations[t][-1]["content"]})
            hists[t].append({"role": "assistant", "content": response})
            if is_left:
                last_left[t] = response
            else:
                last_right[t] = response

    for round_idx in range(1, args.rounds_per_topic + 1):
        print(f"  라운드 {round_idx}/{args.rounds_per_topic} — LEFT 배치 생성...")
        run_side_turn("left", round_idx)
        print(f"  라운드 {round_idx}/{args.rounds_per_topic} — RIGHT 배치 생성...")
        run_side_turn("right", round_idx)

    out_path = os.path.join(BASE_DIR, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n완료! {len(rows)}개 턴 저장: {out_path}")


if __name__ == "__main__":
    main()
