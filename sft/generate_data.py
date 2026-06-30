#!/usr/bin/env python3
"""
sft/generate_data.py — SFT 훈련 데이터 생성
═══════════════════════════════════════════════

사용법:
    conda activate debate
    python sft/generate_data.py --side left
    python sft/generate_data.py --side right

동작:
    1. sft/questions.yaml에서 시드 질문 로드
    2. 질문이 100개 미만이면 LLM으로 자동 확장
    3. 각 질문마다 N회(기본 20회) 응답 생성 (temperature=1.0)
    4. sft/data/{side}_train.jsonl 저장
"""

import argparse
import json
import os
import sys

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agents import create_left_agent, create_right_agent


# ────────────────────────────────────────────────────────────
# 설정 로드
# ────────────────────────────────────────────────────────────

def load_configs():
    cfg_dir = os.path.join(BASE_DIR, "config")
    with open(os.path.join(cfg_dir, "model.yaml"), encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    with open(os.path.join(cfg_dir, "prompts.yaml"), encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    model_cfg = {
        "model_id":     full_cfg["model_id"],
        "left_gpu":     full_cfg["left_gpu"],
        "right_gpu":    full_cfg["right_gpu"],
        "quantization": full_cfg["quantization"],
    }
    # 데이터 생성 시에는 temperature를 1.0으로 고정
    gen_cfg = {
        "max_new_tokens":     full_cfg["max_new_tokens"],
        "temperature":        1.0,
        "top_p":              full_cfg["top_p"],
        "repetition_penalty": full_cfg["repetition_penalty"],
    }
    return model_cfg, gen_cfg, prompts["left"], prompts["right"]


# ────────────────────────────────────────────────────────────
# 질문 로드 및 확장
# ────────────────────────────────────────────────────────────

def load_questions(target: int = 100) -> list[str]:
    """questions.yaml에서 질문을 로드한다."""
    q_path = os.path.join(BASE_DIR, "sft", "questions.yaml")
    with open(q_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    questions = list(data.get("seed_questions", []))
    questions += list(data.get("generated_questions", []))
    questions = [q for q in questions if q]

    print(f"  질문 {len(questions)}개 로드 완료.")
    return questions[:target]



# ────────────────────────────────────────────────────────────
# 데이터 생성 메인
# ────────────────────────────────────────────────────────────

def generate(side: str, gpu: int, n_per_question: int) -> None:
    model_cfg, gen_cfg, left_prompt, right_prompt = load_configs()

    # GPU 오버라이드 (--gpu 인자)
    if side == "left":
        model_cfg["left_gpu"] = gpu
        agent = create_left_agent(model_cfg, gen_cfg, left_prompt)
    else:
        model_cfg["right_gpu"] = gpu
        agent = create_right_agent(model_cfg, gen_cfg, right_prompt)

    print(f"\n[{side.upper()}] 모델 로딩 중 (GPU {gpu})...")
    agent.load()
    print(f"[{side.upper()}] 모델 로드 완료.\n")

    questions = load_questions(target=100)
    print(f"질문 {len(questions)}개 준비. 질문당 {n_per_question}회 응답 생성 시작...\n")

    out_dir = os.path.join(BASE_DIR, "sft", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{side}_train.jsonl")

    system_prompt = left_prompt if side == "left" else right_prompt
    total = len(questions) * n_per_question
    done = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for q_idx, question in enumerate(questions, 1):
            agent.reset_history()  # 각 질문은 독립된 대화
            for rep in range(n_per_question):
                agent.reset_history()
                response = agent.generate(question)  # stream_callback=None
                record = {
                    "messages": [
                        {"role": "system",    "content": system_prompt},
                        {"role": "user",      "content": question},
                        {"role": "assistant", "content": response},
                    ]
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                done += 1
                print(
                    f"\r  [{side}] Q{q_idx}/{len(questions)} | Rep{rep+1}/{n_per_question} "
                    f"| 전체 {done}/{total} ({done/total*100:.1f}%)",
                    end="", flush=True,
                )

    print(f"\n\n완료! 저장: {out_path}  ({done}개 예시)")


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT 훈련 데이터 생성")
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--gpu",  type=int, default=None,
                        help="GPU 번호 (기본: left=0, right=1)")
    parser.add_argument("--n-per-question", type=int, default=20,
                        help="질문당 응답 생성 횟수 (기본: 20)")
    args = parser.parse_args()

    if args.gpu is None:
        args.gpu = 0 if args.side == "left" else 1

    generate(args.side, args.gpu, args.n_per_question)
