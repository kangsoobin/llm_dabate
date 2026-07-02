#!/usr/bin/env python3
"""
rl/train_grpo.py — GRPO 기반 RL 파인튜닝
════════════════════════════════════════

사용법:
    conda activate debate
    pip install "trl>=0.24" peft datasets sentence-transformers
    # (선택) API judge를 쓰려면: pip install anthropic  또는  pip install openai

    # 1) 멀티라운드 self-play 트랜스크립트 생성 (GPU 필요)
    python rl/simulate.py --left-adapter adapters/left --right-adapter adapters/right \
        --n-topics 40 --rounds-per-topic 3 --out rl/data/transcripts.jsonl

    # 2) (v2, R_persona용) anchor 텍스트 생성 후 config/reward.yaml에 반영
    python rl/build_anchor.py --side left
    python rl/build_anchor.py --side right

    # 3) GRPO 학습
    python rl/train_grpo.py --side left  --reward-version v2
    python rl/train_grpo.py --side right --reward-version v1

동작:
    1. rl/data/transcripts.jsonl 로드 → 지정 side의 턴만 골라 GRPO 데이터셋 구성 (rl/rollout.py)
    2. 해당 side의 SFT 어댑터(adapters/{side})를 이어서 로드 (없으면 새 LoRA로 시작)
    3. config/reward.yaml 기준 RewardComposer 구성 (v1|v2, judge on/off는 yaml에서 제어)
    4. TRL GRPOTrainer로 학습, adapters/{side}_grpo/ 에 저장

GRPO 손실 형태(클리핑 + KL 페널티)는 docs/reward_design_v2.md §4, 보상 설계.pdf p.2와 동일.
LoRA 기반이므로 π_ref는 별도 모델 복제 없이 "어댑터 비활성화 상태의 base 모델"로 대체된다
(TRL GRPOTrainer의 PEFT 표준 동작) — 3090 24GB 두 장이라는 VRAM 제약에서 ref 모델 복제 비용을 없앤다.
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def train(
    side: str,
    reward_version: str,
    gpu: int,
    group_size: int,
    transcript_path: str,
) -> None:
    import torch
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    from rl.rewards.composer import RewardComposer
    from rl.rollout import build_grpo_dataset

    cfg_dir = os.path.join(BASE_DIR, "config")
    with open(os.path.join(cfg_dir, "model.yaml"), encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(os.path.join(cfg_dir, "reward.yaml"), encoding="utf-8") as f:
        reward_cfg = yaml.safe_load(f)
    reward_cfg["version"] = reward_version

    model_id = model_cfg["model_id"]
    sft_adapter = os.path.join(BASE_DIR, model_cfg.get(f"{side}_adapter") or f"adapters/{side}")
    output_dir = os.path.join(BASE_DIR, "adapters", f"{side}_grpo")

    print(f"\n[{side.upper()}] GRPO 학습 시작 (reward={reward_version}, GPU {gpu})")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": gpu},
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = None
    if os.path.isdir(sft_adapter):
        print(f"  SFT 어댑터 이어받기: {sft_adapter}")
        model = PeftModel.from_pretrained(model, sft_adapter, is_trainable=True)
    else:
        print(f"  SFT 어댑터 없음({sft_adapter}) — base 모델에서 새 LoRA로 시작")
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

    dataset = build_grpo_dataset(transcript_path, side)
    print(f"  GRPO 데이터셋: {len(dataset)}개 턴")

    composer = RewardComposer.from_config(reward_cfg)
    reward_fn = composer.as_trl_reward_fn()
    reward_fn.__name__ = f"debate_reward_{reward_version}"

    grpo_cfg = GRPOConfig(
        output_dir=output_dir,
        num_generations=group_size,
        per_device_train_batch_size=group_size,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=1e-5,
        beta=0.04,             # KL 페널티 계수 β (docs/reward_design_v2.md §4)
        epsilon=0.2,            # 클리핑 ε
        max_prompt_length=2048,
        max_completion_length=model_cfg.get("max_new_tokens", 600),
        temperature=model_cfg.get("temperature", 0.9),
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=grpo_cfg,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    os.makedirs(output_dir, exist_ok=True)
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nGRPO 어댑터 저장 완료: {output_dir}")
    print(
        f'config/model.yaml의 {side}_adapter를 "{os.path.relpath(output_dir, BASE_DIR)}"로 '
        f"바꾸면 토론 시스템에 바로 적용됩니다."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO 기반 RL 파인튜닝")
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--reward-version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=8, help="GRPO 그룹당 샘플 수 G")
    parser.add_argument(
        "--transcripts",
        default=os.path.join("rl", "data", "transcripts.jsonl"),
        help="rl/simulate.py가 생성한 트랜스크립트 경로",
    )
    args = parser.parse_args()

    if args.gpu is None:
        with open(os.path.join(BASE_DIR, "config", "model.yaml"), encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)
        args.gpu = model_cfg["left_gpu"] if args.side == "left" else model_cfg["right_gpu"]

    transcript_path = (
        args.transcripts if os.path.isabs(args.transcripts) else os.path.join(BASE_DIR, args.transcripts)
    )

    train(args.side, args.reward_version, args.gpu, args.group_size, transcript_path)
