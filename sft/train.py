#!/usr/bin/env python3
"""
sft/train.py — QLoRA 파인튜닝
══════════════════════════════

사용법:
    conda activate debate
    pip install peft trl datasets
    python sft/train.py --side left
    python sft/train.py --side right

동작:
    1. sft/data/{side}_train.jsonl 로드
    2. Qwen2.5-14B-Instruct를 4-bit로 로드 + QLoRA 적용
    3. 1 epoch SFT (response 토큰에만 loss)
    4. adapters/{side}/ 에 LoRA 어댑터 저장

소요 시간: ~10분 (RTX 3090 기준)
"""

import argparse
import os
import sys

import torch
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ────────────────────────────────────────────────────────────
# 훈련 메인
# ────────────────────────────────────────────────────────────

def train(side: str, gpu: int, lora_r: int) -> None:
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from trl import SFTTrainer, SFTConfig

    # ── 설정 로드 ────────────────────────────────────────────
    cfg_dir = os.path.join(BASE_DIR, "config")
    with open(os.path.join(cfg_dir, "model.yaml"), encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    model_id = full_cfg["model_id"]
    data_path = os.path.join(BASE_DIR, "sft", "data", f"{side}_train.jsonl")
    output_dir = os.path.join(BASE_DIR, "adapters", side)

    if not os.path.exists(data_path):
        print(f"[오류] 데이터 파일이 없습니다: {data_path}")
        print("먼저 generate_data.py를 실행하세요.")
        sys.exit(1)

    print(f"\n[{side.upper()}] 훈련 시작")
    print(f"  모델: {model_id}")
    print(f"  GPU:  {gpu}")
    print(f"  데이터: {data_path}")
    print(f"  LoRA r: {lora_r}, alpha: {lora_r * 2}")
    print(f"  출력: {output_dir}\n")

    # ── 토크나이저 ───────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 모델 (4-bit) ─────────────────────────────────────────
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

    # ── QLoRA 설정 ───────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 데이터셋 ─────────────────────────────────────────────
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"  데이터셋 크기: {len(dataset)}개\n")

    # ── 훈련 인자 ────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=32,      # effective batch = 32
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        logging_steps=10,
        save_strategy="no",                  # 체크포인트 없이 최종만 저장
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_length=512,
    )

    # ── SFTTrainer ───────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    print("훈련 시작...")
    trainer.train()

    # ── 어댑터 저장 ──────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n어댑터 저장 완료: {output_dir}")
    print(f"\n다음 단계: config/model.yaml 에서 {side}_adapter 경로를 활성화하세요.")
    print(f'  {side}_adapter: "{os.path.relpath(output_dir, BASE_DIR)}"')


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA SFT 파인튜닝")
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--gpu",  type=int, default=None,
                        help="GPU 번호 (기본: left=0, right=1)")
    parser.add_argument("--r",    type=int, default=16,
                        help="LoRA rank (기본: 16)")
    args = parser.parse_args()

    if args.gpu is None:
        cfg_path = os.path.join(BASE_DIR, "config", "model.yaml")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        args.gpu = cfg["left_gpu"] if args.side == "left" else cfg["right_gpu"]

    train(args.side, args.gpu, args.r)
