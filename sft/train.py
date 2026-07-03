#!/usr/bin/env python3
"""
sft/train.py — QLoRA 파인튜닝
══════════════════════════════

사용법:
    conda activate debate
    pip install "transformers>=4.51.0" peft trl datasets
    python sft/train.py --side left
    python sft/train.py --side right

동작:
    1. sft/data/{side}_train.jsonl 로드 (팀원이 공유한 기존 데이터 재사용 — 페르소나 프롬프트/응답
       텍스트 데이터는 base model과 무관하므로 Qwen2.5-14B → Kanana-2-30B-A3B로 바꿔도 그대로 씀)
    2. config/model.yaml의 model_id(현재 kakaocorp/kanana-2-30b-a3b-instruct)를 4-bit로 로드 + QLoRA 적용
    3. 1 epoch SFT (response 토큰에만 loss)
    4. adapters/{side}/ 에 LoRA 어댑터 저장

Kanana-2-30B-A3B는 DeepseekV3ForCausalLM 아키텍처(MLA + MoE, 128 routed + 2 shared experts)라
Qwen과 내부 모듈 이름이 다르다(q_proj는 있지만 k_proj/v_proj 대신 kv_a_proj_with_mqa/kv_b_proj 등).
하드코딩된 target_modules 대신 실제 로드된 모델을 순회해 LoRA 대상을 찾는다(discover_lora_target_modules)
— 이렇게 하면 나중에 다른 base model로 또 바뀌어도 이 스크립트를 그대로 쓸 수 있다.

주의:
- 기존 Qwen 기반 adapters/left, adapters/right는 아키텍처가 달라 이 모델에 이어서 쓸 수 없다.
  config/model.yaml의 left_adapter/right_adapter가 null인 이유가 이것 — 처음부터 재학습해야 함.
- 이전 Qwen2.5-14B 기준 소요 시간(~10분/RTX 3090)은 참고용. Kanana는 총 30B(4bit 기준 가중치만
  ~16GB)라 학습 시 VRAM 여유가 훨씬 빠듯하고 소요 시간도 늘어난다 — config/model.yaml 주석 참고.
"""

import argparse
import os
import re
import sys

import torch
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


# ────────────────────────────────────────────────────────────
# LoRA 대상 모듈 자동 탐색 (base model 아키텍처에 무관하게 동작)
# ────────────────────────────────────────────────────────────

# 밀집(dense) attention(Qwen 등) + MLA attention(Kanana/DeepSeek-V3) + SwiGLU MLP(dense/shared/routed
# 공통) + MoE 라우터를 모두 아우르는 leaf 모듈 이름 후보 집합. 실제로 모델에 존재하는 것만 골라 쓴다.
_LORA_LEAF_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj",                      # 밀집 attention
    "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",   # MLA attention
    "gate_proj", "up_proj", "down_proj",                         # SwiGLU MLP
    "gate",                                                      # MoE 라우터
}
_ROUTED_EXPERT_SEGMENT = re.compile(r"\.experts\.\d+\.")


def discover_lora_target_modules(model, include_routed_experts: bool = False) -> list[str]:
    """
    model.named_modules()를 순회해 실제로 존재하는 LoRA 대상 모듈의 전체 경로를 찾는다.

    include_routed_experts=False(기본)면 128개 routed expert 내부의 gate/up/down_proj는 제외하고
    attention + MoE 라우터 + shared_experts(+ dense layer의 MLP)만 대상으로 삼는다 — 토큰마다 8개
    (routed 6 + shared 2)만 활성화되는 구조에서 routed expert 128개 × 레이어 수 전체에 LoRA를 붙이면
    모듈 수가 폭발적으로 늘어나 학습이 훨씬 무거워지기 때문. 필요하면 --lora-experts로 켤 수 있다.
    """
    found: list[str] = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in _LORA_LEAF_NAMES:
            continue
        if not hasattr(module, "weight"):
            continue
        if _ROUTED_EXPERT_SEGMENT.search(name) and not include_routed_experts:
            continue
        found.append(name)

    if not found:
        raise RuntimeError(
            "LoRA 대상 모듈을 하나도 찾지 못했습니다. base model 구조가 예상과 다를 수 있습니다 — "
            "model.named_modules()를 직접 확인한 뒤 --target-modules로 정규식을 지정하세요."
        )
    return sorted(found)


# ────────────────────────────────────────────────────────────
# 훈련 메인
# ────────────────────────────────────────────────────────────

def train(
    side: str,
    gpu: int,
    lora_r: int,
    target_modules: list[str] | None,
    lora_experts: bool,
) -> None:
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
        print("먼저 generate_data.py를 실행하거나, 팀원이 공유한 jsonl을 sft/data/에 넣으세요.")
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
    if target_modules is None:
        target_modules = discover_lora_target_modules(model, include_routed_experts=lora_experts)
    preview = target_modules[:10]
    print(f"  LoRA 대상 모듈: {len(target_modules)}개 (예: {preview}{' ...' if len(target_modules) > 10 else ''})")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
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
    parser.add_argument(
        "--target-modules", type=str, default=None,
        help="쉼표로 구분한 LoRA 대상 모듈 전체 경로를 직접 지정 (기본: 모델을 로드한 뒤 자동 탐색)",
    )
    parser.add_argument(
        "--lora-experts", action="store_true",
        help="MoE routed expert(128개) 내부 gate/up/down_proj까지 LoRA 대상에 포함 (기본: 제외, 더 가벼움)",
    )
    args = parser.parse_args()

    if args.gpu is None:
        cfg_path = os.path.join(BASE_DIR, "config", "model.yaml")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        args.gpu = cfg["left_gpu"] if args.side == "left" else cfg["right_gpu"]

    target_modules = args.target_modules.split(",") if args.target_modules else None

    train(args.side, args.gpu, args.r, target_modules, args.lora_experts)
