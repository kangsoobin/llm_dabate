#!/usr/bin/env python3
"""
rl/train_grpo.py — GRPO 기반 RL 파인튜닝
════════════════════════════════════════

사용법 (2026-07-04 갱신 — 실행 환경/속도 이슈 반영, HANDOFF.md §3 참고):

    source .venv/bin/activate
    uv pip install sentence-transformers kiwipiepy   # 최초 1회

    # 1) 멀티라운드 self-play 트랜스크립트 생성 — vLLM 권장 (rl/simulate_vllm.py 참고)
    python rl/simulate_vllm.py --gpu 0 --n-topics 40 --rounds-per-topic 3

    # 2) R_persona용 anchor 생성 (GPU 불필요, 수 초)
    python rl/build_anchor.py --side left
    python rl/build_anchor.py --side right

    # 3) 보상 신호 sanity check (GPU 권장, 수 분) — PASS 확인 후 진행
    python rl/check_reward_sanity.py

    # 4-a) vLLM 서버 모드 (권장 — 생성이 수십 배 빠름):
    #      터미널 A (vLLM 서버, 빈 GPU 하나 배정):
    CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model kakaocorp/kanana-2-30b-a3b-instruct \
        --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.85
    #      터미널 B (학습, 다른 빈 GPU):
    python rl/train_grpo.py --side left --gpu 0 --use-vllm

    # 4-b) vLLM 없이 (fallback — transformers MoE 구현이 느려 side당 수십 시간):
    python rl/train_grpo.py --side left --gpu 0

동작:
    1. rl/data/transcripts.jsonl 로드 → 지정 side의 턴만 골라 GRPO 데이터셋 구성 (rl/rollout.py)
    2. base 모델을 bf16으로 로드(기본; 97GB GPU 기준) + 해당 side의 SFT 어댑터 이어받기
       - bf16을 기본으로 하는 이유: (a) 4bit(bnb)는 vLLM 서버로의 weight sync(PEFT merge)가
         불안정하고, (b) SFT에서 겪은 bnb 관련 이슈들(HANDOFF.md §4)을 원천 회피. VRAM이
         작은 GPU에서는 --precision 4bit으로 fallback 가능(단, --use-vllm과 함께 쓰지 말 것).
    3. config/reward.yaml 기준 RewardComposer 구성 (v1|v2, judge on/off는 yaml에서 제어)
    4. TRL GRPOTrainer로 학습, adapters/{side}_grpo/ 에 저장

GRPO 손실 형태(클리핑 + KL 페널티)는 docs/reward_design_v2.md §4, 보상 설계.pdf p.2와 동일.
LoRA 기반이므로 π_ref는 별도 모델 복제 없이 "어댑터 비활성화 상태의 base 모델"로 대체된다
(TRL GRPOTrainer의 PEFT 표준 동작).
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
    precision: str,
    use_vllm: bool,
    vllm_host: str,
    vllm_port: int,
    group_size: int,
    max_steps: int | None,
    transcript_path: str,
) -> None:
    import torch
    from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import GRPOConfig, GRPOTrainer

    from rl.rewards.composer import RewardComposer
    from rl.rollout import build_grpo_dataset
    from runtime_utils import patch_deepseek_v3_moe_dtype_bug

    patch_deepseek_v3_moe_dtype_bug()

    cfg_dir = os.path.join(BASE_DIR, "config")
    with open(os.path.join(cfg_dir, "model.yaml"), encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)
    with open(os.path.join(cfg_dir, "reward.yaml"), encoding="utf-8") as f:
        reward_cfg = yaml.safe_load(f)
    reward_cfg["version"] = reward_version

    model_id = model_cfg["model_id"]
    sft_adapter = os.path.join(BASE_DIR, model_cfg.get(f"{side}_adapter") or f"adapters/{side}")
    output_dir = os.path.join(BASE_DIR, "adapters", f"{side}_grpo")

    print(f"\n[{side.upper()}] GRPO 학습 시작 (reward={reward_version}, precision={precision}, vllm={use_vllm})")

    # ── 보상 컴포저를 모델 로드보다 먼저 구성 — anchor 누락 등 설정 문제로
    #    30B 모델을 로드한 뒤에야 죽는 것을 방지 (fail-fast)
    composer = RewardComposer.from_config(reward_cfg)
    reward_fn = composer.as_trl_reward_fn()
    reward_fn.__name__ = f"debate_reward_{reward_version}"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"device_map": {"": 0}, "trust_remote_code": True}
    if precision == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        if use_vllm:
            print(
                "  [경고] 4bit + vLLM weight sync 조합은 PEFT merge 정밀도 문제로 권장하지 않습니다 "
                "— bf16(--precision bf16)으로 바꾸는 것을 권장."
            )
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if precision == "4bit":
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
                "q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
        )

    dataset = build_grpo_dataset(transcript_path, side)
    print(f"  GRPO 데이터셋: {len(dataset)}개 턴")

    grpo_cfg = GRPOConfig(
        output_dir=output_dir,
        num_generations=group_size,
        per_device_train_batch_size=group_size,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        max_steps=max_steps if max_steps is not None else -1,
        learning_rate=1e-5,
        beta=0.04,              # KL 페널티 계수 β (docs/reward_design_v2.md §4)
        epsilon=0.2,            # 클리핑 ε
        # 프롬프트 = [system(페르소나 ~600토큰)] + 라운드별 히스토리 — 3라운드 기준 2048로는
        # 잘려서 system 프롬프트 앞부분이 날아갈 수 있어 3072로 상향.
        max_prompt_length=3072,
        max_completion_length=model_cfg.get("max_new_tokens", 600),
        temperature=model_cfg.get("temperature", 0.8),
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=use_vllm,
        vllm_mode="server" if use_vllm else "colocate",
        vllm_server_host=vllm_host,
        vllm_server_port=vllm_port,
        logging_steps=1,
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
    parser.add_argument("--gpu", type=int, default=None,
                        help="학습에 쓸 물리 GPU 번호 (프로세스에 이 GPU만 보이도록 제한됨)")
    parser.add_argument("--precision", choices=["bf16", "4bit"], default="bf16",
                        help="bf16(기본, 97GB GPU 기준 권장) | 4bit(작은 GPU fallback)")
    parser.add_argument("--use-vllm", action="store_true",
                        help="TRL vLLM 서버 모드로 생성 가속 (별도 GPU에 trl vllm-serve 필요)")
    parser.add_argument("--vllm-host", default="0.0.0.0")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--group-size", type=int, default=8, help="GRPO 그룹당 샘플 수 G")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="옵티마이저 스텝 상한 (스모크 테스트용, 기본: 1 epoch 전체)")
    parser.add_argument(
        "--transcripts",
        default=os.path.join("rl", "data", "transcripts.jsonl"),
        help="rl/simulate*.py가 생성한 트랜스크립트 경로",
    )
    args = parser.parse_args()

    if args.gpu is None:
        with open(os.path.join(BASE_DIR, "config", "model.yaml"), encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)
        args.gpu = model_cfg["left_gpu"] if args.side == "left" else model_cfg["right_gpu"]

    # 공용 서버 GPU 침범 방지 — torch가 CUDA를 초기화하기 전에 호출 (runtime_utils 참고).
    from runtime_utils import restrict_visible_gpus

    restrict_visible_gpus([args.gpu])

    transcript_path = (
        args.transcripts if os.path.isabs(args.transcripts) else os.path.join(BASE_DIR, args.transcripts)
    )

    train(
        side=args.side,
        reward_version=args.reward_version,
        precision=args.precision,
        use_vllm=args.use_vllm,
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        group_size=args.group_size,
        max_steps=args.max_steps,
        transcript_path=transcript_path,
    )
