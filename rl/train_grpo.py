#!/usr/bin/env python3
"""
rl/train_grpo.py — GRPO 강화학습 (Kanana-2-30B-A3B, SFT 어댑터 이어받기)
═══════════════════════════════════════════════════════════════════════

사용법:
    conda activate debate                       # transformers==4.57.6 필수 (5.x는 로드 불가)
    python rl/train_grpo.py --side left
    python rl/train_grpo.py --side right
    python rl/train_grpo.py --side left --steps 3            # 스모크 테스트
    python rl/train_grpo.py --side left --resume             # 마지막 체크포인트에서 재개
    python rl/train_grpo.py --side left --init-adapter none  # SFT 없이 새 LoRA로 시작

동작:
    1. config/model.yaml의 model_id(Kanana)를 4bit로 로드
       - rl/model_utils.py가 transformers 버전 가드 + MoE dtype 패치 + 멀티 GPU 분산 처리
       - rl/config.yaml의 left_gpus/right_gpus로 정책 모델 GPU를, judge_gpu로 Judge GPU를 배정
    2. SFT 어댑터(adapters/{side})를 이어받아 GRPO 학습 (--init-adapter none이면 새 LoRA —
       이때 LoRA 대상 모듈은 하드코딩 대신 discover_lora_target_modules()로 자동 탐색)
    3. 보상 8종(rl/rewards.py): stance/rebuttal(judge) + engagement/persona/antirep/
       semantic_echo/neutral_phrase/format(규칙·임베딩)
       - rebuttal은 기본 pairwise Bradley-Terry (rl/config.yaml rebuttal_mode)
       - judge_routing.enabled면 불확실 샘플을 API judge로 재채점
    4. adapters/{side}_grpo/ 에 최종 어댑터 저장

trl 1.5.1 주의사항 (설치본 소스에서 검증됨):
    - 새 LoRA로 시작할 때는 peft_config만 넘기고 get_peft_model을 직접 호출하지 말 것.
    - 기존 어댑터를 이어받을 때는 PeftModel.from_pretrained(is_trainable=True)로 만든 모델을
      넘기고 peft_config=None으로 둘 것 (둘 다 넘기면 ValueError). beta>0이면 트레이너가
      어댑터 사본("ref")을 만들어 KL 기준으로 쓴다.
    - max_prompt_length 파라미터 없음 → 프롬프트 길이는 build_prompts.py에서 보장.
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def _setup_env(side: str) -> tuple[list[int], int]:
    """torch import 전에 호출. CUDA_VISIBLE_DEVICES를 [정책 GPU들, Judge GPU]로 제한하고
    프로세스 내부(로컬) 인덱스를 반환한다 — 공용 서버에서 다른 사용자의 GPU를 건드리지 않기
    위한 조치이기도 하다 (HF Trainer는 보이는 GPU 전체를 세서 DataParallel을 시도할 수 있음).

    Returns
    -------
    (policy_local_gpus, judge_local_gpu)
    """
    import yaml

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    with open(os.path.join(BASE_DIR, "rl", "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train_gpus = list(cfg["left_gpus" if side == "left" else "right_gpus"])
    judge_gpu = cfg.get("judge_gpu", "auto")
    if judge_gpu == "auto":
        # 반대편 side의 첫 GPU — 순차 학습이라 학습 중엔 비어 있음
        other = list(cfg["right_gpus" if side == "left" else "left_gpus"])
        judge_gpu = other[0]
    judge_gpu = int(judge_gpu)
    if judge_gpu in train_gpus:
        # 단일 GPU 서버(예: 96GB 1장): 정책 모델과 Judge(7B 4bit)를 같은 GPU에 함께 올린다.
        # VRAM이 정책+Judge+학습 오버헤드를 모두 감당할 수 있을 때만 유효하다.
        visible = list(train_gpus)
        judge_local = train_gpus.index(judge_gpu)
        print(f"[안내] judge_gpu({judge_gpu})가 학습 GPU {train_gpus}와 같은 GPU입니다 — "
              "Judge를 정책 모델과 같은 GPU에 함께 로드합니다 (대용량 단일 GPU 모드).")
    else:
        visible = train_gpus + [judge_gpu]
        judge_local = len(train_gpus)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in visible)
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
          f"(정책=로컬 0..{len(train_gpus)-1}, Judge=로컬 {judge_local})")
    return list(range(len(train_gpus))), judge_local


def train(side: str, steps: int | None, resume: bool, init_adapter: str | None,
          policy_gpus: list[int], judge_gpu: int) -> None:
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from rl.judge import build_judge
    from rl.model_utils import discover_lora_target_modules, load_policy_model
    from rl.rewards import make_reward_funcs
    from rl.utils import DATA_DIR, load_model_config, load_prompts, load_rl_config

    cfg = load_rl_config()
    model_cfg = load_model_config()
    prompts_cfg = load_prompts()
    model_id = model_cfg["model_id"]
    max_steps = steps if steps is not None else int(cfg["max_steps"])

    data_path = os.path.join(DATA_DIR, f"{side}_prompts.jsonl")
    if not os.path.exists(data_path):
        print(f"[오류] 프롬프트 데이터가 없습니다: {data_path}")
        print("먼저 python rl/build_prompts.py 를 실행하세요.")
        sys.exit(1)

    # --init-adapter 미지정 → config/model.yaml의 side 어댑터(SFT) 사용, "none" → 새 LoRA
    if init_adapter is None:
        init_adapter = model_cfg.get(f"{side}_adapter")
    if init_adapter in ("none", "null", ""):
        init_adapter = None
    adapter_dir = os.path.join(BASE_DIR, init_adapter) if init_adapter else None

    ckpt_dir = os.path.join(BASE_DIR, "adapters", f"{side}_grpo_ckpt")
    final_dir = os.path.join(BASE_DIR, "adapters", f"{side}_grpo")

    print(f"\n[{side.upper()}] GRPO 학습 시작")
    print(f"  모델: {model_id} (4bit, 정책 GPU={policy_gpus})")
    print(f"  초기 어댑터: {adapter_dir or '없음 (새 LoRA)'}")
    print(f"  스텝: {max_steps} | 데이터: {data_path}")
    print(f"  체크포인트: {ckpt_dir} | 최종: {final_dir}\n")

    # ── 토크나이저 ───────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 정책 모델 (버전 가드 + MoE dtype 패치 + 멀티 GPU는 load_policy_model이 처리) ──
    model = load_policy_model(model_id, policy_gpus)

    peft_config = None
    if adapter_dir and os.path.isdir(adapter_dir):
        print(f"  SFT 어댑터 이어받기: {adapter_dir}")
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    else:
        if adapter_dir:
            print(f"  [경고] {adapter_dir}가 없어 새 LoRA로 시작합니다.")
        lora_r = int(cfg["lora_r"])
        target_modules = discover_lora_target_modules(model)
        print(f"  새 LoRA 대상 모듈 {len(target_modules)}개 자동 탐색됨 "
              f"(예: {target_modules[:4]} ...)")
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.0,   # RL에서는 rollout/학습 logprob 일치를 위해 dropout 제거
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )

    # ── Judge + 보상 함수 8종 ────────────────────────────────
    judge = build_judge(cfg, gpu=judge_gpu)
    reward_funcs, reward_weights = make_reward_funcs(
        side, judge, cfg, persona_text=prompts_cfg[side],
    )

    # ── 데이터셋 ─────────────────────────────────────────────
    dataset = load_dataset("json", data_files=data_path, split="train")
    print(f"  프롬프트 {len(dataset)}개 로드\n")

    # ── GRPOConfig ───────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=ckpt_dir,
        max_steps=max_steps,
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=int(cfg["warmup_steps"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        num_generations=int(cfg["num_generations"]),
        max_completion_length=int(cfg["max_completion_length"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
        beta=float(cfg["beta"]),
        mask_truncated_completions=True,
        reward_weights=reward_weights,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        save_strategy="steps",
        save_steps=int(cfg["save_steps"]),
        save_total_limit=int(cfg["save_total_limit"]),
        log_completions=True,
        num_completions_to_print=2,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    print("학습 시작...")
    trainer.train(resume_from_checkpoint=resume or None)

    # ── 최종 어댑터 저장 ─────────────────────────────────────
    os.makedirs(final_dir, exist_ok=True)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n어댑터 저장 완료: {final_dir}")
    print(f"평가: python rl/eval_grpo.py --side {side} --adapter adapters/{side}_grpo")
    print(f"배포: config/model.yaml 의 {side}_adapter 를 \"adapters/{side}_grpo\" 로 변경")


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO 강화학습 (Kanana)")
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--steps", type=int, default=None,
                        help="최대 스텝 수 (기본: rl/config.yaml의 max_steps)")
    parser.add_argument("--resume", action="store_true",
                        help="마지막 체크포인트에서 재개")
    parser.add_argument("--init-adapter", type=str, default=None,
                        help='시작 어댑터 경로 (기본: config/model.yaml의 side 어댑터, '
                             '"none"이면 새 LoRA로 시작)')
    args = parser.parse_args()

    policy_gpus, judge_local = _setup_env(args.side)   # 반드시 torch import 전에 실행
    train(args.side, args.steps, args.resume, args.init_adapter, policy_gpus, judge_local)
