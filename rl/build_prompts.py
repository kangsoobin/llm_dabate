#!/usr/bin/env python3
"""
rl/build_prompts.py — GRPO 프롬프트 데이터셋 빌더
═══════════════════════════════════════════════════

사용법:
    conda activate debate
    python rl/build_prompts.py                # 전체 (90문항 × 2시드 × 3라운드, 수 시간)
    python rl/build_prompts.py --questions 3  # 스모크 테스트
    python rl/build_prompts.py --slice-only   # 기존 transcripts.jsonl에서 슬라이싱만 재실행

동작:
    1. sft/questions.yaml 100문항 중 10문항을 홀드아웃(등간격)으로 분리 → rl/data/heldout.yaml
    2. config/model.yaml의 SFT 어댑터(adapters/left·right, Kanana 기준)로 3라운드 미니 토론을
       오프라인 생성 → rl/data/transcripts.jsonl (재실행 시 이어서 진행)
       - 상대 발언도 SFT 어댑터로 생성 — GRPO가 이 어댑터를 이어받아 학습하므로(PLAN.md)
         롤아웃 시점의 상대 발언 분포를 학습 시점과 일치시킨다.
       - GPU 배치는 rl/config.yaml의 left_gpus/right_gpus를 따른다:
         · 두 목록이 겹치면(예: 96GB 단일 GPU 서버, 둘 다 [0]) **공유 모드** — base 모델을
           한 번만 올리고 LEFT/RIGHT LoRA 어댑터를 턴마다 set_adapter로 갈아끼운다.
           Kanana 인스턴스 2개(2×~60GB)는 한 장에 안 들어가기 때문. 생성은 순차 실행.
         · 겹치지 않으면 기존 방식 — 인스턴스 2개를 각 GPU에 올리고 토론 2개를 반 라운드
           어긋나게 병렬 실행.
    3. 트랜스크립트를 라운드별 학습 프롬프트로 슬라이싱 → rl/data/{left,right}_prompts.jsonl
       - 프롬프트 포맷은 core/session.py의 _build_message를 그대로 재사용
       - 1,400 토큰 초과 시 상대 발언/자기 과거 발언을 단계적으로 트리밍

주의: Kanana 로딩은 rl/model_utils.py가 transformers 버전 가드(4.57.6 필수) + MoE dtype
패치를 처리한다. bnb 4bit이 routed expert를 양자화 못 해 인스턴스 1개가 24GB를 훌쩍 넘는다
(2026-07-04 실측 — 3090 24GB 단일 로딩 OOM).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import yaml

from rl.utils import (
    DATA_DIR,
    build_message,
    count_prompt_tokens,
    load_model_config,
    load_prompts,
    load_questions,
    load_rl_config,
    split_questions,
    stateless_generate,
    trim_sentences,
)

TRANSCRIPT_PATH = os.path.join(DATA_DIR, "transcripts.jsonl")
HELDOUT_PATH = os.path.join(DATA_DIR, "heldout.yaml")


# ────────────────────────────────────────────────────────────
# 미니 토론 (한 질문 × 한 시드)
# ────────────────────────────────────────────────────────────

class MiniDebate:
    """양 에이전트의 히스토리를 자체 관리하며 반 라운드씩 진행한다."""

    def __init__(self, qid, question, seed, temperature, agents, prompts, rounds, max_new_tokens):
        self.qid = qid
        self.question = question
        self.seed = seed
        self.temperature = temperature
        self.agents = agents          # {"left": BaseAgent, "right": BaseAgent}
        self.prompts = prompts        # {"left": str, "right": str}
        self.n_rounds = rounds
        self.max_new_tokens = max_new_tokens
        self.halfstep = 0
        self.hist = {"left": [], "right": []}   # 에이전트별 user/assistant 턴
        self.rounds = [{"round": r + 1, "left": None, "right": None} for r in range(rounds)]

    @property
    def done(self) -> bool:
        return self.halfstep >= 2 * self.n_rounds

    def step(self) -> None:
        r = self.halfstep // 2
        side = "left" if self.halfstep % 2 == 0 else "right"
        if side == "left":
            opponent = self.rounds[r - 1]["right"] if r > 0 else ""
        else:
            opponent = self.rounds[r]["left"]
        opp_name = self.agents["right" if side == "left" else "left"].name

        user_msg = build_message(self.question, opponent, opp_name, side)
        messages = (
            [{"role": "system", "content": self.prompts[side]}]
            + self.hist[side]
            + [{"role": "user", "content": user_msg}]
        )
        agent = self.agents[side]
        if hasattr(agent, "activate"):
            agent.activate()   # 공유 모드: 이 side의 LoRA 어댑터로 스왑
        text = stateless_generate(
            agent, messages,
            max_new_tokens=self.max_new_tokens, temperature=self.temperature,
        )
        self.hist[side] += [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": text},
        ]
        self.rounds[r][side] = text
        self.halfstep += 1

    def to_record(self) -> dict:
        return {
            "qid": self.qid,
            "seed": self.seed,
            "temperature": self.temperature,
            "question": self.question,
            "rounds": self.rounds,
        }


class SharedPersonaAgent:
    """단일 base 모델 + LoRA 어댑터 스왑으로 페르소나 하나를 표현하는 경량 에이전트.

    stateless_generate가 요구하는 인터페이스(tokenizer/model/gpu_id/gen_config)를 제공하고,
    activate()가 이 side의 어댑터를 활성화한다. 두 인스턴스가 같은 model 객체를 공유하므로
    병렬 생성은 불가 — run_debates(parallel=False)로 순차 실행해야 한다.
    """

    def __init__(self, side, name, model, tokenizer, gen_config, gpu_id, has_adapters):
        self.side = side
        self.name = name
        self.model = model
        self.tokenizer = tokenizer
        self.gen_config = gen_config
        self.gpu_id = gpu_id
        self._has_adapters = has_adapters

    def activate(self) -> None:
        if self._has_adapters:
            self.model.set_adapter(self.side)


def load_shared_agents(model_cfg, gen_cfg, gpus: list[int]) -> dict:
    """base 모델 1개 + LEFT/RIGHT 어댑터 2개를 로드해 어댑터 스왑 에이전트 쌍을 만든다."""
    from transformers import AutoTokenizer

    from rl.model_utils import load_policy_model

    model_id = model_cfg["model_id"]
    print(f"공유 base 모델 로딩 중 (GPU {gpus}, 어댑터 스왑 모드)...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_policy_model(model_id, gpus)
    model.eval()

    left_dir = model_cfg.get("left_adapter")
    right_dir = model_cfg.get("right_adapter")
    left_path = os.path.join(BASE_DIR, left_dir) if left_dir else None
    right_path = os.path.join(BASE_DIR, right_dir) if right_dir else None
    has_adapters = bool(
        left_path and right_path and os.path.isdir(left_path) and os.path.isdir(right_path)
    )
    if has_adapters:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, left_path, adapter_name="left")
        model.load_adapter(right_path, adapter_name="right")
        model.eval()
        print(f"  LEFT/RIGHT LoRA 어댑터 로드 완료 — 턴마다 set_adapter로 스왑")
    else:
        print("  [경고] 어댑터가 없어 순수 base로 양쪽 발언을 생성합니다.")

    gpu_id = gpus[0]
    return {
        "left": SharedPersonaAgent("left", "이진보", model, tokenizer, gen_cfg, gpu_id, has_adapters),
        "right": SharedPersonaAgent("right", "김보수", model, tokenizer, gen_cfg, gpu_id, has_adapters),
    }


def run_debates(jobs: list[MiniDebate], out_path: str, parallel: bool = True) -> None:
    """미니 토론들을 실행해 out_path에 기록한다.

    parallel=True  — 인스턴스 2개(GPU 분리) 전제. 두 슬롯을 반 라운드 어긋나게 돌려
                     LEFT/RIGHT 생성을 겹친다. 슬롯 0은 짝수 틱, 슬롯 1은 홀수 틱에만 새
                     토론을 시작하므로 매 틱에서 두 슬롯은 항상 서로 다른 에이전트를 쓴다.
    parallel=False — 공유 모델(어댑터 스왑) 전제. 순차 실행 (같은 모델 동시 호출 금지).
    """
    if not parallel:
        total = len(jobs)
        t0 = time.time()
        with open(out_path, "a", encoding="utf-8") as fout:
            for k, debate in enumerate(jobs, 1):
                while not debate.done:
                    debate.step()
                fout.write(json.dumps(debate.to_record(), ensure_ascii=False) + "\n")
                fout.flush()
                elapsed = time.time() - t0
                eta = elapsed / k * (total - k)
                print(f"  토론 {k}/{total} 완료 (Q{debate.qid} seed{debate.seed}) "
                      f"| 경과 {elapsed/60:.0f}분, 남은 예상 {eta/60:.0f}분", flush=True)
        return

    pending = deque(jobs)
    slots: list[MiniDebate | None] = [None, None]
    total = len(jobs)
    completed = 0
    t0 = time.time()
    tick = 0

    with ThreadPoolExecutor(max_workers=2) as executor, open(out_path, "a", encoding="utf-8") as fout:
        while pending or any(s is not None for s in slots):
            for s in (0, 1):
                if slots[s] is None and pending and tick % 2 == s:
                    slots[s] = pending.popleft()
            futures = {s: executor.submit(slots[s].step) for s in (0, 1) if slots[s] is not None}
            for s, future in futures.items():
                future.result()
                if slots[s].done:
                    fout.write(json.dumps(slots[s].to_record(), ensure_ascii=False) + "\n")
                    fout.flush()
                    completed += 1
                    elapsed = time.time() - t0
                    eta = elapsed / completed * (total - completed)
                    print(f"  토론 {completed}/{total} 완료 (Q{slots[s].qid} seed{slots[s].seed}) "
                          f"| 경과 {elapsed/60:.0f}분, 남은 예상 {eta/60:.0f}분", flush=True)
                    slots[s] = None
            tick += 1


# ────────────────────────────────────────────────────────────
# 트랜스크립트 → 학습 프롬프트 슬라이싱
# ────────────────────────────────────────────────────────────

# (상대 발언 문장 수, 자기 과거 발언 문장 수) 트리밍 단계 — 초과 시 다음 단계로
TRIM_LEVELS = [(3, None), (2, None), (2, 2)]


def _make_prompt(question, side, sys_prompt, opp_name, cur_opp, prev_opp, own_prev,
                 opp_sents, own_sents):
    cur_msg = build_message(question, trim_sentences(cur_opp, opp_sents), opp_name, side) \
        if cur_opp else question
    if own_prev is None:
        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": cur_msg},
        ]
    prev_msg = build_message(question, trim_sentences(prev_opp, opp_sents), opp_name, side) \
        if prev_opp else question
    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prev_msg},
        {"role": "assistant", "content": trim_sentences(own_prev, own_sents)},
        {"role": "user", "content": cur_msg},
    ]


def slice_transcripts(records, side, sys_prompt, opp_name, tokenizer, cfg):
    """미니 토론 기록을 라운드별 GRPO 프롬프트 행으로 변환한다."""
    max_tokens = int(cfg["max_prompt_tokens"])
    first_seed = min(r["seed"] for r in records) if records else 0
    rows, dropped = [], 0

    for rec in records:
        question, rounds = rec["question"], rec["rounds"]
        for r in range(1, len(rounds) + 1):
            # LEFT는 상대의 직전 라운드 발언, RIGHT는 같은 라운드 LEFT 발언을 본다
            if side == "left":
                cur_opp = rounds[r - 2]["right"] if r > 1 else ""
                prev_opp = rounds[r - 3]["right"] if r > 2 else ""
            else:
                cur_opp = rounds[r - 1]["left"]
                prev_opp = rounds[r - 2]["left"] if r > 1 else ""

            # LEFT 1라운드 프롬프트는 시드와 무관하게 동일 → 첫 시드만 유지
            if side == "left" and r == 1 and rec["seed"] != first_seed:
                continue

            own_prev = rounds[r - 2][side] if r > 1 else None
            own_history = [rounds[i][side] for i in range(r - 1)]

            for opp_sents, own_sents in TRIM_LEVELS:
                prompt = _make_prompt(question, side, sys_prompt, opp_name,
                                      cur_opp, prev_opp, own_prev, opp_sents, own_sents)
                if count_prompt_tokens(tokenizer, prompt) <= max_tokens:
                    rows.append({
                        "prompt": prompt,
                        "side": side,
                        "qid": rec["qid"],
                        "seed": rec["seed"],
                        "round": r,
                        "question": question,
                        # judge의 반박 평가 대상 = 모델이 실제로 본 (트리밍된) 상대 발언
                        "opponent_statement": trim_sentences(cur_opp, opp_sents) if cur_opp else "",
                        "own_history": own_history,
                    })
                    break
            else:
                dropped += 1
    return rows, dropped


def write_prompts(records, tokenizer, prompts_cfg, names, cfg):
    for side in ("left", "right"):
        opp = "right" if side == "left" else "left"
        rows, dropped = slice_transcripts(
            records, side, prompts_cfg[side], names[opp], tokenizer, cfg,
        )
        out_path = os.path.join(DATA_DIR, f"{side}_prompts.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        lengths = sorted(count_prompt_tokens(tokenizer, r["prompt"]) for r in rows)
        by_round = {r: sum(1 for x in rows if x["round"] == r) for r in (1, 2, 3)}
        p = lambda q: lengths[int(len(lengths) * q)] if lengths else 0
        print(f"[{side.upper()}] {len(rows)}행 저장 → {out_path}")
        print(f"  라운드 분포: {by_round} | 탈락 {dropped}행")
        print(f"  프롬프트 토큰: p50={p(0.5)}, p95={p(0.95)}, max={lengths[-1] if lengths else 0}")


# ────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────

def main(args) -> None:
    cfg = load_rl_config()
    model_cfg = load_model_config()
    prompts_cfg = load_prompts()
    os.makedirs(DATA_DIR, exist_ok=True)

    # 질문 분리 + 홀드아웃 저장
    questions = load_questions()
    train_qs, heldout = split_questions(questions, int(cfg["n_heldout"]))
    with open(HELDOUT_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump({"heldout_questions": heldout}, f, allow_unicode=True)
    print(f"홀드아웃 {len(heldout)}문항 저장 → {HELDOUT_PATH}")

    if args.questions:
        train_qs = train_qs[: args.questions]

    temperatures = list(cfg["seed_temperatures"])[: int(cfg["n_seeds"])]

    if not args.slice_only:
        # 이미 완료된 (qid, seed)는 건너뛴다 (재실행 시 이어서 진행)
        done = set()
        if os.path.exists(TRANSCRIPT_PATH):
            with open(TRANSCRIPT_PATH, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    done.add((rec["qid"], rec["seed"]))

        gen_cfg = {
            "max_new_tokens": int(cfg["offline_max_new_tokens"]),
            "temperature": 1.0,
            "top_p": model_cfg["top_p"],
            "repetition_penalty": model_cfg["repetition_penalty"],
        }
        # GPU 목록이 겹치면(단일 대형 GPU 서버) 공유 모드, 아니면 인스턴스 2개 병렬 모드
        left_gpus = [int(g) for g in cfg.get("left_gpus", [model_cfg.get("left_gpu", 0)])]
        right_gpus = [int(g) for g in cfg.get("right_gpus", [model_cfg.get("right_gpu", 1)])]
        shared_mode = bool(set(left_gpus) & set(right_gpus))

        if shared_mode:
            agents = load_shared_agents(model_cfg, gen_cfg, left_gpus)
        else:
            from agents import create_left_agent, create_right_agent

            # SFT 어댑터로 상대 발언을 생성한다 — GRPO가 이 어댑터를 이어받아 학습하므로
            # (train_grpo.py 기본 동작) 롤아웃 시점의 상대 발언 분포를 학습 시점과 일치시킨다.
            left = create_left_agent(dict(model_cfg, left_gpu=left_gpus[0]), gen_cfg,
                                     prompts_cfg["left"],
                                     adapter_path=model_cfg.get("left_adapter"))
            right = create_right_agent(dict(model_cfg, right_gpu=right_gpus[0]), gen_cfg,
                                       prompts_cfg["right"],
                                       adapter_path=model_cfg.get("right_adapter"))
            print(f"SFT 에이전트 로딩 중 (LEFT adapter={model_cfg.get('left_adapter')}, "
                  f"RIGHT adapter={model_cfg.get('right_adapter')})...")
            left.load()
            right.load()
            agents = {"left": left, "right": right}
        prompts = {"left": prompts_cfg["left"], "right": prompts_cfg["right"]}

        jobs = [
            MiniDebate(qid, q, seed, temp, agents, prompts,
                       int(cfg["debate_rounds"]), int(cfg["offline_max_new_tokens"]))
            for qid, q in train_qs
            for seed, temp in enumerate(temperatures)
            if (qid, seed) not in done
        ]
        mode_desc = "공유 모델 순차" if shared_mode else "인스턴스 2개 병렬"
        print(f"미니 토론 {len(jobs)}건 생성 시작 ({mode_desc}, 완료된 {len(done)}건 건너뜀)...")
        run_debates(jobs, TRANSCRIPT_PATH, parallel=not shared_mode)

    # 슬라이싱
    if not os.path.exists(TRANSCRIPT_PATH):
        print(f"[오류] 트랜스크립트가 없습니다: {TRANSCRIPT_PATH}")
        sys.exit(1)
    with open(TRANSCRIPT_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    train_qids = {qid for qid, _ in train_qs}
    records = [r for r in records if r["qid"] in train_qids]
    print(f"\n트랜스크립트 {len(records)}건 슬라이싱...")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"], trust_remote_code=True)
    names = {"left": "이진보", "right": "김보수"}
    write_prompts(records, tokenizer, prompts_cfg, names, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO 프롬프트 데이터셋 빌더")
    parser.add_argument("--questions", type=int, default=None,
                        help="사용할 학습 질문 수 제한 (스모크 테스트용)")
    parser.add_argument("--slice-only", action="store_true",
                        help="생성 없이 기존 transcripts.jsonl에서 슬라이싱만 수행")
    main(parser.parse_args())
