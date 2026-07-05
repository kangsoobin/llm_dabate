#!/usr/bin/env python3
"""
rl/eval_grpo.py — 어댑터 평가 (readme.md 검증 프로토콜)
═══════════════════════════════════════════════════════

사용법:
    conda activate debate
    python rl/eval_grpo.py --side left  --adapter adapters/left_grpo
    python rl/eval_grpo.py --side left  --adapter none            --tag base
    python rl/eval_grpo.py --side right --adapter adapters/right_grpo

평가 3종 (홀드아웃 질문 사용, rl/data/heldout.yaml):
    1. 스탠스 곡선: 토픽 N개 × 10라운드 토론(상대는 순수 base) 후 매 라운드
       "입장을 0~10으로만 답하라" 자기 평가 → 5로 수렴하지 않아야 성공
    2. 중립 표현 카운트: 토론 발언에서 블랙리스트 표현 검출
    3. Before/After: 홀드아웃 전 질문 1·2라운드 답변 생성 + Judge 채점

결과: logs/rl_eval_{side}_{tag}.json + 콘솔 요약
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

RATING_QUESTION = (
    "지금까지의 토론 주제에 대한 당신의 현재 입장을 0에서 10 사이의 숫자 하나로만 답하십시오. "
    "0은 완전 반대(진보 편), 10은 완전 찬성(보수 편)을 뜻합니다. 숫자만 출력하세요."
)


def parse_rating(text: str) -> int | None:
    match = re.search(r"\d+", text)
    if not match:
        return None
    return max(0, min(10, int(match.group(0))))


def free_agent(agent) -> None:
    import torch
    agent.model = None
    gc.collect()
    torch.cuda.empty_cache()


def main(args) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import yaml

    from agents import create_left_agent, create_right_agent
    from rl.rewards import NEUTRAL_PHRASES
    from rl.utils import (
        DATA_DIR, build_message, load_model_config, load_prompts,
        load_rl_config, stateless_generate,
    )

    cfg = load_rl_config()
    model_cfg = load_model_config()
    prompts_cfg = load_prompts()

    heldout_path = os.path.join(DATA_DIR, "heldout.yaml")
    if not os.path.exists(heldout_path):
        print(f"[오류] 홀드아웃 파일이 없습니다: {heldout_path} — 먼저 build_prompts.py 실행")
        sys.exit(1)
    with open(heldout_path, encoding="utf-8") as f:
        heldout = yaml.safe_load(f)["heldout_questions"]

    side, opp_side = args.side, ("right" if args.side == "left" else "left")
    adapter = None if args.adapter in (None, "none") else os.path.join(BASE_DIR, args.adapter)
    tag = args.tag or (os.path.basename(adapter) if adapter else "base")

    # ── 모델 로드: 평가 대상(GPU0) + SFT 상대(GPU1) ─────────
    gen_cfg = {
        "max_new_tokens": int(cfg["max_completion_length"]),
        "temperature": model_cfg["temperature"],
        "top_p": model_cfg["top_p"],
        "repetition_penalty": model_cfg["repetition_penalty"],
    }
    factories = {"left": create_left_agent, "right": create_right_agent}
    policy_cfg = dict(model_cfg, left_gpu=0, right_gpu=0)
    opp_cfg = dict(model_cfg, left_gpu=1, right_gpu=1)

    print(f"평가 대상 로딩: {side.upper()} + {args.adapter} (GPU0)")
    policy = factories[side](policy_cfg, gen_cfg, prompts_cfg[side], adapter_path=adapter)
    policy.load()
    # 상대는 순수 base — 모든 조건(base/GRPO 어댑터)이 동일한 상대와 토론하도록 고정
    print(f"상대 로딩: {opp_side.upper()} 순수 base (GPU1)")
    opponent = factories[opp_side](opp_cfg, gen_cfg, prompts_cfg[opp_side], adapter_path=None)
    opponent.load()

    results = {
        "side": side, "adapter": args.adapter, "tag": tag,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "stance_debates": [], "side_by_side": [],
    }

    # ── 1+2. 스탠스 곡선 + 중립 표현 ─────────────────────────
    for topic in heldout[: args.topics]:
        policy.reset_history()
        opponent.reset_history()
        ratings, statements = [], []
        print(f"\n[토론] {topic}")
        for rnd in range(1, args.rounds + 1):
            # 배포와 동일하게 LEFT가 먼저 발언
            first, second = (policy, opponent) if side == "left" else (opponent, policy)
            first_msg = build_message(topic, second.last_response, second.name, first.side)
            first_resp = first.generate(first_msg)
            second_msg = build_message(topic, first_resp, first.name, second.side)
            second.generate(second_msg)

            statements.append(policy.last_response)
            rating_msgs = (
                [{"role": "system", "content": prompts_cfg[side]}]
                + policy.history
                + [{"role": "user", "content": RATING_QUESTION}]
            )
            rating_text = stateless_generate(policy, rating_msgs, max_new_tokens=8, greedy=True)
            ratings.append(parse_rating(rating_text))
            print(f"  라운드 {rnd}: 자기평가={ratings[-1]}", flush=True)

        neutral_hits = sum(s.count(p) for s in statements for p in NEUTRAL_PHRASES)
        results["stance_debates"].append({
            "topic": topic, "ratings": ratings,
            "neutral_phrase_hits": neutral_hits, "statements": statements,
        })

    # ── 3. Before/After 소재: 홀드아웃 전 질문 1·2라운드 답변 ──
    print("\n[Before/After] 홀드아웃 질문별 1·2라운드 답변 생성...")
    for question in heldout:
        policy.reset_history()
        opponent.reset_history()
        if side == "left":
            r1 = policy.generate(build_message(question, "", opponent.name, side))
            opp_r1 = opponent.generate(build_message(question, r1, policy.name, opp_side))
        else:
            left_r1 = opponent.generate(build_message(question, "", policy.name, opp_side))
            r1 = policy.generate(build_message(question, left_r1, opponent.name, side))
            opp_r1 = opponent.generate(build_message(question, r1, policy.name, opp_side))
        r2 = policy.generate(build_message(question, opp_r1, opponent.name, side))
        results["side_by_side"].append({
            "question": question, "r1": r1, "opponent": opp_r1, "r2": r2,
        })
        print(f"  완료: {question[:40]}...", flush=True)

    # ── Judge 채점 (상대 모델을 내리고 GPU1에 Judge 로드) ────
    print("\n상대 모델 해제 후 Judge 로딩...")
    free_agent(opponent)
    from rl.judge import JudgeClient
    judge = JudgeClient(model_id=cfg["judge_model_id"], gpu=1,
                        max_new_tokens=int(cfg["judge_max_new_tokens"]),
                        batch_size=int(cfg["judge_batch_size"]))

    items = [
        {"question": row["question"], "opponent": row["opponent"], "reply": row["r2"]}
        for row in results["side_by_side"]
    ]
    scores = judge.score_batch(items)
    for row, score in zip(results["side_by_side"], scores):
        row["judge"] = score

    debate_items = [
        {"question": d["topic"], "opponent": "", "reply": s}
        for d in results["stance_debates"] for s in d["statements"]
    ]
    debate_scores = judge.score_batch(debate_items)
    idx = 0
    for d in results["stance_debates"]:
        d["judge_stances"] = [debate_scores[idx + i]["stance"] for i in range(len(d["statements"]))]
        idx += len(d["statements"])

    # ── 저장 + 요약 ──────────────────────────────────────────
    log_dir = os.path.join(BASE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    out_path = os.path.join(log_dir, f"rl_eval_{side}_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    sign = -1.0 if side == "left" else 1.0
    print(f"\n{'=' * 60}\n평가 요약 — {side.upper()} / {tag}\n{'=' * 60}")
    print("\n[스탠스 곡선] (LEFT는 0에, RIGHT는 10에 가까울수록 좋음)")
    for d in results["stance_debates"]:
        print(f"  {d['topic'][:30]}...: {d['ratings']}")
        mean_judge = sum(d["judge_stances"]) / len(d["judge_stances"])
        print(f"    Judge stance 평균: {mean_judge:+.2f} | 중립 표현: {d['neutral_phrase_hits']}회")
    r2_stance = [row["judge"]["stance"] for row in results["side_by_side"]]
    r2_rebuttal = [row["judge"]["rebuttal"] for row in results["side_by_side"]]
    n = len(results["side_by_side"])
    print(f"\n[Before/After — 2라운드 답변 Judge 채점, {n}문항]")
    print(f"  성향 보상(부호 반영) 평균: {sign * sum(r2_stance) / n:+.3f}")
    print(f"  반박 품질 평균: {sum(r2_rebuttal) / n:.2f} / 5")
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO 어댑터 평가")
    parser.add_argument("--side", choices=["left", "right"], required=True)
    parser.add_argument("--adapter", type=str, required=True,
                        help='어댑터 경로 (예: adapters/left_grpo) 또는 "none"(순수 base)')
    parser.add_argument("--tag", type=str, default=None,
                        help="결과 파일 태그 (기본: 어댑터 폴더명)")
    parser.add_argument("--topics", type=int, default=3, help="스탠스 곡선 토픽 수")
    parser.add_argument("--rounds", type=int, default=10, help="토론 라운드 수")
    main(parser.parse_args())
