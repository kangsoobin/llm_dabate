#!/usr/bin/env python3
"""
rl/check_reward_sanity.py — GRPO 학습 전 보상 신호 검증 (go/no-go 게이트)
═══════════════════════════════════════════════════════════════════════

사용법:
    python rl/build_anchor.py --side left && python rl/build_anchor.py --side right
    python rl/check_reward_sanity.py

왜 필요한가:
    v2 보상은 judge 대신 임베딩 유사도/어휘 커버리지를 쓴다 (Judge Ceiling 회피,
    docs/reward_design_v2.md §1). 그런데 참조 논문(Abdulhai et al., NeurIPS 2025,
    "Consistently Simulating Human Personas with Multi-Turn RL")은 "유사도 지표는 미묘한
    페르소나 모순을 감지하지 못할 수 있다"고 명시적으로 경고한다. 임베딩 보상이 실제로
    우리 도메인에서 변별력이 있는지 **학습을 돌리기 전에** 확인하는 것이 이 스크립트다.
    변별이 안 되면 보상이 노이즈라는 뜻이므로 GRPO를 돌려봤자 의미가 없다.

검증 항목:
    1. R_persona drift: LEFT anchor에 대해
         (a) LEFT SFT 응답(같은 페르소나)  >  (b) RIGHT SFT 응답(반대 페르소나)
         (a) LEFT SFT 응답                >  (c) 중립·양시론 응답(수기 작성)
       이 두 분리가 유의미한 마진으로 벌어지는지. RIGHT anchor도 대칭으로 확인.
    2. R_engagement coverage: 같은 질문에 대한 응답 쌍(논점 공유)이
       다른 질문에 대한 응답 쌍보다 커버리지가 높게 나오는지.

각 항목 PASS/FAIL과 수치를 출력하고, 하나라도 FAIL이면 exit code 1.
"""

from __future__ import annotations

import json
import os
import random
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import yaml  # noqa: E402

from rl.rewards.utils import SentenceEmbedder, key_point_coverage  # noqa: E402
from rl.rewards.v2_redesign import _load_anchor_texts  # noqa: E402

N_SAMPLES = 40
MARGIN = 0.03  # 평균 cos-sim 분리 최소 마진 (bge 계열은 유사도 분포가 좁아 절대값은 작게 설정)

# 우리가 SFT/RL로 억제하려는 전형적 중립·양시론 발언 (readme.md '정성 검증' 목록 기반)
NEUTRAL_RESPONSES = [
    "이 문제는 양쪽 모두 일리가 있다고 생각합니다. 진보와 보수의 주장을 균형 있게 검토해야 합니다.",
    "균형 잡힌 시각으로 보면 어느 한쪽이 옳다고 단정하기 어려운 복잡한 문제입니다.",
    "복잡한 사안인 만큼 다양한 관점을 종합적으로 고려하는 것이 중요합니다. 섣부른 결론은 위험합니다.",
    "일부는 동의합니다. 상대측 주장에도 타당한 부분이 있고, 저희 쪽 주장에도 한계가 있습니다.",
    "이 사안은 중립적으로 접근해야 합니다. 양측의 근거를 모두 살펴본 후 신중하게 판단해야 합니다.",
    "찬반을 떠나 사회적 합의가 우선입니다. 서로의 입장을 존중하며 대화로 풀어가야 한다고 봅니다.",
    "어느 쪽도 완전히 옳거나 그르지 않습니다. 절충안을 찾는 것이 현실적인 해법일 것입니다.",
    "전문가들 사이에서도 의견이 갈리는 문제라 단정적으로 말씀드리기 어렵습니다.",
]


def load_sft_records(side: str, n: int, seed: int = 42) -> list[dict]:
    """sft/data/{side}_train.jsonl에서 (question, response) 레코드 n개 샘플링."""
    path = os.path.join(BASE_DIR, "sft", "data", f"{side}_train.jsonl")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            msg = json.loads(line)["messages"]
            question = next((m["content"] for m in msg if m["role"] == "user"), "")
            response = next((m["content"] for m in msg if m["role"] == "assistant"), "")
            if question and response:
                records.append({"question": question, "response": response})
    random.Random(seed).shuffle(records)
    return records[:n]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def check_persona_drift(embedder: SentenceEmbedder, anchors: dict, left, right) -> list[tuple[str, bool, str]]:
    results = []
    for side, own, opposite in (("left", left, right), ("right", right, left)):
        anchor_vec = embedder.encode_mean(anchors[side])
        sim_own = mean([embedder.cos_sim_vec(r["response"], anchor_vec) for r in own])
        sim_opp = mean([embedder.cos_sim_vec(r["response"], anchor_vec) for r in opposite])
        sim_neu = mean([embedder.cos_sim_vec(t, anchor_vec) for t in NEUTRAL_RESPONSES])

        ok_opp = sim_own - sim_opp >= MARGIN
        ok_neu = sim_own - sim_neu >= MARGIN
        detail = f"own={sim_own:.4f}  opposite={sim_opp:.4f}  neutral={sim_neu:.4f}"
        results.append((f"R_persona[{side}] own > opposite (margin {MARGIN})", ok_opp, detail))
        results.append((f"R_persona[{side}] own > neutral  (margin {MARGIN})", ok_neu, detail))
    return results


def check_engagement_coverage(left, right) -> list[tuple[str, bool, str]]:
    """같은 질문에 대한 LEFT/RIGHT 응답 쌍 vs 무관한 질문 쌍의 커버리지 비교."""
    right_by_q = {}
    for r in right:
        right_by_q.setdefault(r["question"], r["response"])

    same_q, cross_q = [], []
    rng = random.Random(7)
    for l_rec in left:
        r_resp = right_by_q.get(l_rec["question"])
        if r_resp:
            same_q.append(key_point_coverage(r_resp, l_rec["response"]))
        other = rng.choice(right)
        if other["question"] != l_rec["question"]:
            cross_q.append(key_point_coverage(other["response"], l_rec["response"]))

    m_same, m_cross = mean(same_q), mean(cross_q)
    ok = bool(same_q) and m_same > m_cross
    detail = f"same-question={m_same:.4f} (n={len(same_q)})  cross-question={m_cross:.4f} (n={len(cross_q)})"
    return [("R_engagement coverage: 같은 질문 쌍 > 다른 질문 쌍", ok, detail)]


def main() -> None:
    with open(os.path.join(BASE_DIR, "config", "reward.yaml"), encoding="utf-8") as f:
        reward_cfg = yaml.safe_load(f)
    v2_cfg = reward_cfg.get("v2", {})

    anchors = _load_anchor_texts(v2_cfg, BASE_DIR)
    for side in ("left", "right"):
        if not any(t.strip() for t in anchors.get(side, [])):
            print(f"[오류] {side} anchor가 없습니다 — 먼저 `python rl/build_anchor.py --side {side}` 실행")
            sys.exit(1)

    embedder = SentenceEmbedder(model_id=v2_cfg.get("embedding_model", "BAAI/bge-m3"))
    print(f"임베딩 모델: {embedder.model_id}")

    left = load_sft_records("left", N_SAMPLES)
    right = load_sft_records("right", N_SAMPLES)
    print(f"SFT 샘플: LEFT {len(left)}개 / RIGHT {len(right)}개, 중립 문장 {len(NEUTRAL_RESPONSES)}개\n")

    results = []
    results += check_persona_drift(embedder, anchors, left, right)
    results += check_engagement_coverage(left, right)

    print("─" * 72)
    n_fail = 0
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"[{status}] {name}\n        {detail}")
    print("─" * 72)

    if n_fail:
        print(f"\n{n_fail}개 항목 FAIL — 보상 신호가 변별력이 없습니다. GRPO를 돌리기 전에")
        print("임베딩 모델 교체(config/reward.yaml v2.embedding_model), anchor 샘플 수 증가,")
        print("또는 judge 보조 신호 활성화(persona.judge_aux_weight)를 검토하세요.")
        sys.exit(1)
    print("\n모든 항목 PASS — GRPO 학습을 진행해도 좋습니다.")


if __name__ == "__main__":
    main()
