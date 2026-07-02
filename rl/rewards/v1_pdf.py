"""
rl/rewards/v1_pdf.py
──────────────────────
`보상 설계.pdf` (강수빈) 원안을 그대로 구현한 보상 컴포넌트.

1. 성향 일관성 보상 R_stance  = -S(y)  (LEFT) / +S(y) (RIGHT), S(y) ∈ [-1,1]
2. 반박 품질 보상   R_rebuttal = (Q(y) - 1) / 4,               Q(y) ∈ [1,5]
3. 자기 반복 패널티 R_antirep  = 0 (J<τ) / -γ·(J-τ) (J>=τ),    J(y,H) ∈ [0,1]

S, Q는 모두 외부 Judge 모델 평가값이므로 이 버전은 judge가 필수다.
"""

from __future__ import annotations

from .base import DebateTurnSample
from .judge import Judge
from .utils import jaccard_similarity


class StanceReward:
    """R_stance(y) = -S(y) (LEFT) / +S(y) (RIGHT)."""

    name = "stance"

    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    def __call__(self, sample: DebateTurnSample) -> float:
        s = self.judge.score_stance(sample.response)
        return -s if sample.side == "left" else s


class RebuttalReward:
    """R_rebuttal(y) = (Q(y) - 1) / 4."""

    name = "rebuttal"

    def __init__(self, judge: Judge) -> None:
        self.judge = judge

    def __call__(self, sample: DebateTurnSample) -> float:
        if not sample.opponent_response:
            return 0.0
        q = self.judge.score_rebuttal(sample.opponent_response, sample.response)
        return (q - 1) / 4


class AntiRepetitionReward:
    """
    R_antirep(y) = 0                     if J(y,H) < τ
    R_antirep(y) = -γ · (J(y,H) - τ)     if J(y,H) >= τ
    """

    name = "antirep"

    def __init__(self, tau: float = 0.25, gamma: float = 1.5) -> None:
        self.tau = tau
        self.gamma = gamma

    def __call__(self, sample: DebateTurnSample) -> float:
        j = jaccard_similarity(sample.response, sample.own_history)
        if j < self.tau:
            return 0.0
        return -self.gamma * (j - self.tau)


def build_v1_components(judge: Judge | None, cfg: dict) -> list[tuple]:
    """cfg == config/reward.yaml 의 v1 섹션. 반환값: [(component, weight), ...]"""
    if judge is None:
        raise ValueError(
            "reward version=v1(보상 설계.pdf 원안)은 judge가 필수입니다. "
            "config/reward.yaml의 judge.backend를 local 또는 api로 설정하세요."
        )
    weights = cfg.get("weights", {})
    antirep_cfg = cfg.get("antirep", {})
    return [
        (StanceReward(judge), weights.get("stance", 1.0)),
        (RebuttalReward(judge), weights.get("rebuttal", 1.0)),
        (
            AntiRepetitionReward(
                tau=antirep_cfg.get("tau", 0.25),
                gamma=antirep_cfg.get("gamma", 1.5),
            ),
            weights.get("antirep", 1.0),
        ),
    ]
