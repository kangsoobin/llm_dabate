"""
rl/rewards/v2_redesign.py
────────────────────────────
docs/reward_design_v2.md 에서 설계한 보상 컴포넌트.

R_persona(§3.1) / R_engagement(§3.2) / R_diversity(§3.3) / R_novelty(§3.4) / R_grounding(§3.5, 기본 비활성)

judge 없이도(judge=None) 전부 동작하도록 설계됨 — Judge Ceiling 의존도를 낮추는 것이
이 재설계의 핵심 동기 중 하나 (docs/reward_design_v2.md §1).
"""

from __future__ import annotations

from typing import Callable, Optional

from .base import DebateTurnSample
from .judge import Judge
from .utils import SentenceEmbedder, jaccard_similarity, key_point_coverage


class PersonaConsistencyReward:
    """docs/reward_design_v2.md §3.1 — 기준 페르소나 유지 + 상대 쪽으로의 동조(sycophancy) 패널티."""

    name = "persona"

    def __init__(
        self,
        embedder: SentenceEmbedder,
        anchor_texts: dict[str, str],
        lambda_syc: float = 1.0,
        judge: Optional[Judge] = None,
        judge_aux_weight: float = 0.0,
    ) -> None:
        self.embedder = embedder
        self.anchor_texts = anchor_texts  # {"left": "...", "right": "..."} — rl/build_anchor.py로 생성
        self.lambda_syc = lambda_syc
        self.judge = judge
        self.judge_aux_weight = judge_aux_weight

    def __call__(self, sample: DebateTurnSample) -> float:
        anchor = self.anchor_texts.get(sample.side, "")
        drift = max(0.0, min(1.0, self.embedder.cos_sim(sample.response, anchor))) if anchor else 0.0

        sycophancy = 0.0
        if sample.opponent_response and sample.own_history:
            prev_own = sample.own_history[-1]
            sim_now = self.embedder.cos_sim(sample.response, sample.opponent_response)
            sim_prev = self.embedder.cos_sim(prev_own, sample.opponent_response)
            sycophancy = sim_now - sim_prev  # 이번 턴에 상대 쪽으로 더 가까워졌는가

        reward = drift - self.lambda_syc * max(0.0, sycophancy)

        if self.judge is not None and self.judge_aux_weight > 0:
            s = self.judge.score_stance(sample.response)
            signed_s = -s if sample.side == "left" else s
            reward += self.judge_aux_weight * signed_s

        return reward


class EngagementReward:
    """docs/reward_design_v2.md §3.2 — key-point 커버리지(judge-free) + judge 1~5점(선택적 보조)."""

    name = "engagement"

    def __init__(self, judge: Optional[Judge] = None, judge_weight: float = 0.0) -> None:
        self.judge = judge
        self.judge_weight = judge_weight if judge is not None else 0.0

    def __call__(self, sample: DebateTurnSample) -> float:
        if not sample.opponent_response:
            return 0.0
        coverage = key_point_coverage(sample.opponent_response, sample.response)
        if self.judge is None or self.judge_weight <= 0:
            return coverage
        q = self.judge.score_rebuttal(sample.opponent_response, sample.response)
        judge_score = (q - 1) / 4
        return (1 - self.judge_weight) * coverage + self.judge_weight * judge_score


class DiversityReward:
    """docs/reward_design_v2.md §3.3 — 상대 발언·자기 과거 발언과의 임베딩 유사도 패널티 (슬라이드 13 R1)."""

    name = "diversity"

    def __init__(self, embedder: SentenceEmbedder) -> None:
        self.embedder = embedder

    def __call__(self, sample: DebateTurnSample) -> float:
        sim_opponent = (
            self.embedder.cos_sim(sample.response, sample.opponent_response)
            if sample.opponent_response
            else 0.0
        )
        sim_self = 0.0
        for past in sample.own_history:
            sim_self = max(sim_self, self.embedder.cos_sim(sample.response, past))
        return 1.0 - max(sim_opponent, sim_self)


class NoveltyReward:
    """docs/reward_design_v2.md §3.4 — v1 R_antirep과 동일 수식(문자 그대로의 반복에 대한 안전망)."""

    name = "novelty"

    def __init__(self, tau: float = 0.25, gamma: float = 1.5) -> None:
        self.tau = tau
        self.gamma = gamma

    def __call__(self, sample: DebateTurnSample) -> float:
        j = jaccard_similarity(sample.response, sample.own_history)
        if j < self.tau:
            return 0.0
        return -self.gamma * (j - self.tau)


class GroundingReward:
    """
    docs/reward_design_v2.md §3.5 — sample.evidence가 주어질 때만 동작.
    근거 검색(RAG) 파이프라인이 아직 없어 기본 가중치 0(비활성)으로 config에 지정되어 있다.
    nli_scorer: callable(evidence: str, claim: str) -> float, 미주입 시 항상 0 반환.
    """

    name = "grounding"

    def __init__(self, nli_scorer: Optional[Callable[[str, str], float]] = None) -> None:
        self.nli_scorer = nli_scorer

    def __call__(self, sample: DebateTurnSample) -> float:
        if not sample.evidence or self.nli_scorer is None:
            return 0.0
        return self.nli_scorer(sample.evidence, sample.response)


def build_v2_components(judge: Optional[Judge], cfg: dict) -> list[tuple]:
    """cfg == config/reward.yaml 의 v2 섹션. 반환값: [(component, weight), ...]"""
    weights = cfg.get("weights", {})
    embedder = SentenceEmbedder(model_id=cfg.get("embedding_model", "jhgan/ko-sroberta-multitask"))
    persona_cfg = cfg.get("persona", {})
    engagement_cfg = cfg.get("engagement", {})
    novelty_cfg = cfg.get("novelty", {})

    return [
        (
            PersonaConsistencyReward(
                embedder=embedder,
                anchor_texts=cfg.get("anchor_texts", {}),
                lambda_syc=persona_cfg.get("lambda_syc", 1.0),
                judge=judge,
                judge_aux_weight=persona_cfg.get("judge_aux_weight", 0.0),
            ),
            weights.get("persona", 1.0),
        ),
        (
            EngagementReward(judge=judge, judge_weight=engagement_cfg.get("judge_weight", 0.0)),
            weights.get("engagement", 1.0),
        ),
        (DiversityReward(embedder=embedder), weights.get("diversity", 0.8)),
        (
            NoveltyReward(tau=novelty_cfg.get("tau", 0.25), gamma=novelty_cfg.get("gamma", 1.5)),
            weights.get("novelty", 0.5),
        ),
        (GroundingReward(), weights.get("grounding", 0.0)),
    ]
