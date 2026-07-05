"""
rl/rewards/v3_multiagent.py
──────────────────────────
v2 reward를 기반으로, 2026-07-04 평가에서 드러난 약점을 직접 겨냥한 멀티에이전트 GRPO 보상.

v3에서는 네 원안의 세 축을 유지하되 이름을 더 분명히 나눴다.
- stance_alignment: 성향/정체성 유지
- issue_coverage + direct_rebuttal: 논점 포착과 정면 반박
- semantic_distinctiveness + argument_advancement + anti-repetition: 반복 루프 억제와 논의 전개
"""

from __future__ import annotations

from .base import DebateTurnSample
from .judge import Judge
from .utils import SentenceEmbedder, jaccard_similarity, key_point_coverage, tokenize
from .v2_redesign import (
    DiversityReward,
    EngagementReward,
    GroundingReward,
    NoveltyReward,
    PersonaConsistencyReward,
    _load_anchor_texts,
)

_REBUTTAL_CUES = (
    "그러나", "하지만", "반면", "오히려", "그 주장은", "그 논리는", "문제는",
    "사실과 다르", "현실과 다르", "간과", "외면", "무시", "반박", "틀렸",
    "과장", "역효과", "부작용", "왜냐하면", "근거", "실제로", "현실은",
)


class CounterArgumentReward:
    """
    상대 키워드를 언급하는 것만으로 R_engagement를 얻는 reward hacking을 줄인다.
    핵심 논점 커버리지와 한국어 반박 표지(cue)를 함께 요구한다.
    """

    name = "direct_rebuttal"

    def __init__(self, cue_weight: float = 0.35) -> None:
        self.cue_weight = cue_weight

    def __call__(self, sample: DebateTurnSample) -> float:
        if not sample.opponent_response:
            return 0.0
        coverage = key_point_coverage(sample.opponent_response, sample.response)
        cue_score = 1.0 if any(cue in sample.response for cue in _REBUTTAL_CUES) else 0.0
        return (1.0 - self.cue_weight) * coverage + self.cue_weight * cue_score


class ArgumentProgressionReward:
    """
    후반 라운드에서 같은 프레임을 복붙하는 현상을 줄이기 위한 보상.
    자기 과거 발언 대비 새 내용어 비율을 보상하되, 상대 논점을 전혀 다루지 않는 새말하기는
    낮은 점수를 받도록 coverage로 게이트한다.
    """

    name = "argument_advancement"

    def __init__(self, min_round: int = 2, coverage_floor: float = 0.25) -> None:
        self.min_round = min_round
        self.coverage_floor = coverage_floor

    def __call__(self, sample: DebateTurnSample) -> float:
        if sample.round_num < self.min_round or not sample.own_history:
            return 0.0
        response_terms = set(tokenize(sample.response))
        if not response_terms:
            return 0.0
        history_terms = set(tokenize(" ".join(sample.own_history)))
        new_ratio = len(response_terms - history_terms) / len(response_terms)
        coverage = key_point_coverage(sample.opponent_response, sample.response) if sample.opponent_response else 0.0
        gate = self.coverage_floor + (1.0 - self.coverage_floor) * coverage
        return new_ratio * gate


class LateRoundRepetitionPenalty:
    """
    기존 R_novelty는 전 라운드 동일 기준이라 5라운드 이후 복붙 루프를 충분히 누르지 못했다.
    late_start 이후에는 더 낮은 tau와 더 큰 gamma로 Jaccard 반복을 추가 패널티한다.
    """

    name = "late_loop_penalty"

    def __init__(self, late_start: int = 4, tau: float = 0.18, gamma: float = 2.5) -> None:
        self.late_start = late_start
        self.tau = tau
        self.gamma = gamma

    def __call__(self, sample: DebateTurnSample) -> float:
        if sample.round_num < self.late_start or not sample.own_history:
            return 0.0
        j = jaccard_similarity(sample.response, sample.own_history)
        if j < self.tau:
            return 0.0
        return -self.gamma * (j - self.tau)


def build_v3_components(judge: Judge | None, cfg: dict) -> list[tuple]:
    """cfg == config/reward.yaml 의 v3 섹션. v2를 계승하고 v3 전용 보상을 추가한다."""
    weights = cfg.get("weights", {})
    embedder = SentenceEmbedder(model_id=cfg.get("embedding_model", "BAAI/bge-m3"))
    persona_cfg = cfg.get("stance_alignment", cfg.get("persona", {}))
    engagement_cfg = cfg.get("issue_coverage", cfg.get("engagement", {}))
    novelty_cfg = cfg.get("lexical_anti_repetition", cfg.get("novelty", {}))
    direct_cfg = cfg.get("direct_rebuttal", {})
    advance_cfg = cfg.get("argument_advancement", {})
    late_cfg = cfg.get("late_loop_penalty", {})

    def named(component, name: str):
        component.name = name
        return component

    return [
        (
            named(
                PersonaConsistencyReward(
                    embedder=embedder,
                    anchor_texts=_load_anchor_texts(cfg),
                    lambda_syc=persona_cfg.get("lambda_syc", 1.0),
                    syc_coverage_gate=persona_cfg.get("syc_coverage_gate", True),
                    judge=judge,
                    judge_aux_weight=persona_cfg.get("judge_aux_weight", 0.0),
                ),
                "stance_alignment",
            ),
            weights.get("stance_alignment", weights.get("persona", 1.0)),
        ),
        (
            named(
                EngagementReward(judge=judge, judge_weight=engagement_cfg.get("judge_weight", 0.0)),
                "issue_coverage",
            ),
            weights.get("issue_coverage", weights.get("engagement", 1.1)),
        ),
        (
            CounterArgumentReward(cue_weight=direct_cfg.get("cue_weight", 0.35)),
            weights.get("direct_rebuttal", weights.get("counterargument", 0.5)),
        ),
        (
            named(DiversityReward(embedder=embedder), "semantic_distinctiveness"),
            weights.get("semantic_distinctiveness", weights.get("diversity", 1.0)),
        ),
        (
            ArgumentProgressionReward(
                min_round=advance_cfg.get("min_round", 2),
                coverage_floor=advance_cfg.get("coverage_floor", 0.25),
            ),
            weights.get("argument_advancement", weights.get("progression", 0.6)),
        ),
        (
            named(
                NoveltyReward(tau=novelty_cfg.get("tau", 0.25), gamma=novelty_cfg.get("gamma", 1.5)),
                "lexical_anti_repetition",
            ),
            weights.get("lexical_anti_repetition", weights.get("novelty", 0.4)),
        ),
        (
            LateRoundRepetitionPenalty(
                late_start=late_cfg.get("late_start", 4),
                tau=late_cfg.get("tau", 0.18),
                gamma=late_cfg.get("gamma", 2.5),
            ),
            weights.get("late_loop_penalty", weights.get("late_repetition", 0.7)),
        ),
        (named(GroundingReward(), "evidence_grounding"), weights.get("evidence_grounding", weights.get("grounding", 0.0))),
    ]
