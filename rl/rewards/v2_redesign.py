"""
rl/rewards/v2_redesign.py
────────────────────────────
docs/reward_design_v2.md 에서 설계한 보상 컴포넌트.

R_persona(§3.1) / R_engagement(§3.2) / R_diversity(§3.3) / R_novelty(§3.4) / R_grounding(§3.5, 기본 비활성)

judge 없이도(judge=None) 전부 동작하도록 설계됨 — Judge Ceiling 의존도를 낮추는 것이
이 재설계의 핵심 동기 중 하나 (docs/reward_design_v2.md §1).

2026-07-04 개선 (GRPO 실전 투입 전 정비):
- anchor: 텍스트 이어붙이기(임베딩 모델 max_seq_length에서 잘림) → 샘플별 임베딩 평균으로
  변경 — 문서 §3.1의 원안("seed 응답 임베딩 평균")과 일치시킴. anchor가 비어 있으면
  조용히 0을 반환하는 대신 생성 시점에 에러를 던진다 (fail-fast).
- sycophancy 페널티에 key-point coverage 게이트 추가: 상대 논점을 실제로 다루면서
  가까워진 것(정당한 반박/부분 수용)은 덜 처벌하고, 논점을 회피하며 가까워진 것(동조)만
  강하게 처벌한다. 문서 §3.1의 "R_engagement가 낮으면서 이 값이 양수인 경우 교차 검증"
  아이디어의 구현이며, 상대 쪽으로의 모든 수렴이 동조는 아니라는 지적(Not All Flips Are
  Conformity, arXiv:2606.00820)에 대한 대응이기도 하다.
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
        anchor_texts: dict[str, list[str]],
        lambda_syc: float = 1.0,
        syc_coverage_gate: bool = True,
        judge: Optional[Judge] = None,
        judge_aux_weight: float = 0.0,
    ) -> None:
        self.embedder = embedder
        # {"left": [...], "right": [...]} — rl/build_anchor.py가 생성한 샘플 텍스트 리스트
        self.anchor_texts = {
            side: ([texts] if isinstance(texts, str) else list(texts or []))
            for side, texts in (anchor_texts or {}).items()
        }
        for side in ("left", "right"):
            if not any(t.strip() for t in self.anchor_texts.get(side, [])):
                raise ValueError(
                    f"R_persona anchor가 비어 있습니다 (side={side}). "
                    "`python rl/build_anchor.py --side left/right`를 먼저 실행해 "
                    "rl/data/anchors/{side}.json을 생성하세요 (config/reward.yaml의 v2.anchor_files 참고). "
                    "anchor 없이 학습하면 페르소나 보상이 조용히 무력화되므로 에러로 막습니다."
                )
        self.lambda_syc = lambda_syc
        self.syc_coverage_gate = syc_coverage_gate
        self.judge = judge
        self.judge_aux_weight = judge_aux_weight
        self._anchor_vecs: dict[str, object] = {}

    def _anchor_vec(self, side: str):
        if side not in self._anchor_vecs:
            self._anchor_vecs[side] = self.embedder.encode_mean(self.anchor_texts[side])
        return self._anchor_vecs[side]

    def __call__(self, sample: DebateTurnSample) -> float:
        drift = max(0.0, min(1.0, self.embedder.cos_sim_vec(sample.response, self._anchor_vec(sample.side))))

        # sycophancy: 직전 자기 발언 대비, 이번 발언이 "현재 상대 발언" 쪽으로 얼마나
        # 더 가까워졌는가. (문서 초안 수식은 opponent_prev와 비교했으나, "상대의 이번
        # 주장에 흔들렸는가"를 재는 데는 현재 상대 발언 기준이 더 직접적이라 이쪽으로 통일 —
        # docs/reward_design_v2.md §3.1도 이 정의로 갱신됨.)
        sycophancy = 0.0
        if sample.opponent_response and sample.own_history:
            prev_own = sample.own_history[-1]
            sim_now = self.embedder.cos_sim(sample.response, sample.opponent_response)
            sim_prev = self.embedder.cos_sim(prev_own, sample.opponent_response)
            sycophancy = sim_now - sim_prev

        syc_penalty = max(0.0, sycophancy)
        if self.syc_coverage_gate and syc_penalty > 0 and sample.opponent_response:
            # 상대 논점을 정면으로 다루며 가까워진 경우(coverage↑)는 동조가 아니라
            # 반박/교전일 가능성이 높으므로 페널티를 (1 - coverage)로 완화한다.
            coverage = key_point_coverage(sample.opponent_response, sample.response)
            syc_penalty *= 1.0 - coverage

        reward = drift - self.lambda_syc * syc_penalty

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


def _load_anchor_texts(cfg: dict, base_dir: str | None = None) -> dict[str, list[str]]:
    """
    anchor 텍스트 로드. 우선순위:
    1. cfg["anchor_files"] — rl/build_anchor.py가 생성한 side별 JSON(list[str]) 파일 경로
    2. cfg["anchor_texts"] — yaml에 직접 넣은 문자열/리스트 (하위 호환)
    """
    import json
    import os

    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    result: dict[str, list[str]] = {}
    anchor_files = cfg.get("anchor_files") or {}
    for side in ("left", "right"):
        path = anchor_files.get(side)
        if path:
            full = path if os.path.isabs(path) else os.path.join(base_dir, path)
            if os.path.exists(full):
                with open(full, encoding="utf-8") as f:
                    result[side] = json.load(f)
                continue
        fallback = (cfg.get("anchor_texts") or {}).get(side, "")
        result[side] = [fallback] if isinstance(fallback, str) else list(fallback)
    return result


def build_v2_components(judge: Optional[Judge], cfg: dict) -> list[tuple]:
    """cfg == config/reward.yaml 의 v2 섹션. 반환값: [(component, weight), ...]"""
    weights = cfg.get("weights", {})
    embedder = SentenceEmbedder(model_id=cfg.get("embedding_model", "BAAI/bge-m3"))
    persona_cfg = cfg.get("persona", {})
    engagement_cfg = cfg.get("engagement", {})
    novelty_cfg = cfg.get("novelty", {})

    return [
        (
            PersonaConsistencyReward(
                embedder=embedder,
                anchor_texts=_load_anchor_texts(cfg),
                lambda_syc=persona_cfg.get("lambda_syc", 1.0),
                syc_coverage_gate=persona_cfg.get("syc_coverage_gate", True),
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
