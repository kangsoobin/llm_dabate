"""
rl/rewards/composer.py
─────────────────────────
config/reward.yaml의 version(v1|v2)에 따라 컴포넌트를 조립하고,
TRL GRPOTrainer가 요구하는 reward_func(prompts, completions, **kwargs) -> list[float]
시그니처로 감싼다.
"""

from __future__ import annotations

from .base import DebateTurnSample
from .judge import build_judge
from .v1_pdf import build_v1_components
from .v2_redesign import build_v2_components


class RewardComposer:
    """여러 RewardComponent를 가중합해 R_total을 계산한다 (docs/reward_design_v2.md §4)."""

    def __init__(self, components: list[tuple]) -> None:
        # components: [(component_callable, weight), ...]
        self.components = components

    def score(self, sample: DebateTurnSample) -> float:
        return sum(weight * component(sample) for component, weight in self.components)

    def breakdown(self, sample: DebateTurnSample) -> dict[str, float]:
        """컴포넌트별 raw 값 (가중치 곱하기 전) — 로깅/디버깅용."""
        return {component.name: component(sample) for component, _ in self.components}

    @classmethod
    def from_config(cls, cfg: dict) -> "RewardComposer":
        version = cfg.get("version", "v2")
        judge = build_judge(cfg.get("judge", {}))
        section = cfg.get(version, {})
        if version == "v1":
            components = build_v1_components(judge, section)
        elif version == "v2":
            components = build_v2_components(judge, section)
        else:
            raise ValueError(f"알 수 없는 reward version: {version!r} (v1|v2 중 하나여야 함)")
        return cls(components)

    @staticmethod
    def _completion_text(completion) -> str:
        """
        TRL 대화형(conversational) 포맷에서 completion은 [{"role": "assistant", "content": "..."}]
        형태의 메시지 리스트로 전달된다. 표준(standard) 포맷이면 이미 문자열이다.
        """
        if isinstance(completion, list):
            return completion[0]["content"]
        return completion

    def as_trl_reward_fn(self):
        """
        TRL GRPOTrainer의 reward_funcs에 넘길 콜러블을 만든다.
        rl/rollout.py가 만드는 데이터셋에는 side/question/opponent_response/own_history/
        round_num/evidence 컬럼이 있어야 한다 (GRPOTrainer가 자동으로 kwargs에 실어 전달).
        """

        def reward_fn(prompts, completions, **kwargs) -> list[float]:
            sides = kwargs["side"]
            questions = kwargs["question"]
            opponents = kwargs["opponent_response"]
            histories = kwargs["own_history"]
            rounds = kwargs.get("round_num", [1] * len(completions))
            evidences = kwargs.get("evidence", [None] * len(completions))

            scores = []
            for i, completion in enumerate(completions):
                sample = DebateTurnSample(
                    response=self._completion_text(completion),
                    side=sides[i],
                    question=questions[i],
                    opponent_response=opponents[i],
                    own_history=list(histories[i]) if histories[i] else [],
                    round_num=rounds[i],
                    evidence=evidences[i],
                )
                scores.append(self.score(sample))
            return scores

        return reward_fn
