"""
rl/rewards/base.py
───────────────────
보상 컴포넌트 공통 인터페이스.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class DebateTurnSample:
    """GRPO 롤아웃에서 보상 계산에 필요한 한 턴의 컨텍스트."""

    response: str                                    # 이번에 생성된(평가 대상) 발언
    side: str                                         # "left" | "right"
    question: str                                     # 사회자 질문(토론 주제)
    opponent_response: str                            # 직전 상대 발언 (없으면 "")
    own_history: list[str] = field(default_factory=list)  # 같은 side의 과거 발언들 (직전 것 제외)
    round_num: int = 1
    evidence: Optional[str] = None                    # RAG로 검색된 근거 문서 (없으면 None) — R_grounding용


class RewardComponent(Protocol):
    """모든 보상 컴포넌트가 구현해야 하는 시그니처."""

    name: str

    def __call__(self, sample: DebateTurnSample) -> float: ...
