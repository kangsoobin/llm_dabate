"""
core/synthesizer.py
────────────────────
중립 Synthesizer 역할의 프롬프트 빌더 (엔진 무관 — vLLM/HF 어디서든 사용).

하이브리드 운용 (2026-07-04 정성 평가 결과에 근거한 설계):
  - 8라운드 평가에서 5라운드 이후 양측이 직전 발언을 사실상 복붙하는 반복 붕괴가 관측됨
    (docs/eval/20260704_sft_vs_grpo_report.md). 같은 질문이 계속 반복 주입되는 구조가 원인의
    절반이므로, Synthesizer가 토론 중간에 개입해 "다뤄지지 않은 쟁점"으로 화두를 전환한다.
  - 토론 종료 시에는 목업(평가 데이터셋.pdf)의 결과 패널 구조로 최종 종합을 작성한다.
  - 판정·승패 평가는 하지 않는다 (Judge Ceiling 대응 — 중간발표 슬라이드 12).

사용처: rl/simulate_vllm.py --synthesizer-every N --synthesizer-final,
        추후 UI(app.py 등)에서도 동일 빌더 재사용 가능.
"""

from __future__ import annotations

import re

SIDE_LABEL = {"left": "진보 측(이진보)", "right": "보수 측(김보수)"}

_NEXT_Q_RE = re.compile(r"다음\s*질문\s*[:：]\s*(.+)")


def format_transcript(turns: list[tuple[str, str]], max_turns: int | None = None) -> str:
    """turns: [(side, text), ...] (시간순). max_turns를 주면 마지막 N턴만 사용."""
    if max_turns is not None:
        turns = turns[-max_turns:]
    blocks = []
    for side, text in turns:
        label = SIDE_LABEL.get(side, side)
        blocks.append(f"[{label}]\n{text.strip()}")
    return "\n\n".join(blocks)


def build_mid_intervention_messages(
    system_prompt: str,
    topic: str,
    turns: list[tuple[str, str]],
    max_turns: int = 6,
) -> list[dict]:
    """
    토론 중간 개입: 지금까지의 쟁점을 정리하고, 아직 다뤄지지 않은 하위 쟁점으로
    사회자가 던질 '다음 질문'을 제안하게 한다. 응답 마지막 줄은 반드시
    "다음 질문: ..." 형식 — parse_next_question()으로 추출한다.
    """
    transcript = format_transcript(turns, max_turns=max_turns)
    user = (
        f"토론 주제: {topic}\n\n"
        f"지금까지의 토론 발언 (최근 순서대로):\n\n{transcript}\n\n"
        "위 토론을 보고 다음 두 가지를 작성하라.\n"
        "1. 지금까지 양측이 맞붙은 핵심 쟁점을 2~3문장으로 정리하라.\n"
        "2. 양측이 계속 같은 논거를 반복하고 있다. 이 주제 안에서 아직 제대로 다뤄지지 않은 "
        "하위 쟁점(예: 구체적 정책 수단, 재원, 시행 시기, 부작용 대책, 국제 비교 등) 하나를 골라, "
        "사회자가 양측에게 던질 새 질문을 한 문장으로 만들어라.\n\n"
        "반드시 마지막 줄을 정확히 다음 형식으로 끝내라.\n"
        "다음 질문: <새 질문 한 문장>"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def build_final_synthesis_messages(
    system_prompt: str,
    topic: str,
    turns: list[tuple[str, str]],
) -> list[dict]:
    """토론 종료 후 최종 종합 — 평가 데이터셋 목업의 결과 패널 구조."""
    transcript = format_transcript(turns)
    user = (
        f"토론 주제: {topic}\n\n"
        f"전체 토론 발언 (시간순):\n\n{transcript}\n\n"
        "위 토론 전체를 아래 구조로 종합하라. 각 항목은 소제목을 그대로 쓰고, "
        "양측 발언에 실제로 등장한 내용만 근거로 삼아라.\n\n"
        "핵심 쟁점 요약: (토론의 대립 축을 2~3문장으로)\n"
        "진보 측 핵심 논거: (2~4개, 간결한 문장으로)\n"
        "보수 측 핵심 논거: (2~4개, 간결한 문장으로)\n"
        "합의 가능 지점: (양측이 공통으로 인정했거나 접점이 될 수 있는 부분. 없으면 '없음'과 이유)\n"
        "남은 쟁점: (해소되지 않았거나 추가 논의가 필요한 질문들)"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user},
    ]


def parse_next_question(synthesizer_output: str, fallback: str) -> str:
    """중간 개입 응답에서 '다음 질문: ...'을 추출. 실패하면 fallback(원래 질문) 반환."""
    matches = _NEXT_Q_RE.findall(synthesizer_output)
    if matches:
        q = matches[-1].strip().strip('"').strip()
        if len(q) >= 5:
            return q
    return fallback
