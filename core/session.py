"""
core/session.py
───────────────
토론 세션 관리.

핵심 역할:
  - 라운드 진행 (양쪽 / 한쪽 지정)
  - 상대 발언을 다음 Agent의 user 메시지에 자동 주입
  - 전체 토론 기록을 JSON으로 저장
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from core.display import (
    print_round_header,
    print_agent_header,
    print_agent_footer,
    print_system,
    make_stream_callback,
)

if TYPE_CHECKING:
    from agents.base_agent import BaseAgent


def build_debate_message(
    question: str,
    opponent_response: str,
    opponent_name: str,
    speaker_side: str,          # "left" or "right"
    is_closing: bool = False,
) -> str:
    """
    사회자 질문에 상대방의 직전 발언을 주입한 메시지를 생성한다.
    상대 발언이 없으면(첫 발언 등) 질문만 반환한다.

    모듈 함수로 분리한 이유: rl/simulate_vllm.py처럼 BaseAgent(HF 모델 로딩)를 쓰지 않는
    파이프라인에서도 동일한 프롬프트 포맷을 재사용하기 위함 — 여기 포맷이 바뀌면
    토론 UI·self-play 데이터 생성이 전부 같이 바뀌어야 하므로 반드시 이 함수 한 곳만 수정할 것.
    """
    if not opponent_response:
        return question

    opponent_label = "우파" if speaker_side == "left" else "좌파"
    own_label      = "진보" if speaker_side == "left" else "보수"

    if is_closing:
        instruction = (
            f"지금까지의 토론을 마무리하며, {own_label}적 관점에서 "
            f"핵심 주장을 간결하게 정리하고 최종 발언을 하라."
        )
    else:
        instruction = (
            f"위 {opponent_label}의 주장을 정면으로 반박하고, "
            f"{own_label}적 관점에서 사회자의 질문에 답하라."
        )

    return (
        f"사회자 질문: {question}\n\n"
        f"{'━' * 40}\n"
        f"방금 {opponent_label}({opponent_name})이(가) 한 발언:\n"
        f"{opponent_response}\n"
        f"{'━' * 40}\n\n"
        f"{instruction}"
    )


class DebateSession:
    """
    두 Agent를 조율해 토론 세션을 진행한다.

    Parameters
    ----------
    left_agent : BaseAgent
        좌파(이진영 교수) Agent
    right_agent : BaseAgent
        우파(박민준 교수) Agent
    """

    def __init__(self, left_agent: BaseAgent, right_agent: BaseAgent) -> None:
        self.left  = left_agent
        self.right = right_agent

        self.round_num: int = 0
        self.topic: str = ""
        self.started_at: str = datetime.now().isoformat(timespec="seconds")

        # 로그용 전체 기록
        self._log: list[dict] = []

    # ─────────────────────────────────────────────────────────
    # 내부: 상대 발언 주입 메시지 빌더
    # ─────────────────────────────────────────────────────────

    def _build_message(
        self,
        question: str,
        opponent_response: str,
        opponent_name: str,
        speaker_side: str,          # "left" or "right"
        is_closing: bool = False,
    ) -> str:
        """모듈 함수 build_debate_message로 위임 (포맷 정의는 그쪽 한 곳에만 둔다)."""
        return build_debate_message(question, opponent_response, opponent_name, speaker_side, is_closing)

    # ─────────────────────────────────────────────────────────
    # 공개 API
    # ─────────────────────────────────────────────────────────

    def run_full_round(self, question: str) -> tuple[str, str]:
        """
        한 라운드를 진행한다: LEFT 먼저 → RIGHT (LEFT 발언 주입).

        Returns
        -------
        (left_response, right_response)
        """
        self.round_num += 1
        print_round_header(self.round_num, question)

        # ── LEFT 응답 ────────────────────────────────────────
        # LEFT는 이전 RIGHT 발언을 주입 (없으면 질문만)
        left_msg = self._build_message(
            question,
            self.right.last_response,
            self.right.name,
            speaker_side="left",
        )
        print_agent_header(self.left.name, "left")
        left_cb = make_stream_callback("left")
        left_resp = self.left.generate(left_msg, stream_callback=left_cb)
        print_agent_footer("left")

        # ── RIGHT 응답 ───────────────────────────────────────
        # RIGHT는 방금 LEFT가 한 발언을 주입
        right_msg = self._build_message(
            question,
            left_resp,
            self.left.name,
            speaker_side="right",
        )
        print_agent_header(self.right.name, "right")
        right_cb = make_stream_callback("right")
        right_resp = self.right.generate(right_msg, stream_callback=right_cb)
        print_agent_footer("right")

        # ── 로그 기록 ────────────────────────────────────────
        self._log.append({
            "round": self.round_num,
            "type": "full",
            "moderator": question,
            "left":  left_resp,
            "right": right_resp,
        })

        return left_resp, right_resp

    def run_directed(self, side: str, question: str) -> str:
        """
        한 쪽 Agent에게만 발언 기회를 준다.
        상대방의 직전 발언은 여전히 컨텍스트로 주입된다.

        Parameters
        ----------
        side : str
            "left" 또는 "right"
        question : str
            사회자 질문

        Returns
        -------
        str
            발언한 Agent의 응답
        """
        self.round_num += 1
        print_round_header(self.round_num, f"[{side.upper()} 전용] {question}")

        if side == "left":
            msg = self._build_message(
                question,
                self.right.last_response,
                self.right.name,
                speaker_side="left",
            )
            print_agent_header(self.left.name, "left")
            cb = make_stream_callback("left")
            response = self.left.generate(msg, stream_callback=cb)
            print_agent_footer("left")
            self._log.append({
                "round": self.round_num,
                "type": "directed_left",
                "moderator": question,
                "left": response,
                "right": None,
            })

        else:  # "right"
            msg = self._build_message(
                question,
                self.left.last_response,
                self.left.name,
                speaker_side="right",
            )
            print_agent_header(self.right.name, "right")
            cb = make_stream_callback("right")
            response = self.right.generate(msg, stream_callback=cb)
            print_agent_footer("right")
            self._log.append({
                "round": self.round_num,
                "type": "directed_right",
                "moderator": question,
                "left": None,
                "right": response,
            })

        return response

    def save_log(self, log_dir: str = "logs") -> str:
        """
        전체 토론 기록을 JSON 파일로 저장한다.

        Parameters
        ----------
        log_dir : str
            저장 디렉토리 경로 (없으면 자동 생성)

        Returns
        -------
        str
            저장된 파일의 절대 경로
        """
        os.makedirs(log_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename  = f"debate_{timestamp}.json"
        filepath  = os.path.join(log_dir, filename)

        data = {
            "started_at":  self.started_at,
            "ended_at":    datetime.now().isoformat(timespec="seconds"),
            "topic":       self.topic,
            "model":       self.left.model_id,
            "quantization": self.left.quantization,
            "left_agent":  self.left.name,
            "right_agent": self.right.name,
            "total_rounds": self.round_num,
            "rounds":      self._log,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return os.path.abspath(filepath)

    def print_vram_status(self) -> None:
        """현재 양쪽 GPU VRAM 사용량을 출력한다."""
        print_system(
            f"GPU {self.left.gpu_id} (LEFT)  VRAM 사용: "
            f"{self.left.vram_usage_gb():.1f} GB"
        )
        print_system(
            f"GPU {self.right.gpu_id} (RIGHT) VRAM 사용: "
            f"{self.right.vram_usage_gb():.1f} GB"
        )
