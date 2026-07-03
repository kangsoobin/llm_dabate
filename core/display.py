"""
core/display.py
───────────────
터미널 컬러 출력 및 UI 포맷 유틸리티.

색상 규칙:
  🔵 BLUE   — LEFT agent (이진보, 진보)
  🔴 RED    — RIGHT agent (박보수, 보수)
  🟡 YELLOW — 사회자(사용자) 입력 프롬프트
  🔵 CYAN   — 시스템 안내 메시지
"""

from __future__ import annotations
from typing import Callable

# ── ANSI 컬러 코드 ───────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"

BLUE   = "\033[94m"    # LEFT (진보 — 파란색)
RED    = "\033[91m"    # RIGHT (보수 — 빨간색)
YELLOW = "\033[93m"    # 사회자
CYAN   = "\033[96m"    # 시스템
GREEN  = "\033[92m"    # 성공/완료
MAGENTA = "\033[95m"   # 강조

# ── 폭 상수 ─────────────────────────────────────────────────
WIDTH = 62


def _line(char: str = "─") -> str:
    return char * WIDTH


# ────────────────────────────────────────────────────────────
# 시작 배너
# ────────────────────────────────────────────────────────────

def print_banner(left_name: str = "이진영 교수", right_name: str = "박민준 교수") -> None:
    """프로그램 시작 시 출력되는 타이틀 배너."""
    print()
    print(f"{BOLD}{CYAN}╔{'═' * WIDTH}╗{RESET}")
    title = "🎙️  LLM 정책 토론 시스템  v1.0  🎙️"
    pad = WIDTH - len(title) + 4   # 이모지 바이트 보정
    print(f"{BOLD}{CYAN}║  {title}{' ' * max(0, pad)}║{RESET}")
    sub = f"🔵 {left_name} (진보)   │   🔴 {right_name} (보수)"
    pad2 = WIDTH - len(sub) + 4
    print(f"{BOLD}{CYAN}║  {sub}{' ' * max(0, pad2)}║{RESET}")
    print(f"{BOLD}{CYAN}╚{'═' * WIDTH}╝{RESET}")
    print()


# ────────────────────────────────────────────────────────────
# 라운드 헤더
# ────────────────────────────────────────────────────────────

def print_round_header(round_num: int, question: str) -> None:
    """새 라운드 시작 시 구분선과 주제를 출력."""
    print()
    label = f" ROUND {round_num} "
    side_len = (WIDTH - len(label)) // 2
    print(f"{BOLD}{MAGENTA}{'═' * side_len}{label}{'═' * side_len}{RESET}")
    # 질문이 길면 줄바꿈
    q_display = question if len(question) <= WIDTH - 4 else question[:WIDTH - 7] + "..."
    print(f"{BOLD}📌 {q_display}{RESET}")
    print()


# ────────────────────────────────────────────────────────────
# Agent 발언 헤더 / 푸터
# ────────────────────────────────────────────────────────────

def print_agent_header(name: str, side: str) -> None:
    """Agent가 발언을 시작할 때 출력되는 헤더."""
    color = BLUE if side == "left" else RED
    icon  = "🔵" if side == "left" else "🔴"
    label = "진보" if side == "left" else "보수"
    print(f"\n{BOLD}{color}{icon} {name} ({label}) ▼{RESET}")
    print(f"{color}{_line('━')}{RESET}")


def print_agent_footer(side: str) -> None:
    """Agent 발언이 끝난 직후 개행 추가."""
    color = BLUE if side == "left" else RED
    print(f"\n{color}{_line('─')}{RESET}")


# ────────────────────────────────────────────────────────────
# 기타 구분선 / 시스템 메시지
# ────────────────────────────────────────────────────────────

def print_divider() -> None:
    """일반 구분선."""
    print(f"\n{DIM}{_line()}{RESET}")


def print_system(msg: str) -> None:
    """시스템 안내 메시지 (모델 로딩 상태 등)."""
    print(f"  {CYAN}→{RESET}  {msg}")


def print_command_prompt() -> None:
    """매 라운드 종료 후 커맨드 안내를 출력."""
    print()
    print(f"{DIM}{_line()}{RESET}")
    print(
        f"  {BOLD}명령어:{RESET}  "
        f"{YELLOW}[질문 입력]{RESET}→양쪽 응답   "
        f"{BLUE}[l 질문]{RESET}→LEFT만   "
        f"{RED}[r 질문]{RESET}→RIGHT만   "
        f"{DIM}[q]→종료·저장{RESET}"
    )
    print(f"{DIM}{_line()}{RESET}")


# ────────────────────────────────────────────────────────────
# 스트리밍 콜백 팩토리
# ────────────────────────────────────────────────────────────

def make_stream_callback(side: str) -> Callable[[str], None]:
    """
    토큰이 생성될 때마다 해당 Agent 색상으로 즉시 출력하는 콜백을 반환한다.

    Parameters
    ----------
    side : str
        "left" → BLUE,  "right" → RED

    Returns
    -------
    Callable[[str], None]
        BaseAgent.generate()에 stream_callback으로 전달할 함수
    """
    color = BLUE if side == "left" else RED

    def _callback(token: str) -> None:
        print(f"{color}{token}{RESET}", end="", flush=True)

    return _callback
