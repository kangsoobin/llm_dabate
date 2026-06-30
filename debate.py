#!/usr/bin/env python3
"""
debate.py — LLM 정책 토론 시스템 메인 진입점
══════════════════════════════════════════════

사용법:
    conda activate debate
    python debate.py

커맨드:
    [질문 입력]       → LEFT / RIGHT 양쪽 순서대로 응답
    l [질문]          → LEFT(이진영 교수)에게만 질문
    r [질문]          → RIGHT(박민준 교수)에게만 질문
    q                 → 토론 종료 & JSON 로그 저장
    vram              → 현재 GPU VRAM 사용량 출력
    reset             → 양쪽 대화 히스토리 초기화
"""

import os
import sys
import time

import yaml

# ── 프로젝트 루트를 sys.path에 추가 ─────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agents import create_left_agent, create_right_agent
from core import (
    DebateSession,
    print_banner,
    print_command_prompt,
    print_system,
    YELLOW, CYAN, BOLD, RESET, GREEN, RED,
)


# ────────────────────────────────────────────────────────────
# 설정 로드
# ────────────────────────────────────────────────────────────

def load_configs() -> tuple[dict, dict, str, str, str | None, str | None]:
    """
    config/model.yaml 과 config/prompts.yaml을 로드한다.

    Returns
    -------
    (model_cfg, gen_cfg, left_prompt, right_prompt, left_adapter, right_adapter)
    """
    cfg_dir = os.path.join(BASE_DIR, "config")

    with open(os.path.join(cfg_dir, "model.yaml"), encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)

    with open(os.path.join(cfg_dir, "prompts.yaml"), encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    # 모델 관련 설정과 생성 파라미터 분리
    model_cfg = {
        "model_id":    full_cfg["model_id"],
        "left_gpu":    full_cfg["left_gpu"],
        "right_gpu":   full_cfg["right_gpu"],
        "quantization": full_cfg["quantization"],
    }
    left_adapter  = full_cfg.get("left_adapter")  or None
    right_adapter = full_cfg.get("right_adapter") or None
    gen_cfg = {
        "max_new_tokens":    full_cfg["max_new_tokens"],
        "temperature":       full_cfg["temperature"],
        "top_p":             full_cfg["top_p"],
        "repetition_penalty": full_cfg["repetition_penalty"],
    }

    return model_cfg, gen_cfg, prompts["left"], prompts["right"], left_adapter, right_adapter


# ────────────────────────────────────────────────────────────
# 모델 로딩 (진행 상황 출력 포함)
# ────────────────────────────────────────────────────────────

def load_models(
    model_cfg: dict,
    gen_cfg: dict,
    left_prompt: str,
    right_prompt: str,
    left_adapter: str | None = None,
    right_adapter: str | None = None,
):
    """두 Agent를 순서대로 로드하고 DebateSession을 반환한다."""

    left_agent  = create_left_agent(model_cfg, gen_cfg, left_prompt,  adapter_path=left_adapter)
    right_agent = create_right_agent(model_cfg, gen_cfg, right_prompt, adapter_path=right_adapter)

    print()
    print_system(f"모델: {model_cfg['model_id']}")
    print_system(f"양자화: {model_cfg['quantization']}")
    print()

    # LEFT 로딩
    print_system(
        f"GPU {model_cfg['left_gpu']}에 "
        f"{left_agent.name} (LEFT) 로딩 중... ⏳"
    )
    t0 = time.time()
    left_agent.load()
    elapsed = time.time() - t0
    print_system(
        f"{left_agent.name} 로드 완료 ✔  "
        f"({elapsed:.0f}초 / VRAM {left_agent.vram_usage_gb():.1f} GB)"
    )

    print()

    # RIGHT 로딩
    print_system(
        f"GPU {model_cfg['right_gpu']}에 "
        f"{right_agent.name} (RIGHT) 로딩 중... ⏳"
    )
    t0 = time.time()
    right_agent.load()
    elapsed = time.time() - t0
    print_system(
        f"{right_agent.name} 로드 완료 ✔  "
        f"({elapsed:.0f}초 / VRAM {right_agent.vram_usage_gb():.1f} GB)"
    )

    print()
    print(f"  {GREEN}{BOLD}두 모델 로드 완료. 토론 준비 완료! 🎙️{RESET}")
    print()

    return DebateSession(left_agent, right_agent)


# ────────────────────────────────────────────────────────────
# 메인 토론 루프
# ────────────────────────────────────────────────────────────

def main() -> None:
    # ── 설정 로드 ────────────────────────────────────────────
    try:
        model_cfg, gen_cfg, left_prompt, right_prompt, left_adapter, right_adapter = load_configs()
    except FileNotFoundError as e:
        print(f"{RED}설정 파일을 찾을 수 없습니다: {e}{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{RED}설정 로드 오류: {e}{RESET}")
        sys.exit(1)

    # ── 배너 출력 ────────────────────────────────────────────
    print_banner()

    print_system("서버 환경 요약:")
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_gb = torch.cuda.mem_get_info(i)[0] / (1024**3)
                print_system(
                    f"  GPU {i}: {props.name}  "
                    f"여유 {free_gb:.1f}/{props.total_memory/(1024**3):.1f} GB"
                )
        else:
            print(f"{RED}  CUDA를 사용할 수 없습니다. 'conda activate debate' 후 재실행하세요.{RESET}")
            sys.exit(1)
    except ImportError:
        print(f"{RED}  PyTorch가 설치되지 않았습니다. install_env.sh를 먼저 실행하세요.{RESET}")
        sys.exit(1)

    print()

    # ── 모델 로딩 ────────────────────────────────────────────
    session = load_models(model_cfg, gen_cfg, left_prompt, right_prompt, left_adapter, right_adapter)

    # ── 첫 번째 주제 입력 ────────────────────────────────────
    print(f"  {YELLOW}{BOLD}💬 토론 주제를 입력하세요{RESET}")
    print(f"  {YELLOW}예: 최저임금 인상이 필요한가? / 원자력 발전 확대해야 하나?{RESET}")
    print()

    while True:
        try:
            topic = input(f"  {BOLD}{YELLOW}❯ 주제: {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{CYAN}토론을 취소합니다.{RESET}")
            return

        if topic:
            break
        print(f"  {RED}주제를 입력해 주세요.{RESET}")

    NUM_ROUNDS = 4
    CLOSING_PROMPT = "토론을 마무리하겠습니다. 지금까지의 논의를 바탕으로 자신의 핵심 주장을 간결하게 정리하고 최종 발언을 해주세요."

    while True:
        session.topic = topic
        session.left.reset_history()
        session.right.reset_history()
        session.round_num = 0
        session._log = []

        # ── 4라운드 자동 진행 ─────────────────────────────────
        for rnd in range(1, NUM_ROUNDS + 1):
            print_system(f"{'━' * 50}")
            print_system(f"  라운드 {rnd} / {NUM_ROUNDS}")
            print_system(f"{'━' * 50}")
            session.run_full_round(topic)

        # ── 최종 발언 ─────────────────────────────────────────
        print_system(f"{'━' * 50}")
        print_system("  최종 발언")
        print_system(f"{'━' * 50}")

        from core.display import make_stream_callback, print_agent_header, print_agent_footer

        left_close_msg = session._build_message(
            CLOSING_PROMPT, session.right.last_response, session.right.name, "left", is_closing=True
        )
        print_agent_header(session.left.name, "left")
        session.left.generate(left_close_msg, stream_callback=make_stream_callback("left"))
        print_agent_footer("left")

        right_close_msg = session._build_message(
            CLOSING_PROMPT, session.left.last_response, session.left.name, "right", is_closing=True
        )
        print_agent_header(session.right.name, "right")
        session.right.generate(right_close_msg, stream_callback=make_stream_callback("right"))
        print_agent_footer("right")

        # ── 로그 저장 ─────────────────────────────────────────
        log_dir = os.path.join(BASE_DIR, "logs")
        path = session.save_log(log_dir)
        print()
        print(f"  {GREEN}{BOLD}토론 종료. 기록 저장됨: {path}{RESET}")
        print()

        # ── 새 주제 or 종료 ───────────────────────────────────
        try:
            again = input(f"  {BOLD}{YELLOW}❯ 새로운 주제를 입력하세요 (종료: q): {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            again = "q"

        if again.lower() == "q" or not again:
            break
        topic = again

    print(f"  {CYAN}수고하셨습니다. 👋{RESET}\n")


# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
