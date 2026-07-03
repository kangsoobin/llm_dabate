#!/usr/bin/env python3
"""
app.py — LLM Debate Arena  (Streamlit UI)
══════════════════════════════════════════

실행:
    conda activate debate
    streamlit run app.py
"""

import os
import sys

import streamlit as st
import yaml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from agents import create_left_agent, create_right_agent
from core.session import DebateSession

# ────────────────────────────────────────────────────────────
# CSS
# ────────────────────────────────────────────────────────────

CSS = """
<style>
/* ── 기본 배경 ── */
.stApp { background-color: #f5f7fa; }
#MainMenu, footer, header { visibility: hidden; }

/* ── 채팅 메시지 공통 ── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 10px 14px !important;
    margin: 4px 0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}

/* ── LEFT (진보) — 파란색 ── */
[data-testid="stChatMessage"]:has(.msg-left) {
    background-color: #e8f0fe !important;
    border-left: 4px solid #4285f4 !important;
}

/* ── RIGHT (보수) — 빨간색 ── */
[data-testid="stChatMessage"]:has(.msg-right) {
    background-color: #fce8e8 !important;
    border-left: 4px solid #ea4335 !important;
}

/* ── 사회자 — 회색 ── */
[data-testid="stChatMessage"]:has(.msg-moderator) {
    background-color: #f1f3f4 !important;
    border-left: 4px solid #9e9e9e !important;
}

/* ── 입력창 ── */
[data-testid="stChatInput"] textarea {
    border-radius: 24px !important;
}

/* ── 헤더 divider ── */
hr { margin: 8px 0 16px 0 !important; }
</style>
"""

# ────────────────────────────────────────────────────────────
# 모델 로딩 (앱 수명 동안 1회만)
# ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_debate_session() -> DebateSession:
    """두 Agent를 GPU에 로드하고 DebateSession을 반환 (캐시)."""
    cfg_dir = os.path.join(BASE_DIR, "config")

    with open(os.path.join(cfg_dir, "model.yaml"), encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    with open(os.path.join(cfg_dir, "prompts.yaml"), encoding="utf-8") as f:
        prompts = yaml.safe_load(f)

    model_cfg = {
        "model_id":     full_cfg["model_id"],
        "left_gpu":     full_cfg["left_gpu"],
        "right_gpu":    full_cfg["right_gpu"],
        "quantization": full_cfg["quantization"],
    }
    left_adapter  = full_cfg.get("left_adapter")  or None
    right_adapter = full_cfg.get("right_adapter") or None
    gen_cfg = {
        "max_new_tokens":     full_cfg["max_new_tokens"],
        "temperature":        full_cfg["temperature"],
        "top_p":              full_cfg["top_p"],
        "repetition_penalty": full_cfg["repetition_penalty"],
    }

    left_agent  = create_left_agent(model_cfg, gen_cfg, prompts["left"],  adapter_path=left_adapter)
    right_agent = create_right_agent(model_cfg, gen_cfg, prompts["right"], adapter_path=right_adapter)
    left_agent.load()
    right_agent.load()

    return DebateSession(left_agent, right_agent)


# ────────────────────────────────────────────────────────────
# 메시지 렌더링 헬퍼
# ────────────────────────────────────────────────────────────

NUM_ROUNDS = 4
CLOSING_PROMPT = "토론을 마무리하겠습니다. 지금까지의 논의를 바탕으로 자신의 핵심 주장을 간결하게 정리하고 최종 발언을 해주세요."

# CSS 색상 트리거용 숨김 마커
_MARKER = {
    "moderator": '<span class="msg-moderator" style="display:none"></span>',
    "left":      '<span class="msg-left"      style="display:none"></span>',
    "right":     '<span class="msg-right"     style="display:none"></span>',
}


def render_message(msg: dict, left_name: str, right_name: str) -> None:
    """session_state 에 저장된 메시지 dict 하나를 chat_message 로 렌더링."""
    role    = msg["role"]
    content = msg["content"]

    if role == "round_header":
        st.markdown(
            f"<div style='text-align:center;color:#888;font-size:13px;"
            f"margin:12px 0 4px 0;letter-spacing:1px;'>── {content} ──</div>",
            unsafe_allow_html=True,
        )

    elif role == "moderator":
        with st.chat_message("user", avatar="🎙️"):
            st.markdown(_MARKER["moderator"], unsafe_allow_html=True)
            st.markdown("**나 (사회자)**")
            st.write(content)

    elif role == "left":
        with st.chat_message("assistant", avatar="🔵"):
            st.markdown(_MARKER["left"], unsafe_allow_html=True)
            st.markdown(f"**🔵 {left_name} (진보)**")
            st.write(content)

    elif role == "right":
        with st.chat_message("assistant", avatar="🔴"):
            st.markdown(_MARKER["right"], unsafe_allow_html=True)
            st.markdown(f"**🔴 {right_name} (보수)**")
            st.write(content)


# ────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Debate Arena",
        page_icon="🎙️",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    st.markdown(CSS, unsafe_allow_html=True)

    # ── 세션 상태 초기화 ────────────────────────────────────────
    if "messages" not in st.session_state:
        st.session_state.messages = []          # [{role, content}, ...]
    if "do_reset"  not in st.session_state:
        st.session_state.do_reset = False
    if "models_ready" not in st.session_state:
        st.session_state.models_ready = False

    # ── 헤더 ────────────────────────────────────────────────────
    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.markdown("## 🎙️ Debate Arena")
        st.caption("🔵 이진보 (더불어민주당) vs 김보수 (국민의힘) 🔴 — 사회자가 주제를 입력하면 두 Agent가 토론합니다")
    with col_btn:
        st.write("")   # 수직 정렬용 여백
        if st.button("🔄 새로운 주제", use_container_width=True):
            st.session_state.messages = []
            st.session_state.do_reset = True
            st.rerun()

    st.divider()

    # ── 모델 로딩 ────────────────────────────────────────────────
    if not st.session_state.models_ready:
        load_msg = st.info(
            "🔧 모델을 GPU에 로딩 중입니다... "
            "처음 실행 시 수 분이 소요됩니다. 잠시만 기다려 주세요.",
            icon="⏳",
        )
    session = load_debate_session()
    st.session_state.models_ready = True
    if not st.session_state.messages:          # 로딩 메시지 지우기
        try:
            load_msg.empty()                   # type: ignore[union-attr]
        except Exception:
            pass

    # ── 새 주제 리셋 처리 ───────────────────────────────────────
    if st.session_state.do_reset:
        session.left.reset_history()
        session.right.reset_history()
        session.round_num = 0
        session._log      = []
        session.topic     = ""
        st.session_state.do_reset = False

    # ── 기존 메시지 렌더링 ──────────────────────────────────────
    for msg in st.session_state.messages:
        render_message(msg, session.left.name, session.right.name)

    # ── 빈 화면 안내 ────────────────────────────────────────────
    if not st.session_state.messages:
        st.markdown(
            """
            <div style="text-align:center; color:#999; padding:60px 0 40px 0;">
                <div style="font-size:52px;">🎙️</div>
                <div style="font-size:18px; font-weight:500; margin-top:14px; color:#555;">
                    아래에 토론 주제를 입력하면 시작합니다
                </div>
                <div style="font-size:14px; margin-top:8px;">
                    예: 기본소득을 도입해야 할까요?&nbsp;&nbsp;/&nbsp;&nbsp;원자력 발전을 확대해야 하나?
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── 채팅 입력 (토론 주제) ─────────────────────────────────────
    disabled = bool(st.session_state.messages)  # 토론 중엔 입력 비활성화
    if prompt := st.chat_input("토론 주제를 입력하세요...", disabled=disabled):

        session.topic = prompt
        session.left.reset_history()
        session.right.reset_history()
        session.round_num = 0
        session._log = []

        # 사회자 메시지
        st.session_state.messages.append({"role": "moderator", "content": prompt})
        with st.chat_message("user", avatar="🎙️"):
            st.markdown(_MARKER["moderator"], unsafe_allow_html=True)
            st.markdown("**나 (사회자)**")
            st.write(prompt)

        # ── 4라운드 자동 진행 ─────────────────────────────────
        for rnd in range(1, NUM_ROUNDS + 1):
            header = f"라운드 {rnd}"
            st.session_state.messages.append({"role": "round_header", "content": header})
            st.markdown(
                f"<div style='text-align:center;color:#888;font-size:13px;"
                f"margin:12px 0 4px 0;letter-spacing:1px;'>── {header} ──</div>",
                unsafe_allow_html=True,
            )

            left_msg = session._build_message(
                prompt, session.right.last_response, session.right.name, "left"
            )
            with st.chat_message("assistant", avatar="🔵"):
                st.markdown(_MARKER["left"], unsafe_allow_html=True)
                st.markdown(f"**🔵 {session.left.name} (진보)**")
                left_resp = st.write_stream(session.left.generate_iter(left_msg))

            right_msg = session._build_message(
                prompt, session.left.last_response, session.left.name, "right"
            )
            with st.chat_message("assistant", avatar="🔴"):
                st.markdown(_MARKER["right"], unsafe_allow_html=True)
                st.markdown(f"**🔴 {session.right.name} (보수)**")
                right_resp = st.write_stream(session.right.generate_iter(right_msg))

            session.round_num += 1
            session._log.append({
                "round": session.round_num, "type": "full",
                "moderator": prompt, "left": left_resp, "right": right_resp,
            })
            st.session_state.messages.append({"role": "left",  "content": left_resp})
            st.session_state.messages.append({"role": "right", "content": right_resp})

        # ── 최종 발언 ─────────────────────────────────────────
        st.session_state.messages.append({"role": "round_header", "content": "최종 발언"})
        st.markdown(
            "<div style='text-align:center;color:#888;font-size:13px;"
            "margin:12px 0 4px 0;letter-spacing:1px;'>── 최종 발언 ──</div>",
            unsafe_allow_html=True,
        )

        left_close_msg = session._build_message(
            CLOSING_PROMPT, session.right.last_response, session.right.name, "left", is_closing=True
        )
        with st.chat_message("assistant", avatar="🔵"):
            st.markdown(_MARKER["left"], unsafe_allow_html=True)
            st.markdown(f"**🔵 {session.left.name} (진보)**")
            left_close = st.write_stream(session.left.generate_iter(left_close_msg))

        right_close_msg = session._build_message(
            CLOSING_PROMPT, session.left.last_response, session.left.name, "right", is_closing=True
        )
        with st.chat_message("assistant", avatar="🔴"):
            st.markdown(_MARKER["right"], unsafe_allow_html=True)
            st.markdown(f"**🔴 {session.right.name} (보수)**")
            right_close = st.write_stream(session.right.generate_iter(right_close_msg))

        session.round_num += 1
        session._log.append({
            "round": session.round_num, "type": "closing",
            "moderator": CLOSING_PROMPT, "left": left_close, "right": right_close,
        })
        st.session_state.messages.append({"role": "left",  "content": left_close})
        st.session_state.messages.append({"role": "right", "content": right_close})
        st.session_state.messages.append({
            "role": "round_header", "content": "토론 종료 — 🔄 새로운 주제 버튼으로 다시 시작하세요"
        })


if __name__ == "__main__":
    main()
