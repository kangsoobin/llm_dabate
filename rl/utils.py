"""
rl/utils.py
───────────
RL 파이프라인 공유 헬퍼: 설정 로드, 메시지 빌더, 무상태 생성, 문장 트리밍.
"""

from __future__ import annotations

import os
import re
import sys

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

RL_DIR = os.path.join(BASE_DIR, "rl")
DATA_DIR = os.path.join(RL_DIR, "data")


# ────────────────────────────────────────────────────────────
# 설정 로드
# ────────────────────────────────────────────────────────────

def load_rl_config() -> dict:
    with open(os.path.join(RL_DIR, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_config() -> dict:
    with open(os.path.join(BASE_DIR, "config", "model.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompts() -> dict:
    with open(os.path.join(BASE_DIR, "config", "prompts.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_questions() -> list[str]:
    with open(os.path.join(BASE_DIR, "sft", "questions.yaml"), encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    questions = list(data.get("seed_questions", []))
    questions += list(data.get("generated_questions", []))
    return [q for q in questions if q]


def split_questions(questions: list[str], n_heldout: int) -> tuple[list[tuple[int, str]], list[str]]:
    """질문을 학습용/홀드아웃으로 나눈다. 홀드아웃은 토픽 분산을 위해 등간격 추출.

    Returns
    -------
    (train, heldout)
        train은 (원본 인덱스, 질문) 튜플 목록.
    """
    step = max(1, len(questions) // max(1, n_heldout))
    heldout_idx = set(range(step - 1, len(questions), step)[:n_heldout])
    train = [(i, q) for i, q in enumerate(questions) if i not in heldout_idx]
    heldout = [questions[i] for i in sorted(heldout_idx)]
    return train, heldout


# ────────────────────────────────────────────────────────────
# 메시지 빌더 — core/session.py 포맷을 그대로 재사용
# ────────────────────────────────────────────────────────────

def build_message(question: str, opponent_response: str, opponent_name: str, speaker_side: str) -> str:
    """DebateSession._build_message와 동일한 문자열을 생성한다.

    _build_message는 self를 사용하지 않는 순수 함수이므로 unbound 호출로 재사용해
    학습 프롬프트와 배포 시 입력 분포를 일치시킨다.
    """
    from core.session import DebateSession
    return DebateSession._build_message(None, question, opponent_response, opponent_name, speaker_side)


# ────────────────────────────────────────────────────────────
# 무상태 생성 — Agent 히스토리를 건드리지 않고 명시적 메시지로 생성
# ────────────────────────────────────────────────────────────

def stateless_generate(
    agent,
    messages: list[dict],
    max_new_tokens: int = 320,
    temperature: float = 0.8,
    greedy: bool = False,
) -> str:
    """명시적 메시지 목록으로 한 번 생성한다 (agent.history 미사용/미변경)."""
    import torch

    tokenizer, model = agent.tokenizer, agent.model
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    )
    device = f"cuda:{agent.gpu_id}"
    # transformers 5.x는 BatchEncoding, 4.x는 LongTensor를 반환
    if hasattr(encoded, "input_ids"):
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
    else:
        input_ids = encoded.to(device)
        attention_mask = None

    gen_kwargs = {
        "input_ids": input_ids,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if attention_mask is not None:
        gen_kwargs["attention_mask"] = attention_mask
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "top_p": agent.gen_config.get("top_p", 0.9),
            "repetition_penalty": agent.gen_config.get("repetition_penalty", 1.05),
        })

    with torch.no_grad():
        output = model.generate(**gen_kwargs)
    new_tokens = output[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ────────────────────────────────────────────────────────────
# 텍스트 유틸
# ────────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def trim_sentences(text: str, n: int | None) -> str:
    """앞에서부터 n문장만 남긴다. n이 None이면 원문 그대로."""
    if n is None:
        return text.strip()
    sentences = _SENT_SPLIT.split(text.strip())
    return " ".join(sentences[:n]).strip()


def count_prompt_tokens(tokenizer, messages: list[dict]) -> int:
    """채팅 템플릿 적용 후 프롬프트 토큰 수를 반환한다."""
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return len(ids)
