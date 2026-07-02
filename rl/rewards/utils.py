"""
rl/rewards/utils.py
────────────────────
Judge 없이(judge-free) 계산 가능한 보상 컴포넌트들이 공유하는 유틸리티.
- 단어 단위 Jaccard 유사도 (R_novelty, v1 R_antirep)
- 핵심 논점(key point) 추출 및 커버리지 (R_engagement, ArgKP/KPA 경량 근사)
- 문장 임베딩 (R_persona, R_diversity)
"""

from __future__ import annotations

import re
from typing import Optional

# 형태소 분석기 없이 쓰는 최소한의 한국어 불용어(조사/서술어성 어미) 근사 목록.
_STOPWORDS = {
    "그리고", "그러나", "하지만", "따라서", "때문에", "그런데", "그래서",
    "있다", "없다", "합니다", "입니다", "것이다", "것입니다", "합니다만",
}

_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text) if t not in _STOPWORDS and len(t) > 1]


def jaccard_similarity(text: str, history: list[str]) -> float:
    """text와 history(과거 발언들 전체를 합친 것) 간 단어 단위 Jaccard 유사도."""
    if not history:
        return 0.0
    a = set(tokenize(text))
    b = set(tokenize(" ".join(history)))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_key_points(text: str, top_n: int = 8) -> set[str]:
    """
    상대 발언에서 핵심 논점 후보(명사구 근사)를 추출한다.

    형태소 분석기 없이 2글자 이상 토큰의 빈도로 근사한 ArgKP/KPA(ACL 2021 argmining-1.16)
    스타일의 경량 대체재. 실서비스에서는 KoNLPy/Kiwi 등으로 명사구 추출기를 교체해
    이 함수만 갈아끼우면 된다.
    """
    freq: dict[str, int] = {}
    for tok in tokenize(text):
        freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return {tok for tok, _ in ranked[:top_n]}


def key_point_coverage(opponent_text: str, response_text: str) -> float:
    """opponent_text의 핵심 논점 중 response_text가 실제로 언급한 비율. (docs/reward_design_v2.md §3.2)"""
    kp = extract_key_points(opponent_text)
    if not kp:
        return 0.0
    response_tokens = set(tokenize(response_text))
    return len(kp & response_tokens) / len(kp)


class SentenceEmbedder:
    """
    sentence-transformers 기반 임베딩 래퍼. 지연 로딩(첫 encode 호출 시 모델 로드).
    R_persona, R_diversity에서 하나의 인스턴스를 공유해서 쓴다 (모델 중복 로드 방지).
    """

    def __init__(
        self,
        model_id: str = "jhgan/ko-sroberta-multitask",
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_id, device=self.device)

    def encode(self, text: str):
        self._ensure_loaded()
        return self._model.encode(text, normalize_embeddings=True)

    def cos_sim(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        import numpy as np

        a, b = self.encode(text_a), self.encode(text_b)
        return float(np.dot(a, b))  # normalize_embeddings=True 이므로 내적이 곧 cosine similarity
