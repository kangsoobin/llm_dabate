"""
rl/rewards/utils.py
────────────────────
Judge 없이(judge-free) 계산 가능한 보상 컴포넌트들이 공유하는 유틸리티.
- 형태소 단위 Jaccard 유사도 (R_novelty, v1 R_antirep)
- 핵심 논점(key point) 추출 및 커버리지 (R_engagement, ArgKP/KPA 경량 근사)
- 문장 임베딩 (R_persona, R_diversity)

2026-07-04 개선 (GRPO 실전 투입 전 정비):
- 토큰화를 정규식 어절 단위 → kiwipiepy 형태소 단위로 교체. 한국어는 조사가 어절에
  붙기 때문에 어절 단위로는 "최저임금을"과 "최저임금이"가 다른 토큰이 되어
  key-point coverage가 체계적으로 과소평가된다 (kiwi 미설치 시 기존 정규식으로 fallback).
- 임베딩 기본 모델을 ko-sroberta(max_seq_length 128 — 600토큰 발언이 앞부분만 잘림)
  → BAAI/bge-m3(8192 토큰)로 교체. config/reward.yaml에서 변경 가능.
- 임베딩 캐시 추가: GRPO에서 같은 opponent/history/anchor 텍스트가 그룹 내 8개 완성마다
  반복 인코딩되는 것을 방지.
"""

from __future__ import annotations

import re
from typing import Optional

# 형태소 분석기 fallback용 최소 불용어(조사/서술어성 어미) 근사 목록.
_STOPWORDS = {
    "그리고", "그러나", "하지만", "따라서", "때문에", "그런데", "그래서",
    "있다", "없다", "합니다", "입니다", "것이다", "것입니다", "합니다만",
}

_TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9]+")

# kiwipiepy 지연 로딩 싱글턴 (미설치 환경에서는 None 유지 → 정규식 fallback)
_KIWI = None
_KIWI_FAILED = False

# 내용어 품사: 명사(NNG/NNP), 동사/형용사 어간(VV/VA), 일반부사(MAG), 외국어(SL), 숫자(SN)
_CONTENT_TAGS = ("NNG", "NNP", "VV", "VA", "MAG", "SL", "SN")
_NOUN_TAGS = ("NNG", "NNP", "SL")


def _get_kiwi():
    global _KIWI, _KIWI_FAILED
    if _KIWI is not None or _KIWI_FAILED:
        return _KIWI
    try:
        from kiwipiepy import Kiwi

        _KIWI = Kiwi()
    except ImportError:
        _KIWI_FAILED = True
        print(
            "[rl.rewards.utils] kiwipiepy가 없어 어절 정규식 토큰화로 fallback합니다 — "
            "한국어 key-point coverage 정확도가 크게 떨어지므로 `uv pip install kiwipiepy` 권장."
        )
    return _KIWI


def tokenize(text: str) -> list[str]:
    """내용어 형태소 목록 (kiwi 미설치 시 어절 정규식 fallback)."""
    kiwi = _get_kiwi()
    if kiwi is None:
        return [t for t in _TOKEN_RE.findall(text) if t not in _STOPWORDS and len(t) > 1]
    return [
        tok.form
        for tok in kiwi.tokenize(text)
        if tok.tag.startswith(_CONTENT_TAGS) and len(tok.form) > 1
    ]


def jaccard_similarity(text: str, history: list[str]) -> float:
    """text와 history(과거 발언들 전체를 합친 것) 간 내용어 형태소 단위 Jaccard 유사도."""
    if not history:
        return 0.0
    a = set(tokenize(text))
    b = set(tokenize(" ".join(history)))
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def extract_key_points(text: str, top_n: int = 8) -> set[str]:
    """
    상대 발언에서 핵심 논점 후보(명사)를 빈도순으로 추출한다.
    ArgKP/KPA(ACL 2021 argmining-1.16) 스타일의 경량 근사 — kiwi 명사(NNG/NNP) 기반.
    """
    kiwi = _get_kiwi()
    if kiwi is None:
        tokens = tokenize(text)
    else:
        tokens = [
            tok.form
            for tok in kiwi.tokenize(text)
            if tok.tag.startswith(_NOUN_TAGS) and len(tok.form) > 1
        ]
    freq: dict[str, int] = {}
    for tok in tokens:
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

    - 텍스트→벡터 캐시 내장 (GRPO 그룹 내 반복 인코딩 방지, FIFO 상한 _CACHE_MAX)
    - encode_mean(): 여러 텍스트의 정규화 임베딩 평균 (R_persona anchor용,
      docs/reward_design_v2.md §3.1 "seed 응답 임베딩 평균")
    """

    _CACHE_MAX = 4096

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        device: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._cache: dict[str, "object"] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_id, device=self.device)

    def encode(self, text: str):
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        self._ensure_loaded()
        vec = self._model.encode(text, normalize_embeddings=True)
        if len(self._cache) >= self._CACHE_MAX:
            # 단순 FIFO: 가장 먼저 들어온 키부터 절반 비움
            for key in list(self._cache.keys())[: self._CACHE_MAX // 2]:
                del self._cache[key]
        self._cache[text] = vec
        return vec

    def encode_mean(self, texts: list[str]):
        """각 텍스트를 개별 인코딩한 뒤 평균 → 재정규화한 벡터를 반환."""
        import numpy as np

        vecs = [self.encode(t) for t in texts if t]
        if not vecs:
            raise ValueError("encode_mean: 비어있지 않은 텍스트가 하나도 없습니다.")
        mean = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean)
        return mean / norm if norm > 0 else mean

    def cos_sim(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        import numpy as np

        a, b = self.encode(text_a), self.encode(text_b)
        return float(np.dot(a, b))  # normalize_embeddings=True 이므로 내적이 곧 cosine similarity

    def cos_sim_vec(self, text: str, vec) -> float:
        """미리 계산된(정규화된) 벡터와의 cosine similarity."""
        if not text:
            return 0.0
        import numpy as np

        return float(np.dot(self.encode(text), vec))
