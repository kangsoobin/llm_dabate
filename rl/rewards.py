"""
rl/rewards.py
─────────────
GRPO 보상 함수 8종 (2026-07-04, Kanana 전환 + 논문 차용 고도화).

Judge 기반:
  1. reward_stance          성향 일관성: LEFT는 -S, RIGHT는 +S. RoutingJudge(rl/judge.py)를
                             쓰면 중립 근처 불확실 샘플만 API judge로 재채점 (arXiv:2510.20369)
  2. reward_rebuttal        반박 품질. 기본은 pairwise 모드 — judge가 그룹 내 답변 쌍을 비교하고
                             Bradley-Terry로 집계해 [0,1] 강도 점수를 만든다 (arXiv:2605.28313).
                             GRPO advantage가 어차피 그룹 내 상대 비교라 절대 점수(1~5)보다
                             노이즈가 적다. rebuttal_mode="absolute"면 기존 (Q-1)/4 방식.
                             상대 발언 없는 행(1라운드)은 None(제외)

규칙/임베딩 기반 (judge 무관, 값싸고 검증 가능):
  3. reward_engagement      상대 발언 핵심 키워드 실제 언급 비율 — judge 반박 점수의
                             독립적 대조 신호 (judge 하나에만 의존하는 리스크 완화)
  4. reward_persona         [신규] 발언과 페르소나 정의 텍스트의 임베딩 정합도(prompt-to-line
                             consistency, arXiv:2511.00222) — judge 없이 페르소나 이탈(중립
                             회귀)을 직접 보상. 임베딩 못 쓰면 0 처리(soft-fail)
  5. reward_antirep         자기 반복: -γ·max(0, max_j Jaccard(y, H_j) - τ)  (글자 그대로 반복)
  6. reward_semantic_echo   임베딩 기반 — 다른 단어로 같은 말 반복(의미적 자기반복)
                             + 상대 발언 쪽으로의 은근한 수렴(동조)까지 감지. soft-fail 지원
  7. reward_neutral_phrase  중립 표현 블랙리스트 패널티
  8. reward_format          너무 짧음 / 미종결 / 목록 형식 패널티

2026-07-04 변경:
  - reward_korean_ratio(한글 비율/중국어 드리프트 패널티) 제거 — Qwen은 다국어 모델이라
    중국어로 새는 경향이 있었지만 Kanana(카카오, 한국어 특화)는 그 리스크가 훨씬 낮아 불필요.
  - engagement/semantic_echo 신규 추가(판정 신호 다각화), persona/pairwise/라우팅 논문 차용.

trl 1.5.1 규약: 각 함수는 f(prompts, completions, completion_ids, **kwargs)로 호출되며
데이터셋의 추가 컬럼(question, opponent_statement, own_history, ...)이 kwargs로 전달된다.
"""

from __future__ import annotations

import re
from itertools import combinations

# readme.md 검증 프로토콜 + prompts.yaml 금지 목록에서 추출한 중립 드리프트 표현
NEUTRAL_PHRASES = [
    "양쪽 다 일리",
    "양측 다 일리",
    "양쪽 모두 일리",
    "양측 모두 일리",
    "균형 잡힌 시각",
    "균형잡힌 시각",
    "복잡한 문제입니다",
    "복잡한 사안입니다",
    "일부 동의",
    "절충",
    "양측 모두",
    "중립적",
]

_WORD_RE = re.compile(r"[가-힣A-Za-z0-9]+")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-•▪*]|\d+[.)])\s")

# 형태소 분석기 없이 쓰는 최소한의 한국어 불용어(조사성 서술어/접속사) 근사 목록.
# reward_engagement의 핵심 키워드 추출에서 이런 토큰이 "키워드"로 잘못 뽑히는 것을 막는다.
_STOPWORDS_KO = {
    "그리고", "그러나", "하지만", "따라서", "때문에", "그런데", "그래서", "그리하여",
    "있다", "없다", "합니다", "입니다", "것이다", "것입니다", "합니다만", "습니다",
    "우리는", "저는", "당신은", "이것은", "그것은",
}


def completion_text(completion) -> str:
    """trl 대화형 completion([{"role","content"}]) 또는 문자열에서 텍스트 추출."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        return completion[-1].get("content", "")
    return ""


# ────────────────────────────────────────────────────────────
# 규칙 기반 순수 함수 (단위 테스트 대상)
# ────────────────────────────────────────────────────────────

def jaccard(a: str, b: str) -> float:
    wa, wb = set(_WORD_RE.findall(a)), set(_WORD_RE.findall(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def antirep_penalty(text: str, history: list[str], tau: float = 0.25, gamma: float = 1.5) -> float:
    if not history:
        return 0.0
    j_max = max(jaccard(text, h) for h in history)
    return -gamma * max(0.0, j_max - tau)


def neutral_phrase_penalty(text: str) -> float:
    hits = sum(text.count(p) for p in NEUTRAL_PHRASES)
    return max(-1.5, -0.5 * hits)


def extract_key_terms(text: str, top_n: int = 6) -> set[str]:
    """상대 발언에서 핵심 어휘 후보를 빈도 기반으로 근사 추출한다 (reward_engagement용).

    형태소 분석기 없이 2글자 이상 토큰의 빈도로 근사한다 — 정교하지 않지만 judge와
    독립적으로 "핵심어를 언급이라도 했는가"를 값싸고 검증 가능하게 측정할 수 있다.
    """
    freq: dict[str, int] = {}
    for tok in _WORD_RE.findall(text):
        if tok in _STOPWORDS_KO or len(tok) < 2:
            continue
        freq[tok] = freq.get(tok, 0) + 1
    ranked = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)
    return {tok for tok, _ in ranked[:top_n]}


def engagement_score(opponent_text: str, response_text: str) -> float:
    """opponent_text의 핵심 어휘 중 response_text가 실제로 언급한 비율 ∈ [0,1].

    judge의 1~5점 반박 평가(주관적)와 독립적인 기계적 검증 신호 — 키워드만 나열하고
    실제로는 회피하는 경우를 judge가 놓쳐도(Judge Ceiling) 이 신호는 최소한 "언급은
    했는가"를 잡아낸다. 반대로 이 신호만으로는 "언급했지만 반박은 안 함"을 못 잡으므로
    judge의 reward_rebuttal과 상호보완적으로 쓴다(둘 다 개별 보상으로 유지).
    """
    if not opponent_text:
        return 0.0
    kp = extract_key_terms(opponent_text)
    if not kp:
        return 0.0
    response_terms = set(_WORD_RE.findall(response_text))
    return len(kp & response_terms) / len(kp)


class SemanticEmbedder:
    """문장 임베딩 지연 로딩 래퍼. 모델 로드/인코딩 실패 시 예외를 던지지 않고 None을
    반환해 reward_semantic_echo가 조용히 0으로 우회하도록 한다(soft-fail) — sentence-
    transformers가 설치되지 않은 서버에서도 나머지 6개 보상은 정상 동작해야 하기 때문.
    """

    def __init__(self, model_id: str = "jhgan/ko-sroberta-multitask", device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._load_failed = False

    def _ensure_loaded(self) -> None:
        if self._model is not None or self._load_failed:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_id, device=self.device)
        except Exception as e:  # noqa: BLE001 — 임베딩은 보조 신호이므로 실패해도 학습은 계속되어야 함
            print(f"[경고] SemanticEmbedder 로드 실패({e!r}) — reward_semantic_echo는 0으로 우회합니다.")
            self._load_failed = True

    def cos_sim(self, text_a: str, text_b: str) -> float | None:
        if not text_a or not text_b:
            return 0.0
        self._ensure_loaded()
        if self._model is None:
            return None
        import numpy as np
        va, vb = self._model.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(va, vb))


def semantic_echo_penalty(
    text: str,
    own_history: list[str],
    opponent_statement: str,
    embedder: SemanticEmbedder,
    tau: float = 0.75,
    gamma: float = 1.0,
) -> float:
    """reward_antirep(글자 그대로의 반복)이 못 잡는 두 가지를 임베딩 유사도로 잡는다.

    1. 의미적 자기반복: 다른 단어로 과거 자기 발언과 같은 주장을 반복하는 경우
    2. 은근한 동조: 이번 발언이 상대 발언과 의미적으로 지나치게 가까워지는 경우
       (논거 없이 상대 쪽으로 수렴 — sycophancy)

    antirep과 같은 tau/gamma 임계값-패널티 패턴을 그대로 따르되, 비교 대상을 Jaccard가
    아닌 코사인 유사도로 확장한다. 임베딩을 못 쓰면(soft-fail) 0을 반환해 안전하게 우회.
    """
    candidates = list(own_history) + ([opponent_statement] if opponent_statement else [])
    if not candidates:
        return 0.0
    sims = [embedder.cos_sim(text, c) for c in candidates]
    sims = [s for s in sims if s is not None]
    if not sims:
        return 0.0  # 임베딩 로드 실패(soft-fail) — 이 컴포넌트만 0, 나머지 보상은 정상 진행
    peak = max(sims)
    return -gamma * max(0.0, peak - tau)


def format_penalty(text: str, n_completion_tokens: int, max_completion_length: int) -> float:
    penalty = 0.0
    stripped = text.strip()
    if len(stripped) < 80:
        penalty -= 1.0
    # 토큰 캡에 걸렸는데 문장이 끝나지 않음 = 잘린 발언
    if n_completion_tokens >= max_completion_length and not stripped.endswith((".", "!", "?", "…")):
        penalty -= 0.5
    if _LIST_MARKER_RE.search(stripped):
        penalty -= 0.5
    return penalty


def persona_consistency_score(text: str, persona_text: str, embedder: "SemanticEmbedder") -> float:
    """발언이 페르소나 정의 텍스트와 의미적으로 정합하는 정도 ∈ [0,1].

    arXiv:2511.00222(Consistently Simulating Human Personas with Multi-Turn RL)의
    prompt-to-line consistency를 임베딩 코사인 유사도로 경량 구현한 것. GRPO는 그룹 내
    z-score라 절대값이 아닌 그룹 내 상대 차이만 학습 신호가 되므로, "페르소나에서 덜 벗어난
    답변"이 그룹 안에서 상대적으로 높은 보상을 받는다. 임베딩 실패 시 0(soft-fail).
    """
    if not persona_text:
        return 0.0
    sim = embedder.cos_sim(text, persona_text)
    if sim is None:
        return 0.0
    return max(0.0, min(1.0, sim))


def group_consecutive(keys: list) -> list[list[int]]:
    """연속된 동일 키 구간을 인덱스 그룹으로 묶는다.

    trl GRPOTrainer는 같은 프롬프트의 num_generations개 completion을 연속 배치로 전달하므로,
    (question, opponent_statement)가 같은 연속 구간 = 같은 GRPO 그룹이다.
    """
    groups: list[list[int]] = []
    for i, key in enumerate(keys):
        if groups and keys[groups[-1][-1]] == key:
            groups[-1].append(i)
        else:
            groups.append([i])
    return groups


def bradley_terry_scores(n_items: int, comparisons: list[tuple], iters: int = 50) -> list[float]:
    """쌍대 비교 결과를 Bradley-Terry MLE(minorization-maximization)로 집계해
    그룹 내 강도 점수를 [0,1] min-max 정규화로 반환한다 (arXiv:2605.28313).

    comparisons: [(i, j, w_i, w_j), ...] — i와 j의 비교에서 각자가 얻은 승수
                 (승리 1.0/0.0, 무승부 0.5/0.5).
    """
    eps = 1e-6
    wins = [0.0] * n_items
    opponents: list[list[tuple[int, float]]] = [[] for _ in range(n_items)]  # (상대, 대국수)
    for i, j, w_i, w_j in comparisons:
        n_games = w_i + w_j
        wins[i] += w_i
        wins[j] += w_j
        opponents[i].append((j, n_games))
        opponents[j].append((i, n_games))

    strengths = [1.0] * n_items
    for _ in range(iters):
        new = []
        for i in range(n_items):
            denom = sum(n_games / (strengths[i] + strengths[j]) for j, n_games in opponents[i])
            new.append(max(wins[i], eps) / denom if denom > 0 else strengths[i])
        total = sum(new)
        strengths = [s * n_items / total for s in new] if total > 0 else new

    lo, hi = min(strengths), max(strengths)
    if hi - lo < 1e-9:
        return [0.5] * n_items
    return [(s - lo) / (hi - lo) for s in strengths]


# ────────────────────────────────────────────────────────────
# 보상 함수 팩토리
# ────────────────────────────────────────────────────────────

def make_reward_funcs(side: str, judge, cfg: dict,
                      persona_text: str | None = None) -> tuple[list, list[float]]:
    """side("left"|"right")용 보상 함수 목록과 가중치 목록을 반환한다.

    persona_text: config/prompts.yaml의 해당 side 페르소나 전문 — reward_persona가
    prompt-to-line 정합도의 기준점으로 사용. None이면 persona 보상은 항상 0.
    """
    assert side in ("left", "right")
    sign = -1.0 if side == "left" else 1.0
    tau = float(cfg.get("antirep_tau", 0.25))
    gamma = float(cfg.get("antirep_gamma", 1.5))
    semantic_tau = float(cfg.get("semantic_echo_tau", 0.75))
    semantic_gamma = float(cfg.get("semantic_echo_gamma", 1.0))
    max_completion_length = int(cfg.get("max_completion_length", 400))
    rebuttal_mode = cfg.get("rebuttal_mode", "pairwise")
    max_pairs = int(cfg.get("rebuttal_pairwise_max_pairs", 28))
    embedder = SemanticEmbedder(model_id=cfg.get("embedding_model", "jhgan/ko-sroberta-multitask"))

    # judge 호출 결과를 같은 completion 배치 안에서 공유한다 (절대 채점 1회 + 쌍대 비교 1회).
    # trl은 한 스텝 내 모든 reward func에 동일한 completions 리스트 객체를 넘긴다.
    cache: dict = {"key": None, "scores": None, "pairwise": None}

    def _reset_cache_if_new(completions):
        key = id(completions)
        if cache["key"] != key:
            cache["key"] = key
            cache["scores"] = None
            cache["pairwise"] = None

    def _judge_scores(completions, question, opponent_statement, log_metric):
        _reset_cache_if_new(completions)
        if cache["scores"] is None:
            items = [
                {
                    "question": q,
                    "opponent": opp or "",
                    "reply": completion_text(c),
                }
                for c, q, opp in zip(completions, question, opponent_statement)
            ]
            scores = judge.score_batch(items)
            cache["scores"] = scores
            if log_metric is not None:
                fail = sum(1 for s in scores if not s["parse_ok"]) / max(1, len(scores))
                log_metric("judge/parse_fail_rate", fail)
                if hasattr(judge, "last_routed_frac"):
                    log_metric("judge/routed_frac", judge.last_routed_frac)
        return cache["scores"]

    def _pairwise_rebuttal(completions, question, opponent_statement, log_metric):
        """그룹 내 쌍대 비교 → Bradley-Terry 강도 점수 (arXiv:2605.28313)."""
        _reset_cache_if_new(completions)
        if cache["pairwise"] is not None:
            return cache["pairwise"]

        texts = [completion_text(c) for c in completions]
        results: list[float | None] = [None] * len(texts)
        groups = group_consecutive(list(zip(question, opponent_statement)))

        pair_items, pair_meta = [], []  # meta: (group_idx, local_a, local_b) — a가 답변 A 자리
        for gi, g in enumerate(groups):
            if not opponent_statement[g[0]]:
                continue  # 1라운드(상대 발언 없음) → None 유지, trl이 해당 컴포넌트 제외
            if len(g) == 1:
                results[g[0]] = 0.5
                continue
            pairs = list(combinations(range(len(g)), 2))
            if len(pairs) > max_pairs:  # 그룹이 크면 결정적 등간격 서브샘플
                step = len(pairs) / max_pairs
                pairs = [pairs[int(k * step)] for k in range(max_pairs)]
            for a, b in pairs:
                if (a + b) % 2 == 1:
                    a, b = b, a  # 쌍마다 A/B 자리를 교대시켜 judge 위치 편향 완화
                pair_items.append({
                    "question": question[g[a]],
                    "opponent": opponent_statement[g[a]],
                    "reply_a": texts[g[a]],
                    "reply_b": texts[g[b]],
                })
                pair_meta.append((gi, a, b))

        verdicts = judge.compare_rebuttals(pair_items) if pair_items else []
        comparisons_by_group: dict[int, list[tuple]] = {}
        for (gi, a, b), verdict in zip(pair_meta, verdicts):
            w = verdict["winner"]
            wa, wb = (1.0, 0.0) if w == "a" else (0.0, 1.0) if w == "b" else (0.5, 0.5)
            comparisons_by_group.setdefault(gi, []).append((a, b, wa, wb))

        for gi, g in enumerate(groups):
            comps = comparisons_by_group.get(gi)
            if comps:
                strengths = bradley_terry_scores(len(g), comps)
                for local, idx in enumerate(g):
                    results[idx] = strengths[local]

        if log_metric is not None and verdicts:
            fail = sum(1 for v in verdicts if not v["parse_ok"]) / len(verdicts)
            log_metric("judge/pairwise_parse_fail_rate", fail)
        cache["pairwise"] = results
        return results

    def reward_stance(prompts, completions, completion_ids=None,
                      question=None, opponent_statement=None, log_metric=None, **kwargs):
        scores = _judge_scores(completions, question, opponent_statement, log_metric)
        return [sign * s["stance"] for s in scores]

    def reward_rebuttal(prompts, completions, completion_ids=None,
                        question=None, opponent_statement=None, log_metric=None, **kwargs):
        if rebuttal_mode == "pairwise" and hasattr(judge, "compare_rebuttals"):
            return _pairwise_rebuttal(completions, question, opponent_statement, log_metric)
        scores = _judge_scores(completions, question, opponent_statement, log_metric)
        return [
            None if not opp else (s["rebuttal"] - 1) / 4.0
            for s, opp in zip(scores, opponent_statement)
        ]

    def reward_engagement(prompts, completions, completion_ids=None,
                          opponent_statement=None, **kwargs):
        return [
            engagement_score(opp or "", completion_text(c))
            for c, opp in zip(completions, opponent_statement)
        ]

    def reward_persona(prompts, completions, completion_ids=None, **kwargs):
        return [
            persona_consistency_score(completion_text(c), persona_text or "", embedder)
            for c in completions
        ]

    def reward_antirep(prompts, completions, completion_ids=None, own_history=None, **kwargs):
        return [
            antirep_penalty(completion_text(c), h or [], tau, gamma)
            for c, h in zip(completions, own_history)
        ]

    def reward_semantic_echo(prompts, completions, completion_ids=None,
                             own_history=None, opponent_statement=None, **kwargs):
        return [
            semantic_echo_penalty(completion_text(c), h or [], opp or "", embedder, semantic_tau, semantic_gamma)
            for c, h, opp in zip(completions, own_history, opponent_statement)
        ]

    def reward_neutral_phrase(prompts, completions, completion_ids=None, **kwargs):
        return [neutral_phrase_penalty(completion_text(c)) for c in completions]

    def reward_format(prompts, completions, completion_ids=None, **kwargs):
        return [
            format_penalty(completion_text(c), len(ids), max_completion_length)
            for c, ids in zip(completions, completion_ids)
        ]

    funcs = [
        reward_stance,
        reward_rebuttal,
        reward_engagement,
        reward_persona,
        reward_antirep,
        reward_semantic_echo,
        reward_neutral_phrase,
        reward_format,
    ]
    weight_cfg = cfg.get("reward_weights", {})
    weights = [
        float(weight_cfg.get("stance", 1.0)),
        float(weight_cfg.get("rebuttal", 0.7)),
        float(weight_cfg.get("engagement", 0.5)),
        float(weight_cfg.get("persona", 0.5)),
        float(weight_cfg.get("antirep", 1.0)),
        float(weight_cfg.get("semantic_echo", 0.7)),
        float(weight_cfg.get("neutral_phrase", 1.0)),
        float(weight_cfg.get("format", 0.5)),
    ]
    return funcs, weights
