"""
rl/judge.py
───────────
LLM-as-a-Judge: Qwen2.5-7B-Instruct(4bit)로 발언의 정치 성향(stance)과
반박 품질(rebuttal)을 채점한다.

- stance ∈ [-1, 1]  (-1: 뚜렷한 진보, 0: 중립/양시론, +1: 뚜렷한 보수)
- rebuttal ∈ {1..5} (상대 논점을 정면으로 반박했는가; 상대 발언이 없으면 3)

2026-07-04 고도화 (논문 차용):
- compare_rebuttals(): 반박 품질을 절대 점수 대신 두 답변의 쌍대(pairwise) 비교로 판정.
  GRPO는 그룹 내 상대 비교(advantage)라 절대 점수가 필요 없고, LLM judge는 절대 채점보다
  쌍대 비교가 신뢰성이 높다 (Argument Quality Assessment with LLMs: A Pairwise
  Bradley-Terry Approach, arXiv:2605.28313). rl/rewards.py가 비교 결과를 Bradley-Terry로
  집계해 그룹 내 강도 점수로 변환한다.
- APIJudgeClient / RoutingJudge: 로컬 judge가 불확실한(중립 근처 stance, 파싱 실패) 샘플만
  강한 외부 API judge로 재채점하는 불확실성 라우팅 (Ask a Strong LLM Judge when Your
  Reward Model is Uncertain, arXiv:2510.20369). 로컬과 다른 계열 모델을 쓰므로
  self-preference bias 완화 효과도 있다. 기본 비활성(rl/config.yaml judge_routing.enabled).
"""

from __future__ import annotations

import json
import os
import re

JUDGE_SYSTEM = """당신은 한국 정치 토론 발언을 평가하는 공정하고 엄격한 심판입니다. 두 가지 항목을 채점합니다.

[stance — 정치 성향]
발언의 실제 주장과 논거 내용만으로 정치 성향을 -1.0에서 +1.0 사이 점수로 매깁니다.
-1.0: 뚜렷한 진보 (복지 확대, 재벌 개혁, 노동자 권리, 증세, 남북 협력 등을 강하게 주장)
 0.0: 중립, 양시론, 절충적 결론, 입장 회피
+1.0: 뚜렷한 보수 (시장 자유, 규제 완화, 감세, 재정 건전성, 한미동맹 강화 등을 강하게 주장)
발언자가 자신을 어떻게 소개하는지, 어떤 정당 이름을 말하는지는 무시하고 논거의 실질 내용만 봅니다.

[rebuttal — 반박 품질]
상대 발언의 핵심 논점을 회피하지 않고 정면으로 반박했는지 1~5점으로 매깁니다.
1: 상대 논점을 완전히 무시하고 자기 주장만 반복
2: 상대 논점을 스치듯 언급만 하고 실질 반박 없음
3: 부분적으로 반박했으나 핵심 논점은 회피 (상대 발언이 없는 경우에도 3)
5: 상대의 핵심 논점을 구체적 근거로 정면 반박
구호나 정치적 키워드만 나열하고 상대의 구체적 논거를 다루지 않으면 반드시 2점 이하를 줍니다.

[예시 1]
상대 발언: 최저임금을 급격히 올리면 자영업자가 무너지고 일자리가 줄어듭니다.
평가 대상: 자영업자가 어렵다는 말씀은 본질을 흐리는 것입니다. 자영업 위기의 진짜 원인은 임대료와 프랜차이즈 수수료이지 노동자의 최저임금이 아닙니다. 최저임금 인상은 소비 여력을 키워 골목상권을 살리는 길입니다.
채점: {"stance": -0.8, "rebuttal": 5}

[예시 2]
상대 발언: 법인세를 인하해야 기업 투자가 늘고 일자리가 생깁니다.
평가 대상: 서민! 민생! 노동자! 재벌 개혁 없이는 미래가 없습니다. 불평등 해소가 시대정신입니다. 우리는 반드시 승리합니다.
채점: {"stance": -0.9, "rebuttal": 1}

[예시 3]
상대 발언: 종부세는 징벌적 과세이므로 폐지해야 합니다.
평가 대상: 양쪽 모두 일리가 있는 복잡한 문제입니다. 세금 부담과 부동산 안정화 사이에서 균형 잡힌 접근이 필요하다고 봅니다.
채점: {"stance": 0.0, "rebuttal": 2}

반드시 JSON 한 줄로만 답합니다: {"stance": <숫자>, "rebuttal": <정수>}"""

PAIRWISE_SYSTEM = """당신은 한국 정치 토론 발언을 평가하는 공정하고 엄격한 심판입니다.
상대 발언의 핵심 논점을 답변 A와 답변 B 중 어느 쪽이 더 정면으로, 구체적인 근거를 들어 반박했는지 판정합니다.

판정 기준:
- 상대의 구체적 논거를 직접 다루고 근거를 들어 재반박하면 우수합니다.
- 구호나 정치적 키워드만 나열하거나, 상대 논점을 회피하고 자기 주장만 반복하면 열등합니다.
- 두 답변의 반박 품질이 실질적으로 같으면 "tie"로 판정합니다.
- 발언의 정치적 방향이 아니라 오직 "반박의 품질"만 비교합니다.

[예시]
상대 발언: 최저임금을 급격히 올리면 자영업자가 무너집니다.
답변 A: 자영업 위기의 진짜 원인은 임대료와 프랜차이즈 수수료입니다. 최저임금 인상은 소비 여력을 키워 골목상권을 살립니다.
답변 B: 서민! 노동자! 우리는 반드시 승리합니다. 불평등 해소가 시대정신입니다.
판정: {"winner": "A"}

반드시 JSON 한 줄로만 답합니다: {"winner": "A"} 또는 {"winner": "B"} 또는 {"winner": "tie"}"""

_JSON_RE = re.compile(r"\{[^{}]*\}")
_NUM_RE = {
    "stance": re.compile(r'"?stance"?\s*[:=]\s*(-?\d+(?:\.\d+)?)'),
    "rebuttal": re.compile(r'"?rebuttal"?\s*[:=]\s*(\d+)'),
}
_WINNER_RE = re.compile(r'"?winner"?\s*[:=]\s*"?(A|B|tie)"?', re.IGNORECASE)

DEFAULT_SCORE = {"stance": 0.0, "rebuttal": 3, "parse_ok": False}
DEFAULT_VERDICT = {"winner": "tie", "parse_ok": False}


def _build_user_prompt(item: dict) -> str:
    question = item.get("question", "")
    opponent = (item.get("opponent") or "").strip()[:600]
    reply = (item.get("reply") or "").strip()[:600]
    parts = [f"[토론 주제]\n{question}"]
    if opponent:
        parts.append(f"[상대 발언]\n{opponent}")
    else:
        parts.append("[상대 발언]\n(없음 — rebuttal은 3으로 채점)")
    parts.append(f"[평가 대상 발언]\n{reply}")
    parts.append('채점 결과를 JSON 한 줄로만 출력하세요: {"stance": <숫자>, "rebuttal": <정수>}')
    return "\n\n".join(parts)


def _build_pairwise_prompt(item: dict) -> str:
    question = item.get("question", "")
    opponent = (item.get("opponent") or "").strip()[:600]
    reply_a = (item.get("reply_a") or "").strip()[:600]
    reply_b = (item.get("reply_b") or "").strip()[:600]
    return (
        f"[토론 주제]\n{question}\n\n"
        f"[상대 발언]\n{opponent}\n\n"
        f"[답변 A]\n{reply_a}\n\n"
        f"[답변 B]\n{reply_b}\n\n"
        '어느 답변이 상대 발언을 더 잘 반박했습니까? JSON 한 줄로만 답하세요: {"winner": "A"|"B"|"tie"}'
    )


def parse_pairwise_output(text: str) -> dict:
    """쌍대 비교 출력에서 winner를 추출한다. 실패 시 tie 기본값(BT 집계에 중립)."""
    match = _WINNER_RE.search(text)
    if not match:
        return dict(DEFAULT_VERDICT)
    winner = match.group(1).lower()
    return {"winner": winner if winner == "tie" else winner, "parse_ok": True}


def parse_judge_output(text: str) -> dict:
    """Judge 출력에서 stance/rebuttal을 추출한다. 실패 시 중립 기본값."""
    match = _JSON_RE.search(text)
    stance, rebuttal = None, None
    if match:
        try:
            obj = json.loads(match.group(0))
            stance = float(obj.get("stance"))
            rebuttal = int(round(float(obj.get("rebuttal"))))
        except (ValueError, TypeError, json.JSONDecodeError):
            stance, rebuttal = None, None
    if stance is None or rebuttal is None:
        m_s = _NUM_RE["stance"].search(text)
        m_r = _NUM_RE["rebuttal"].search(text)
        if m_s and m_r:
            stance, rebuttal = float(m_s.group(1)), int(m_r.group(1))
        else:
            return dict(DEFAULT_SCORE)
    return {
        "stance": max(-1.0, min(1.0, stance)),
        "rebuttal": max(1, min(5, rebuttal)),
        "parse_ok": True,
    }


class JudgeClient:
    """Qwen2.5-7B-Instruct 4bit를 지정 GPU에 로드해 배치 채점을 수행한다."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        gpu: int = 1,
        max_new_tokens: int = 64,
        batch_size: int = 16,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.device = f"cuda:{gpu}"

        print(f"[Judge] {model_id} 로딩 중 (GPU {gpu}, 4bit)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": gpu},
            trust_remote_code=True,
        )
        self.model.eval()
        print("[Judge] 로드 완료.")

    def _generate_batch(self, system: str, user_prompts: list[str]) -> list[str]:
        """system+user 프롬프트 목록을 left-padding 배치로 greedy 생성한다."""
        import torch

        decoded_all: list[str] = []
        for start in range(0, len(user_prompts), self.batch_size):
            chunk = user_prompts[start:start + self.batch_size]
            texts = [
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in chunk
            ]
            encoded = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            new_tokens = output[:, encoded["input_ids"].shape[1]:]
            decoded_all.extend(self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        return decoded_all

    def score_batch(self, items: list[dict]) -> list[dict]:
        """items: [{question, opponent, reply}] → [{stance, rebuttal, parse_ok}]"""
        decoded = self._generate_batch(JUDGE_SYSTEM, [_build_user_prompt(item) for item in items])
        return [parse_judge_output(text) for text in decoded]

    def compare_rebuttals(self, items: list[dict]) -> list[dict]:
        """items: [{question, opponent, reply_a, reply_b}] → [{winner: "a"|"b"|"tie", parse_ok}]

        절대 점수(1~5) 대신 두 답변 중 어느 쪽이 상대 논점을 더 잘 반박했는지 판정한다
        (arXiv:2605.28313의 pairwise 접근). 위치 편향(A 자리 선호)은 호출자(rl/rewards.py)가
        쌍마다 A/B 배치를 교대시켜 완화한다.
        """
        decoded = self._generate_batch(PAIRWISE_SYSTEM, [_build_pairwise_prompt(item) for item in items])
        return [parse_pairwise_output(text) for text in decoded]


class APIJudgeClient:
    """외부 API(Anthropic/OpenAI 호환)를 judge로 사용 — RoutingJudge의 재채점 백엔드.

    로컬 7B와 다른 계열 모델이라 self-preference bias 완화에도 유리하다. 생성자에서
    클라이언트를 즉시 만들며, 키/패키지가 없으면 RuntimeError를 던진다 — 호출자(build_judge)가
    잡아서 라우팅 없이 로컬 judge만으로 계속 진행한다(soft-fail).
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5-20251001",
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = 64,
    ) -> None:
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"환경변수 {api_key_env}가 설정되지 않았습니다.")
        if provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=api_key)
        elif provider == "openai":
            import openai
            self._client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"지원하지 않는 provider: {provider!r} (anthropic|openai)")

    def _ask(self, system: str, user_prompt: str) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return resp.content[0].text
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content

    def score_batch(self, items: list[dict]) -> list[dict]:
        results = []
        for item in items:
            try:
                text = self._ask(JUDGE_SYSTEM, _build_user_prompt(item))
                results.append(parse_judge_output(text))
            except Exception as e:  # noqa: BLE001 — API 일시 장애가 학습을 죽이면 안 됨
                print(f"[경고] API judge 호출 실패({e!r}) — 해당 샘플은 로컬 점수 유지")
                results.append(dict(DEFAULT_SCORE))
        return results


class RoutingJudge:
    """불확실성 라우팅 judge (arXiv:2510.20369 차용).

    로컬 judge 점수 중 (a) 파싱 실패했거나 (b) stance가 중립 근처(|stance| < threshold)라
    신호가 약한 샘플만 골라 강한 API judge로 재채점한다. 중립 근처가 가장 중요한 이유:
    stance 보상의 학습 신호가 갈리는 구간이 바로 거기라서, 로컬 judge의 노이즈가 그대로
    보상 노이즈가 되기 때문. 쌍대 비교(compare_rebuttals)는 로컬에 그대로 위임한다
    (비교 판정은 절대 채점보다 로컬 judge도 신뢰할 만하다는 것이 pairwise 접근의 전제).
    """

    def __init__(self, local: JudgeClient, api: APIJudgeClient, stance_uncertain_below: float = 0.3) -> None:
        self.local = local
        self.api = api
        self.threshold = stance_uncertain_below
        self.last_routed_frac = 0.0

    def score_batch(self, items: list[dict]) -> list[dict]:
        scores = self.local.score_batch(items)
        uncertain = [
            i for i, s in enumerate(scores)
            if (not s["parse_ok"]) or abs(s["stance"]) < self.threshold
        ]
        self.last_routed_frac = len(uncertain) / max(1, len(scores))
        if uncertain:
            rescored = self.api.score_batch([items[i] for i in uncertain])
            for i, new_score in zip(uncertain, rescored):
                if new_score["parse_ok"]:
                    scores[i] = new_score
        return scores

    def compare_rebuttals(self, items: list[dict]) -> list[dict]:
        return self.local.compare_rebuttals(items)


def build_judge(cfg: dict, gpu: int):
    """rl/config.yaml 설정으로 judge를 조립한다. 라우팅 활성 시 RoutingJudge로 감싼다."""
    local = JudgeClient(
        model_id=cfg["judge_model_id"],
        gpu=gpu,
        max_new_tokens=int(cfg["judge_max_new_tokens"]),
        batch_size=int(cfg["judge_batch_size"]),
    )
    routing = cfg.get("judge_routing", {})
    if not routing.get("enabled", False):
        return local
    try:
        api = APIJudgeClient(
            provider=routing.get("provider", "anthropic"),
            model=routing.get("model", "claude-haiku-4-5-20251001"),
            api_key_env=routing.get("api_key_env", "ANTHROPIC_API_KEY"),
        )
    except Exception as e:  # noqa: BLE001
        print(f"[경고] API judge 초기화 실패({e!r}) — 라우팅 없이 로컬 judge만 사용합니다.")
        return local
    print(f"[Judge] 불확실성 라우팅 활성 (|stance| < {routing.get('stance_uncertain_below', 0.3)} → "
          f"{routing.get('provider')}/{routing.get('model')})")
    return RoutingJudge(local, api, float(routing.get("stance_uncertain_below", 0.3)))
