"""
rl/rewards/judge.py
────────────────────
성향 점수(S)와 반박 품질 점수(Q) 평가에 쓰이는 외부 Judge 모델 인터페이스.

로컬(예: Qwen2.5-7B) / API(OpenAI·Anthropic 호환) 두 가지 구현을 pluggable하게 지원한다.
GPU 서버 사양이 확정되기 전까지는 인터페이스만 두고, 실제 모델 로딩/호출은
config/reward.yaml 의 judge.backend 설정으로 둘 중 하나를 선택해 쓴다.
backend="none" 이면 judge를 아예 쓰지 않고 judge-free 컴포넌트만 동작한다
(v2 재설계는 judge 없이도 동작하도록 만들어졌다 — docs/reward_design_v2.md §1).

주의 (Judge Ceiling, docs/reward_design_v2.md §7 "Judging with Many Minds" arXiv:2505.19477):
LEFT/RIGHT와 같은 backbone(Qwen2.5)을 judge로 쓰면 self-preference bias가 생길 수 있으므로,
가능하면 다른 계열 모델(API 등)을 judge로 쓰는 것을 권장한다.
"""

from __future__ import annotations

import abc
import json
import os
from typing import Optional


class Judge(abc.ABC):
    """정치 성향 점수(S)와 반박 품질 점수(Q)를 매기는 판별자 인터페이스."""

    @abc.abstractmethod
    def score_stance(self, text: str) -> float:
        """text의 정치 성향을 [-1(진보), 1(보수)] 실수로 평가."""

    @abc.abstractmethod
    def score_rebuttal(self, opponent_text: str, response_text: str) -> int:
        """response_text가 opponent_text의 핵심 논점을 정면 반박했는지 [1,5] 정수로 평가."""


def _parse_judge_json(text: str) -> dict:
    """judge 응답에서 JSON 객체를 최대한 관대하게 파싱한다 (코드블록/설명 텍스트가 섞여도 시도)."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}


_STANCE_PROMPT = (
    "다음 발언의 정치적 성향을 -1(매우 진보)에서 +1(매우 보수) 사이 실수로 평가하라.\n"
    '발언: "{text}"\n'
    '반드시 {{"score": <float>}} 형식의 JSON으로만 답하라.'
)

_REBUTTAL_PROMPT = (
    "상대방 발언의 핵심 논점을 회피하지 않고 정면으로 반박했는지 1~5점으로 평가하라.\n"
    '상대방 발언: "{opponent_text}"\n'
    '현재 답변: "{response_text}"\n'
    '반드시 {{"score": <int 1-5>}} 형식의 JSON으로만 답하라.'
)


class LocalJudge(Judge):
    """
    로컬 소형 모델(예: Qwen2.5-7B-Instruct, 4bit)을 judge로 사용.

    LEFT/RIGHT가 이미 RTX 3090 24GB 두 장을 각 1개씩 4bit 14B로 채우고 있으므로
    (CLAUDE.md 참고), 이 judge는 별도 GPU 또는 여유 VRAM이 있는 서버에서만 로드 가능하다.
    GRPO의 그룹 샘플링(num_generations개 동시 생성)까지 고려하면 KV cache가 커지므로
    judge 전용 GPU를 분리하는 것을 권장한다 (config/reward.yaml judge.local.gpu_id).
    """

    def __init__(self, model_id: str, gpu_id: int, quantization: str = "4bit") -> None:
        self.model_id = model_id
        self.gpu_id = gpu_id
        self.quantization = quantization
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """GPU 서버에서 명시적으로 호출해야 함 (지연 로딩)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        load_kwargs: dict = {"device_map": {"": self.gpu_id}, "trust_remote_code": True}
        if self.quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        self._model.eval()

    def _generate_json(self, prompt: str) -> dict:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("LocalJudge.load()를 먼저 호출하세요.")
        import torch

        messages = [{"role": "user", "content": prompt}]
        encoded = self._tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"] if hasattr(encoded, "input_ids") else encoded
        input_ids = input_ids.to(f"cuda:{self.gpu_id}")

        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        text = self._tokenizer.decode(out[0][input_ids.shape[-1] :], skip_special_tokens=True)
        return _parse_judge_json(text)

    def score_stance(self, text: str) -> float:
        result = self._generate_json(_STANCE_PROMPT.format(text=text))
        return float(max(-1.0, min(1.0, result.get("score", 0.0))))

    def score_rebuttal(self, opponent_text: str, response_text: str) -> int:
        result = self._generate_json(
            _REBUTTAL_PROMPT.format(opponent_text=opponent_text, response_text=response_text)
        )
        return int(max(1, min(5, result.get("score", 3))))


class APIJudge(Judge):
    """
    외부 API(OpenAI 또는 Anthropic 호환 chat completion)를 judge로 사용.

    LEFT/RIGHT(Qwen2.5)와 다른 모델 계열을 쓸 수 있어 self-preference bias를
    줄이는 데 유리하다 (judge.py 상단 주의사항 참고). GPU 메모리 부담이 없는 대신
    학습 중 매 스텝 네트워크 호출이 발생하므로 롤아웃 배치가 커지면 지연시간에 유의.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str = "claude-haiku-4-5-20251001",
        api_key_env: str = "ANTHROPIC_API_KEY",
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self._client = None

    def _client_or_raise(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"환경변수 {self.api_key_env}가 설정되지 않았습니다. "
                "API judge를 쓰려면 GPU 서버에서 키를 export한 뒤 사용하세요."
            )
        if self.provider == "anthropic":
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)
        elif self.provider == "openai":
            import openai

            self._client = openai.OpenAI(api_key=api_key)
        else:
            raise ValueError(f"지원하지 않는 provider: {self.provider}")
        return self._client

    def _ask(self, prompt: str) -> dict:
        client = self._client_or_raise()
        if self.provider == "anthropic":
            resp = client.messages.create(
                model=self.model,
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
        else:  # openai
            resp = client.chat.completions.create(
                model=self.model,
                max_tokens=64,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content
        return _parse_judge_json(text)

    def score_stance(self, text: str) -> float:
        result = self._ask(_STANCE_PROMPT.format(text=text))
        return float(max(-1.0, min(1.0, result.get("score", 0.0))))

    def score_rebuttal(self, opponent_text: str, response_text: str) -> int:
        result = self._ask(
            _REBUTTAL_PROMPT.format(opponent_text=opponent_text, response_text=response_text)
        )
        return int(max(1, min(5, result.get("score", 3))))


def build_judge(cfg: dict) -> Optional[Judge]:
    """
    config/reward.yaml의 judge 섹션으로부터 Judge 인스턴스를 생성한다.
    backend가 "none"이면 None을 반환 — judge 없이 judge-free 컴포넌트만 사용.
    """
    backend = cfg.get("backend", "none")
    if backend == "none":
        return None
    if backend == "local":
        local_cfg = cfg["local"]
        judge = LocalJudge(
            model_id=local_cfg["model_id"],
            gpu_id=local_cfg["gpu_id"],
            quantization=local_cfg.get("quantization", "4bit"),
        )
        judge.load()
        return judge
    if backend == "api":
        api_cfg = cfg["api"]
        return APIJudge(
            provider=api_cfg.get("provider", "anthropic"),
            model=api_cfg["model"],
            api_key_env=api_cfg.get("api_key_env", "ANTHROPIC_API_KEY"),
        )
    raise ValueError(f"알 수 없는 judge backend: {backend!r} (none|local|api 중 하나여야 함)")
