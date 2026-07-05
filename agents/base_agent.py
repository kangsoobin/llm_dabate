"""
agents/base_agent.py
────────────────────
LLM 모델 로딩, 추론, 스트리밍을 담당하는 기반 Agent 클래스.
LEFT/RIGHT Agent 모두 이 클래스를 사용하며, 시스템 프롬프트와 GPU 번호만 다르다.
"""

from __future__ import annotations

import torch
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from typing import Callable, Optional


class BaseAgent:
    """
    단일 GPU에 4-bit(또는 fp16/8bit) 로 모델을 로드하고
    스트리밍 토큰 생성을 지원하는 LLM Agent.

    Parameters
    ----------
    model_id : str
        HuggingFace 모델 ID (예: "Qwen/Qwen2.5-14B-Instruct")
    gpu_id : int
        이 Agent가 단독 점유할 CUDA 장치 번호 (0 or 1)
    system_prompt : str
        모델에 주입할 시스템 프롬프트 (페르소나 정의)
    generation_config : dict
        {max_new_tokens, temperature, top_p, repetition_penalty}
    quantization : str
        "4bit" | "8bit" | "fp16"
    name : str
        사람이 읽을 수 있는 Agent 이름 (예: "이진영 교수")
    side : str
        "left" | "right"
    """

    def __init__(
        self,
        model_id: str,
        gpu_id: int,
        system_prompt: str,
        generation_config: dict,
        quantization: str = "4bit",
        name: str = "Agent",
        side: str = "neutral",
        adapter_path: Optional[str] = None,
    ) -> None:
        self.model_id = model_id
        self.gpu_id = gpu_id
        self.system_prompt = system_prompt
        self.gen_config = generation_config
        self.quantization = quantization
        self.name = name
        self.side = side
        self.adapter_path = adapter_path

        # 대화 히스토리 (무제한 유지)
        # 형식: [{"role": "user"|"assistant", "content": str}, ...]
        self.history: list[dict] = []

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    # ─────────────────────────────────────────────────────────
    # 모델 로딩
    # ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        지정 GPU에 모델과 토크나이저를 로드한다.
        4-bit의 경우 NF4 + double quantization + bfloat16 compute를 사용.
        """
        # Kanana(DeepseekV3ForCausalLM) 등 MoE 계열 모델을 로드할 때만 동작하는 안전장치.
        # transformers>=5.0이면 명확한 에러로 즉시 중단(뒤늦은 OOM 방지)하고, 4.x면 MoE forward의
        # dtype 버그를 런타임 패치한다. Qwen 등 비-MoE 모델에는 영향 없음 (rl/model_utils.py 참고).
        try:
            from rl.model_utils import assert_transformers_version, patch_deepseek_v3_moe_dtype_bug
            assert_transformers_version()
            patch_deepseek_v3_moe_dtype_bug()
        except ImportError:
            pass  # rl/ 모듈이 없는 배포 환경 등 — 이 안전장치 없이도 Qwen 등은 정상 동작

        # ── 양자화 설정 ──────────────────────────────────────
        if self.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,   # 2차 양자화로 추가 메모리 절약
            )
            load_kwargs = {
                "quantization_config": bnb_config,
                "device_map": {"": self.gpu_id},  # 단일 GPU 강제 고정
            }
        elif self.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs = {
                "quantization_config": bnb_config,
                "device_map": {"": self.gpu_id},
            }
        else:  # fp16
            load_kwargs = {
                "torch_dtype": torch.float16,
                "device_map": {"": self.gpu_id},
            }

        # ── 토크나이저 ───────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        # pad_token이 없는 모델(Llama 계열 등)을 위한 안전장치
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── 모델 ─────────────────────────────────────────────
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            **load_kwargs,
        )
        self.model.eval()

        if self.adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            self.model.eval()

    # ─────────────────────────────────────────────────────────
    # 텍스트 생성 내부 헬퍼
    # ─────────────────────────────────────────────────────────

    def _prepare_gen_kwargs(self, user_msg: str) -> tuple:
        """
        공통 토크나이징 + gen_kwargs 준비.
        user_msg를 히스토리에 추가하고, (streamer, gen_kwargs) 를 반환한다.
        generate / generate_iter 양쪽에서 공유.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                f"[{self.name}] 모델이 로드되지 않았습니다. load()를 먼저 호출하세요."
            )

        # 히스토리에 사용자 메시지 추가
        self.history.append({"role": "user", "content": user_msg})

        # 전체 메시지 구성: [system] + 대화 히스토리
        messages = [{"role": "system", "content": self.system_prompt}] + self.history

        # 채팅 템플릿 적용
        # transformers 5.x: BatchEncoding 반환 가능 / 4.x: 순수 LongTensor
        encoded = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        device = f"cuda:{self.gpu_id}"
        if hasattr(encoded, "input_ids"):
            input_ids      = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
        else:
            input_ids      = encoded.to(device)
            attention_mask = None

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs: dict = {
            "input_ids":          input_ids,
            "max_new_tokens":     self.gen_config.get("max_new_tokens", 600),
            "do_sample":          True,
            "temperature":        self.gen_config.get("temperature", 0.8),
            "top_p":              self.gen_config.get("top_p", 0.9),
            "repetition_penalty": self.gen_config.get("repetition_penalty", 1.05),
            "pad_token_id":       self.tokenizer.eos_token_id,
            "streamer":           streamer,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask

        return streamer, gen_kwargs

    # ─────────────────────────────────────────────────────────
    # 텍스트 생성 (스트리밍)
    # ─────────────────────────────────────────────────────────

    def generate(
        self,
        user_msg: str,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        터미널용: 응답을 생성하고 전체 문자열을 반환한다.
        stream_callback 이 주어지면 토큰 단위로 즉시 호출한다.
        """
        streamer, gen_kwargs = self._prepare_gen_kwargs(user_msg)

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        full_response = ""
        for token in streamer:
            full_response += token
            if stream_callback is not None:
                stream_callback(token)

        thread.join()

        response = full_response.strip()
        self.history.append({"role": "assistant", "content": response})
        return response

    def generate_iter(self, user_msg: str):
        """
        Streamlit용: 토큰을 하나씩 yield 하는 generator.
        st.write_stream(agent.generate_iter(msg)) 로 사용.

        generator 소진 후 자동으로 history 에 응답이 기록된다.
        """
        streamer, gen_kwargs = self._prepare_gen_kwargs(user_msg)

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        full_response = ""
        for token in streamer:
            full_response += token
            yield token

        thread.join()

        self.history.append({"role": "assistant", "content": full_response.strip()})

    # ─────────────────────────────────────────────────────────
    # 유틸리티
    # ─────────────────────────────────────────────────────────

    @property
    def last_response(self) -> str:
        """히스토리에서 가장 최근 assistant 발언을 반환. 없으면 빈 문자열."""
        for msg in reversed(self.history):
            if msg["role"] == "assistant":
                return msg["content"]
        return ""

    def reset_history(self) -> None:
        """대화 히스토리 초기화 (새 토론 주제 시작 시 사용)."""
        self.history = []

    def vram_usage_gb(self) -> float:
        """현재 이 Agent GPU의 VRAM 사용량(GB)을 반환."""
        if not torch.cuda.is_available():
            return 0.0
        used = torch.cuda.memory_allocated(self.gpu_id)
        return used / (1024 ** 3)

    def __repr__(self) -> str:
        loaded = "로드됨" if self.model is not None else "미로드"
        return (
            f"BaseAgent(name={self.name!r}, side={self.side!r}, "
            f"gpu={self.gpu_id}, quant={self.quantization}, {loaded})"
        )
