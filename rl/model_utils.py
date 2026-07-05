"""
rl/model_utils.py
──────────────────
Kanana-2-30B-A3B(DeepseekV3ForCausalLM, MLA + MoE)를 안전하게 로드하기 위한 공유 유틸리티.
rl/build_prompts.py, rl/train_grpo.py, rl/eval_grpo.py, agents/base_agent.py가 공통으로 쓴다.

Qwen(밀집 dense 아키텍처)과 달리 Kanana는:
  - attention이 MLA(q_proj + kv_a_proj_with_mqa/kv_b_proj)라 LoRA 대상 모듈 이름이 다르다.
  - MoE(128 routed + 2 shared experts)라 4bit 양자화가 routed expert에는 사실상 안 먹힌다
    (bitsandbytes가 fused 텐서를 개별 nn.Linear로 치환하지 못함) — 이 프로젝트에서 24GB 단일
    GPU 로딩을 직접 테스트해 확인됨(2026-07-04). 서버 GPU 용량에 따라 단일/멀티 GPU 배치를
    선택할 수 있도록 device_map을 설정으로 노출한다.
  - transformers의 DeepseekV3MoE.moe() 구현에 4bit+gradient checkpointing 조합에서 dtype
    불일치로 학습이 죽는 버그가 있어 런타임 몽키패치가 필요하다.
  - MoE 라우터(mlp.gate)에는 LoRA를 붙일 수 없다(peft ParamWrapper가 라우터 전용 속성을
    프록시하지 못해 AttributeError) — LoRA 대상에서 항상 제외한다.
"""

from __future__ import annotations

import re
from typing import Optional

import torch

REQUIRED_TRANSFORMERS_MAX = (5, 0, 0)  # 5.0 미만이어야 함


def assert_transformers_version() -> None:
    """transformers >= 5.0이면 명확한 에러로 즉시 중단시킨다 (OOM으로 늦게 실패하는 것 방지)."""
    import transformers

    parts = transformers.__version__.split(".")[:3]
    version = tuple(int(re.sub(r"\D", "", p) or 0) for p in parts)
    if version >= REQUIRED_TRANSFORMERS_MAX:
        raise RuntimeError(
            f"transformers=={transformers.__version__}는 Kanana(DeepseekV3ForCausalLM)에서 쓸 수 "
            "없습니다. transformers>=5.0은 MoE의 128개 routed expert를 fused nn.Parameter로 "
            "구현해 bitsandbytes 4bit 양자화가 안 먹히고 GPU 1장에 80GB+ 를 요구하며 OOM이 "
            '납니다. `pip install "transformers==4.57.6"` (5.0 미만 최신 4.x)로 낮추고 다시 '
            "실행하세요."
        )


# ────────────────────────────────────────────────────────────
# LoRA 대상 모듈 자동 탐색 (base model 아키텍처에 무관하게 동작)
# ────────────────────────────────────────────────────────────

# 밀집 attention(Qwen 등) + MLA attention(Kanana/DeepSeek-V3) + SwiGLU MLP(dense/shared/routed
# 공통) leaf 모듈 이름. 라우터("gate")는 의도적으로 제외 — DeepseekV3TopkRouter는 nn.Linear가
# 아니라 self.weight를 raw nn.Parameter로 들고 있어서 peft가 ParamWrapper로 감싸는데, 라우팅
# 로직이 참조하는 self.gate.e_score_correction_bias(로드밸런싱 보정값)를 ParamWrapper가
# 프록시하지 않아 forward에서 AttributeError로 죽는다. attention/MLP/shared_experts 대비
# 파라미터 비중도 작아 손해는 미미하다.
_LORA_LEAF_NAMES = {
    "q_proj", "k_proj", "v_proj", "o_proj",                      # 밀집 attention
    "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj",   # MLA attention
    "gate_proj", "up_proj", "down_proj",                         # SwiGLU MLP
}
_ROUTED_EXPERT_SEGMENT = re.compile(r"\.experts\.\d+\.")


def discover_lora_target_modules(model, include_routed_experts: bool = False) -> list[str]:
    """model.named_modules()를 순회해 실제 존재하는 LoRA 대상 모듈의 전체 경로를 찾는다.

    include_routed_experts=False(기본)면 128개 routed expert 내부 gate/up/down_proj는 제외하고
    attention + shared_experts(+ dense층 MLP)만 대상으로 삼는다 — 토큰마다 routed expert 몇 개만
    활성화되는 구조에서 128개 × 레이어 수 전체에 LoRA를 붙이면 모듈 수가 폭발적으로 늘어난다.
    """
    found: list[str] = []
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in _LORA_LEAF_NAMES:
            continue
        if not hasattr(module, "weight"):
            continue
        if _ROUTED_EXPERT_SEGMENT.search(name) and not include_routed_experts:
            continue
        found.append(name)

    if not found:
        raise RuntimeError(
            "LoRA 대상 모듈을 하나도 찾지 못했습니다. base model 구조가 예상과 다를 수 있습니다 — "
            "model.named_modules()를 직접 확인하고 discover_lora_target_modules()의 "
            "_LORA_LEAF_NAMES를 실제 모델에 맞게 수정하세요."
        )
    return sorted(found)


# ────────────────────────────────────────────────────────────
# DeepSeek-V3 MoE dtype 버그 런타임 패치
# ────────────────────────────────────────────────────────────

def patch_deepseek_v3_moe_dtype_bug() -> bool:
    """transformers의 DeepseekV3MoE.moe()가 final_hidden_states를 hidden_states.dtype이 아니라
    router 출력(topk_weights)의 dtype으로 만드는데, 4bit + gradient checkpointing 조합에서
    개별 expert 출력(expert_output * expert_weights)이 다른 dtype으로 나와
    `index_add_(): self와 source의 scalar type이 달라야 한다`는 RuntimeError로 죽는다.
    hidden_states 원래 dtype으로 고정하고 index_add_ 직전 명시적으로 dtype을 맞추는 버전으로
    런타임에 교체한다. 모델 로드 전에 호출해야 한다. DeepSeek-V3 계열이 아니면 조용히 스킵한다.

    Returns
    -------
    bool
        패치를 적용했으면 True, 모듈이 없어(비-DeepSeek-V3 모델) 스킵했으면 False.
    """
    try:
        from transformers.models.deepseek_v3 import modeling_deepseek_v3 as _dsv3
    except ImportError:
        return False

    def _fixed_moe(self, hidden_states, topk_indices, topk_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)
            if token_indices.numel() == 0:
                continue
            expert_weights = topk_weights[token_indices, weight_indices]
            expert_input = hidden_states[token_indices]
            expert_output = expert(expert_input)
            weighted_output = (expert_output * expert_weights.unsqueeze(-1)).to(final_hidden_states.dtype)
            final_hidden_states.index_add_(0, token_indices, weighted_output)

        return final_hidden_states.type(hidden_states.dtype)

    _dsv3.DeepseekV3MoE.moe = _fixed_moe
    print("  [패치] transformers DeepseekV3MoE.moe() dtype 버그 런타임 패치 적용됨")
    return True


# ────────────────────────────────────────────────────────────
# 통합 모델 로더
# ────────────────────────────────────────────────────────────

def resolve_device_map(gpus: list[int]) -> tuple[dict | str, Optional[dict]]:
    """GPU 목록으로부터 device_map/max_memory를 만든다.

    1장이면 단일 GPU 고정({"": gpu}), 여러 장이면 "auto" + 장당 max_memory 상한으로 분산한다
    (Kanana 인스턴스 하나가 24GB보다 훨씬 크게 필요할 수 있어 서버에 따라 여러 장에 걸칠 수 있음).
    """
    if len(gpus) == 1:
        return {"": gpus[0]}, None
    max_memory = {g: "1000GiB" for g in gpus}  # 실질 상한은 물리 VRAM이 알아서 제한
    max_memory["cpu"] = "0GiB"  # 디스크/CPU 오프로드는 극도로 느려 명시적으로 차단
    return "auto", max_memory


def load_policy_model(
    model_id: str,
    gpus: list[int],
    trust_remote_code: bool = True,
):
    """4bit NF4로 base 모델을 로드한다 (어댑터는 호출자가 별도로 부착).

    peft LoRA/GRPOTrainer용으로 쓸 때는 prepare_model_for_kbit_training()을 호출하지 말 것 —
    trl 1.5.1 GRPOTrainer가 peft_config를 받으면 내부에서 필요한 처리를 직접 한다.
    """
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    assert_transformers_version()
    patch_deepseek_v3_moe_dtype_bug()

    device_map, max_memory = resolve_device_map(gpus)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    load_kwargs = {
        "quantization_config": bnb_config,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
