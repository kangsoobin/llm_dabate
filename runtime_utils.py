"""
runtime_utils.py
─────────────────
sft/train.py와 rl/train_grpo.py가 공유하는 GPU 서버 런타임 유틸.

2026-07-03 SFT를 실제 GPU 서버에서 돌리면서 발견한 두 가지 문제의 해결책을 모아둔 것
(상세: HANDOFF.md §4-7, §4-9). GRPO도 같은 모델·같은 Trainer 계열을 쓰므로 동일하게 필요하다.
"""

from __future__ import annotations

import os


def restrict_visible_gpus(gpus: list[int]) -> list[int]:
    """
    이 프로세스에 지정한 물리 GPU만 보이도록 CUDA_VISIBLE_DEVICES를 설정하고,
    이후 코드에서 쓸 로컬 GPU 인덱스(0..N-1)를 반환한다.

    공용 서버에서 다른 유저가 GPU를 쓰고 있을 때, transformers Trainer는
    torch.cuda.device_count()로 "보이는" GPU 수를 세서 nn.DataParallel로 자동
    래핑한다 — 이러면 우리가 고르지 않은(=다른 사람이 쓰는) GPU까지 건드려
    그쪽에서 OOM이 난다 (HANDOFF.md §4-9, 실제 발생). 반드시 torch가 CUDA를
    초기화하기 전에 호출할 것.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    print(f"[물리 GPU {gpus}만 이 프로세스에 보이도록 제한 (CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']})]")
    return list(range(len(gpus)))


def patch_deepseek_v3_moe_dtype_bug() -> None:
    """
    transformers 4.57.6의 DeepseekV3MoE.moe() dtype 버그 런타임 몽키패치.

    원본은 `final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)`로
    dtype을 hidden_states가 아니라 router 출력(topk_weights)의 dtype에 맞춰 만든다. 그런데
    개별 expert 출력(expert_output * expert_weights)이 다른 dtype으로 나올 수 있어(4bit 양자화 +
    gradient checkpointing 조합에서 실제 발생) `index_add_(): self (BFloat16) and source (Float)
    must have the same scalar type` RuntimeError로 학습 첫 스텝에서 죽는다. 해당 함수 자체에
    "CALL FOR CONTRIBUTION! I don't have time to optimise this right now"라는 주석이 달려 있을
    만큼 다듬어지지 않은 부분 — index_add_ 직전에 명시적으로 dtype을 맞춰주는 버전으로 교체한다.
    site-packages 파일을 직접 고치면 재설치 시 사라지므로 모델 로드 전에 여기서 패치한다.

    bf16(비양자화) 로드에서는 이 버그가 발현되지 않을 수 있지만 패치는 무해하므로 항상 적용한다.
    """
    import torch

    try:
        from transformers.models.deepseek_v3 import modeling_deepseek_v3 as _dsv3
    except ImportError:
        return
    if not hasattr(_dsv3, "DeepseekV3MoE") or not hasattr(_dsv3.DeepseekV3MoE, "moe"):
        # transformers 5.x는 MoE 구현이 완전히 달라(fused nn.Parameter) 이 패치 대상이 아님.
        # 애초에 5.x는 이 프로젝트와 호환되지 않는다 (HANDOFF.md §4-3).
        print("  [경고] DeepseekV3MoE.moe()가 없습니다 — transformers 버전이 4.57.6인지 확인하세요.")
        return

    def _fixed_moe(self, hidden_states, topk_indices, topk_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = (expert_output * expert_weights.unsqueeze(-1)).to(final_hidden_states.dtype)
                final_hidden_states.index_add_(0, token_indices, weighted_output)

        return final_hidden_states.type(hidden_states.dtype)

    _dsv3.DeepseekV3MoE.moe = _fixed_moe
    print("  [패치] transformers DeepseekV3MoE.moe() dtype 버그 런타임 패치 적용됨")
