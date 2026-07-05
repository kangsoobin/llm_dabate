"""
agents/right_agent.py
─────────────────────
RIGHT Agent 팩토리.
GPU 1에 김보수(국민의힘) 페르소나를 가진 BaseAgent 인스턴스를 생성한다.
"""

from .base_agent import BaseAgent


def create_right_agent(
    model_cfg: dict,
    gen_cfg: dict,
    system_prompt: str,
    adapter_path: str | None = None,
) -> BaseAgent:
    """
    Parameters
    ----------
    model_cfg : dict
        config/model.yaml에서 로드한 모델 설정
        (model_id, right_gpu, quantization)
    gen_cfg : dict
        생성 파라미터 (max_new_tokens, temperature, top_p, repetition_penalty)
    system_prompt : str
        config/prompts.yaml의 right 프롬프트

    Returns
    -------
    BaseAgent
        GPU 1에 할당된 RIGHT Agent (아직 load() 호출 전)
    """
    return BaseAgent(
        model_id=model_cfg["model_id"],
        gpu_id=model_cfg["right_gpu"],
        system_prompt=system_prompt,
        generation_config=gen_cfg,
        quantization=model_cfg["quantization"],
        name="김보수",
        side="right",
        adapter_path=adapter_path,
    )
