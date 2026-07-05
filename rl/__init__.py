"""
rl/
───
GRPO 기반 RL 학습 모듈.

- rewards/    : 보상 컴포넌트 (v1 = 보상 설계.pdf 원안, v2 = docs/reward_design_v2.md 재설계)
- simulate.py : 멀티라운드 self-play 트랜스크립트 생성 (GPU 필요)
- rollout.py  : 트랜스크립트 -> GRPOTrainer용 datasets.Dataset 변환
- train_grpo.py : GRPO 학습 진입점

설계 배경은 docs/reward_design_v2.md 참고.
"""
