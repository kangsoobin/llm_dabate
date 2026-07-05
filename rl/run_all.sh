#!/usr/bin/env bash
# rl/run_all.sh — GRPO 전체 파이프라인 순차 실행
# 사용법: tmux new -s grpo 'bash rl/run_all.sh'   (총 ~28-32시간)
# 중단 시 재실행하면: build_prompts는 이어서 진행, 학습은 --resume을 수동으로 붙일 것.
set -eo pipefail
cd "$(dirname "$0")/.."

# conda 스크립트는 unbound variable을 참조하므로 set -u는 activate 이후에 켠다
source ~/anaconda3/etc/profile.d/conda.sh
conda activate debate
set -u

mkdir -p logs
LOG="logs/rl_run_all_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "로그: $LOG"

echo
echo "════ [1/8] 프롬프트 데이터셋 생성 (~4-6h) ════"
python rl/build_prompts.py

echo
echo "════ [2/8] LEFT 3스텝 스모크 — VRAM/Judge 확인 (~15m) ════"
python rl/train_grpo.py --side left --steps 3
rm -rf adapters/left_grpo_ckpt adapters/left_grpo   # 스모크 산출물 제거 후 본 학습

echo
echo "════ [3/8] LEFT 본 학습 (~10-12h) ════"
python rl/train_grpo.py --side left

echo
echo "════ [4/8] RIGHT 본 학습 (~10-12h) ════"
python rl/train_grpo.py --side right

echo
echo "════ [5/8] LEFT GRPO 평가 ════"
python rl/eval_grpo.py --side left --adapter adapters/left_grpo

echo
echo "════ [6/8] LEFT base 비교 평가 ════"
python rl/eval_grpo.py --side left --adapter none --tag base

echo
echo "════ [7/8] RIGHT GRPO 평가 ════"
python rl/eval_grpo.py --side right --adapter adapters/right_grpo

echo
echo "════ [8/8] RIGHT base 비교 평가 ════"
python rl/eval_grpo.py --side right --adapter none --tag base

echo
echo "════ 전체 완료 ════"
echo "배포하려면 config/model.yaml 에서:"
echo '  left_adapter:  "adapters/left_grpo"'
echo '  right_adapter: "adapters/right_grpo"'
