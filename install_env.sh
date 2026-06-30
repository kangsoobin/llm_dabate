#!/usr/bin/env bash
# ============================================================
#  LLM Debate 시스템 — conda 환경 설치 스크립트
#  대상 실행안: B (Qwen2.5-14B-Instruct × 2, 4-bit quantization)
#  CUDA: 12.4  /  Driver: 550.x
# ============================================================
set -e   # 에러 시 즉시 중단

ENV_NAME="debate"
PYTHON_VER="3.11"

# ── 컬러 헬퍼 ───────────────────────────────────────────────
GREEN="\033[92m"; YELLOW="\033[93m"; RED="\033[91m"
CYAN="\033[96m";  BOLD="\033[1m";    RESET="\033[0m"
ok()   { echo -e "  ${GREEN}✔${RESET}  $*"; }
info() { echo -e "  ${CYAN}→${RESET}  $*"; }
warn() { echo -e "  ${YELLOW}⚠${RESET}  $*"; }
step() { echo -e "\n${BOLD}${CYAN}══ $* ══${RESET}"; }

# ── conda 위치 확인 ─────────────────────────────────────────
step "conda 확인"
if ! command -v conda &>/dev/null; then
    # anaconda3 기본 경로 시도
    CONDA_PATHS=(
        "$HOME/anaconda3/bin/conda"
        "$HOME/miniconda3/bin/conda"
        "/opt/anaconda3/bin/conda"
        "/opt/miniconda3/bin/conda"
    )
    for p in "${CONDA_PATHS[@]}"; do
        if [ -f "$p" ]; then
            export PATH="$(dirname $p):$PATH"
            break
        fi
    done
fi

if ! command -v conda &>/dev/null; then
    echo -e "  ${RED}✘  conda 명령어를 찾을 수 없습니다.${RESET}"
    echo -e "  ${YELLOW}→  아래 명령으로 conda를 먼저 활성화하세요:${RESET}"
    echo -e "     source ~/anaconda3/etc/profile.d/conda.sh"
    exit 1
fi

ok "conda 확인: $(conda --version)"

# ── conda 초기화 (activate 사용 가능하게) ───────────────────
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# ── 기존 환경 확인 ──────────────────────────────────────────
step "conda 환경 확인 / 생성"
if conda env list | grep -q "^${ENV_NAME} "; then
    warn "환경 '${ENV_NAME}' 이미 존재 → 재사용합니다."
    warn "처음부터 새로 만들려면:  conda env remove -n ${ENV_NAME}  후 재실행"
else
    info "새 환경 생성 중: ${ENV_NAME}  (Python ${PYTHON_VER})"
    conda create -n "${ENV_NAME}" python="${PYTHON_VER}" -y
    ok "환경 생성 완료"
fi

conda activate "${ENV_NAME}"
ok "환경 활성화: $(python --version)  |  $(which python)"

# ── pip 업그레이드 ──────────────────────────────────────────
step "pip 업그레이드"
pip install --upgrade pip -q
ok "pip $(pip --version | awk '{print $2}')"

# ── PyTorch (CUDA 12.4) ─────────────────────────────────────
step "PyTorch + CUDA 12.4 설치"
info "약 2–5 GB 다운로드, 네트워크 속도에 따라 수 분 소요..."
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124 \
    --progress-bar on
ok "PyTorch 설치 완료"

# CUDA 인식 확인
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA 인식 실패!"
print(f"  ✔  torch {torch.__version__}  |  CUDA {torch.version.cuda}  |  GPU×{torch.cuda.device_count()}")
EOF

# ── HuggingFace 핵심 라이브러리 ─────────────────────────────
step "transformers / accelerate / bitsandbytes 설치"
pip install \
    "transformers>=4.45.0" \
    "accelerate>=0.34.0" \
    "bitsandbytes>=0.44.0" \
    sentencepiece \
    protobuf \
    huggingface_hub \
    -q --progress-bar on
ok "HuggingFace 패키지 설치 완료"

# ── bitsandbytes CUDA 연결 확인 ─────────────────────────────
step "bitsandbytes CUDA 연결 확인"
python - <<'EOF'
import bitsandbytes as bnb
print(f"  ✔  bitsandbytes {bnb.__version__}")
EOF

# ── 설치 검증 (check_server.py 재실행) ──────────────────────
step "설치 검증 — check_server.py 재실행"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "${SCRIPT_DIR}/check_server.py"

# ── 최종 안내 ───────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  설치 완료!${RESET}"
echo -e "${GREEN}  다음 단계: conda activate ${ENV_NAME}${RESET}"
echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════${RESET}"
echo ""
echo -e "  이후 토론 시스템 실행 방법:"
echo -e "    conda activate ${ENV_NAME}"
echo -e "    python debate.py          # (아직 미생성, 다음 단계)"
echo ""
