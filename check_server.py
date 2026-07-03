#!/usr/bin/env python3
"""
서버 사양 점검 스크립트
LLM Debate 시스템 실행 가능 여부 판단용
"""

import sys
import os
import shutil
import subprocess

# ────────────────────────────────────────────────────────────
# 컬러 출력 헬퍼
# ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):    print(f"  {GREEN}✔{RESET}  {msg}")
def warn(msg):  print(f"  {YELLOW}⚠{RESET}  {msg}")
def fail(msg):  print(f"  {RED}✘{RESET}  {msg}")
def info(msg):  print(f"  {CYAN}→{RESET}  {msg}")
def header(msg):print(f"\n{BOLD}{CYAN}{'─'*60}{RESET}");\
                print(f"{BOLD}{CYAN}  {msg}{RESET}");\
                print(f"{BOLD}{CYAN}{'─'*60}{RESET}")

# ────────────────────────────────────────────────────────────
# 1. Python 버전
# ────────────────────────────────────────────────────────────
header("1. Python 버전")
pv = sys.version_info
info(f"Python {pv.major}.{pv.minor}.{pv.micro}  ({sys.executable})")
if pv.major == 3 and pv.minor >= 9:
    ok("Python 3.9+ 확인")
else:
    warn(f"Python 3.9 미만 — 일부 라이브러리 호환 문제 가능성")

# ────────────────────────────────────────────────────────────
# 2. pip 주요 패키지 설치 여부
# ────────────────────────────────────────────────────────────
header("2. pip 패키지 설치 여부")
packages = ["torch", "transformers", "accelerate", "bitsandbytes",
            "sentencepiece", "protobuf"]
pkg_status = {}
pkg_versions = {}
for pkg in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "version unknown")
        ok(f"{pkg:<20} {ver}")
        pkg_status[pkg] = True
        pkg_versions[pkg] = ver
    except ImportError:
        fail(f"{pkg:<20} 미설치")
        pkg_status[pkg] = False
        pkg_versions[pkg] = None

# Kanana-2-30B-A3B(DeepseekV3ForCausalLM 아키텍처)는 transformers>=4.51.0 필요.
# 구버전이면 model_type "deepseek_v3"를 인식하지 못해 로딩 자체가 실패한다.
_tv = pkg_versions.get("transformers")
if _tv:
    try:
        _tv_tuple = tuple(int(x) for x in _tv.split(".")[:2])
        transformers_ok_for_kanana = _tv_tuple >= (4, 51)
    except ValueError:
        transformers_ok_for_kanana = False
    if transformers_ok_for_kanana:
        ok(f"transformers {_tv} — Kanana-2-30B-A3B(요구: >=4.51.0) 로딩 가능")
    else:
        warn(f"transformers {_tv} — Kanana-2-30B-A3B는 >=4.51.0 필요, 업그레이드하세요 "
             f"(pip install -U \"transformers>=4.51.0\")")
else:
    transformers_ok_for_kanana = False

# ────────────────────────────────────────────────────────────
# 3. CUDA / NVIDIA 드라이버 / CUDA 버전
# ────────────────────────────────────────────────────────────
header("3. CUDA / NVIDIA 드라이버 정보")

# nvidia-smi 로 드라이버 버전
try:
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        stderr=subprocess.DEVNULL
    ).decode().strip().split("\n")
    driver_ver = smi[0]
    ok(f"NVIDIA 드라이버 버전: {driver_ver}")
except Exception:
    fail("nvidia-smi 실행 실패 — NVIDIA 드라이버 미설치 또는 PATH 문제")
    driver_ver = None

# PyTorch CUDA 인식
cuda_available = False
cuda_version   = "N/A"
torch_version  = "N/A"
if pkg_status.get("torch"):
    import torch
    torch_version  = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_version   = torch.version.cuda or "N/A"
    info(f"PyTorch 버전      : {torch_version}")
    info(f"PyTorch CUDA 버전 : {cuda_version}")
    if cuda_available:
        ok("torch.cuda.is_available() = True")
    else:
        fail("torch.cuda.is_available() = False  ← GPU 사용 불가")
else:
    fail("torch 미설치 → CUDA 인식 불가")

# ────────────────────────────────────────────────────────────
# 4. GPU 상세 정보
# ────────────────────────────────────────────────────────────
header("4. GPU 상세 정보")

gpu_info = []   # list of dict per GPU

if cuda_available:
    import torch
    gpu_count = torch.cuda.device_count()
    info(f"감지된 GPU 수: {gpu_count}")
    print()

    for i in range(gpu_count):
        props      = torch.cuda.get_device_properties(i)
        total_vram = props.total_memory / (1024**3)          # GB
        free_vram  = torch.cuda.mem_get_info(i)[0] / (1024**3)
        used_vram  = total_vram - free_vram

        gpu_info.append({
            "index":      i,
            "name":       props.name,
            "total_vram": total_vram,
            "used_vram":  used_vram,
            "free_vram":  free_vram,
        })

        print(f"  {BOLD}GPU {i}: {props.name}{RESET}")
        info(f"  총  VRAM : {total_vram:.1f} GB")
        info(f"  사용 중  : {used_vram:.1f} GB")
        info(f"  남은 VRAM: {free_vram:.1f} GB")

        if free_vram >= 20:
            ok(f"  여유 VRAM 충분 (≥20 GB)")
        elif free_vram >= 14:
            warn(f"  여유 VRAM 보통 (14–20 GB)")
        elif free_vram >= 8:
            warn(f"  여유 VRAM 부족 (8–14 GB) — 4-bit 양자화 필요")
        else:
            fail(f"  여유 VRAM 심각하게 부족 (<8 GB)")
        print()
else:
    # nvidia-smi fallback
    try:
        raw = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip().split("\n")
        for line in raw:
            parts = [p.strip() for p in line.split(",")]
            idx, name, tot, used, free = parts
            tot, used, free = float(tot)/1024, float(used)/1024, float(free)/1024
            gpu_info.append({"index": int(idx), "name": name,
                             "total_vram": tot, "used_vram": used, "free_vram": free})
            print(f"  {BOLD}GPU {idx}: {name}{RESET}")
            info(f"  총 VRAM : {tot:.1f} GB  /  사용: {used:.1f} GB  /  여유: {free:.1f} GB")
        warn("PyTorch CUDA 불가 → nvidia-smi 값으로 표시 (실제 추론 불가)")
    except Exception:
        fail("GPU 정보를 전혀 가져올 수 없음")

# ────────────────────────────────────────────────────────────
# 5. CPU RAM
# ────────────────────────────────────────────────────────────
header("5. CPU RAM")
try:
    import psutil
    vm = psutil.virtual_memory()
    total_gb = vm.total / (1024**3)
    avail_gb = vm.available / (1024**3)
    used_pct = vm.percent
    info(f"총  RAM : {total_gb:.1f} GB")
    info(f"남은 RAM: {avail_gb:.1f} GB  (사용률 {used_pct}%)")
    if avail_gb >= 32:
        ok("RAM 여유 충분")
    elif avail_gb >= 16:
        warn("RAM 여유 보통 — 대용량 모델 CPU offload 시 주의")
    else:
        warn("RAM 여유 부족 — CPU offload 어려움")
    ram_total = total_gb
    ram_avail = avail_gb
except ImportError:
    warn("psutil 미설치 → /proc/meminfo 로 대체")
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem = {}
        for l in lines:
            k, v = l.split(":")[0].strip(), l.split(":")[1].strip()
            mem[k] = v
        info(f"MemTotal    : {mem.get('MemTotal','?')}")
        info(f"MemAvailable: {mem.get('MemAvailable','?')}")

        def kb_to_gb(s):
            return int(s.replace("kB","").strip()) / (1024**2)

        ram_total = kb_to_gb(mem.get("MemTotal","0 kB"))
        ram_avail = kb_to_gb(mem.get("MemAvailable","0 kB"))
        info(f"총 RAM: {ram_total:.1f} GB  /  가용: {ram_avail:.1f} GB")
    except Exception as e:
        fail(f"RAM 정보 읽기 실패: {e}")
        ram_total = 0
        ram_avail = 0

# ────────────────────────────────────────────────────────────
# 6. 디스크 용량
# ────────────────────────────────────────────────────────────
header("6. 디스크 용량 (현재 디렉토리 기준)")
cwd = os.getcwd()
try:
    disk = shutil.disk_usage(cwd)
    total_disk = disk.total / (1024**3)
    free_disk  = disk.free  / (1024**3)
    used_disk  = disk.used  / (1024**3)
    info(f"경로       : {cwd}")
    info(f"총 용량    : {total_disk:.1f} GB")
    info(f"사용 중    : {used_disk:.1f} GB")
    info(f"남은 용량  : {free_disk:.1f} GB")
    if free_disk >= 100:
        ok("디스크 여유 충분 (≥100 GB)")
    elif free_disk >= 50:
        warn(f"디스크 여유 보통 ({free_disk:.0f} GB) — 7B 모델 1~2개 수준")
    elif free_disk >= 20:
        warn(f"디스크 여유 부족 ({free_disk:.0f} GB) — 작은 모델만 가능")
    else:
        fail(f"디스크 여유 심각하게 부족 ({free_disk:.0f} GB)")
except Exception as e:
    fail(f"디스크 정보 읽기 실패: {e}")
    free_disk = 0

# ────────────────────────────────────────────────────────────
# 7. 실행안 가능 여부 판단
# ────────────────────────────────────────────────────────────
header("7. 실행안 가능 여부 판단")

MIN_PKGS = ["torch", "transformers", "accelerate"]
base_ok  = cuda_available and all(pkg_status.get(p) for p in MIN_PKGS)

# GPU 별 여유 VRAM 리스트
free_vrams = [g["free_vram"] for g in gpu_info] if gpu_info else []
num_gpus   = len(free_vrams)

# 모델별 대략적 VRAM 요구량 (GB)
# Qwen2.5-7B  BF16  ≈ 15 GB  (실제 ~14–16 GB)
# Qwen2.5-14B 4bit  ≈ 9 GB   (실제 ~8–10 GB)
# Llama-3.1-8B FP16 ≈ 16 GB  (실제 ~15–17 GB)
# Llama-3.1-8B 8bit ≈ 9 GB
# Kanana-2-30B-A3B 4bit ≈ 17–18 GB (모델 로딩 기준. 총 30B 파라미터, MoE라 활성 파라미터는 3B지만
#   VRAM은 "로딩된 전체 가중치" 기준이라 활성 파라미터 수와 무관하게 30B 전체가 잡힘.
#   QLoRA 학습(옵티마이저 상태 + 활성화)은 이보다 더 필요 — 아래 REQ_D는 inference 기준 하한선.)

REQ_A = 15.0   # Qwen2.5-7B BF16 per GPU
REQ_B = 9.0    # Qwen2.5-14B 4bit per GPU
REQ_C_fp16 = 16.0   # Llama-3.1-8B FP16
REQ_C_8bit = 9.0    # Llama-3.1-8B 8bit
REQ_D = 18.0   # Kanana-2-30B-A3B 4bit per GPU (inference 기준 — SFT/QLoRA 학습 시 더 필요)
DISK_7B  = 15   # GB (모델 파일)
DISK_14B = 29   # GB
DISK_8B  = 16   # GB
DISK_KANANA = 65   # GB (30B, bf16 원본 체크포인트 다운로드 기준 — 4bit는 로드 시점에 변환)

def check_plan(label, model_name, req_vram, disk_req, extra_pkg=None):
    print(f"\n  {BOLD}▶ 실행안 {label}: {model_name}{RESET}")
    issues = []

    # 기본 패키지
    if not base_ok:
        issues.append("torch/transformers/accelerate 미설치 또는 CUDA 불가")

    # 추가 패키지
    if extra_pkg:
        for p in extra_pkg:
            if not pkg_status.get(p):
                issues.append(f"{p} 미설치")

    # GPU 수
    if num_gpus < 2:
        issues.append(f"GPU가 {num_gpus}개뿐 (2개 필요)")
    else:
        # 각 GPU에 req_vram 이상 여유 있는지
        for i, fv in enumerate(free_vrams[:2]):
            if fv < req_vram:
                issues.append(f"GPU {i} 여유 VRAM {fv:.1f}GB < 필요 {req_vram:.0f}GB")

    # 디스크
    if free_disk < disk_req:
        issues.append(f"디스크 여유 {free_disk:.0f}GB < 모델 필요 {disk_req}GB")

    if not issues:
        ok(f"실행 가능  (VRAM 요구: 각 GPU ≥{req_vram:.0f}GB / 디스크 ≥{disk_req}GB)")
        return True
    else:
        for iss in issues:
            fail(iss)
        return False

feasible = {}
feasible["A"] = check_plan(
    "A", "Qwen2.5-7B-Instruct × 2  (BF16, 각 GPU 1개)",
    REQ_A, DISK_7B * 2
)
feasible["B"] = check_plan(
    "B", "Qwen2.5-14B-Instruct × 2  (4-bit, 각 GPU 1개)",
    REQ_B, DISK_14B * 2,
    extra_pkg=["bitsandbytes"]
)

# Plan C: FP16 먼저, 안되면 8bit 제안
plan_c_fp16 = check_plan(
    "C (FP16)", "Llama-3.1-8B-Instruct × 2  (FP16, 각 GPU 1개)",
    REQ_C_fp16, DISK_8B * 2
)
plan_c_8bit = check_plan(
    "C (8-bit)", "Llama-3.1-8B-Instruct × 2  (8-bit, 각 GPU 1개)",
    REQ_C_8bit, DISK_8B * 2,
    extra_pkg=["bitsandbytes"]
)
feasible["C"] = plan_c_fp16 or plan_c_8bit

feasible["D"] = check_plan(
    "D (현재 프로젝트 설정)", "Kanana-2-30B-A3B-Instruct × 2  (4-bit, 각 GPU 1개)",
    REQ_D, DISK_KANANA,
    extra_pkg=["bitsandbytes"]
)
if feasible["D"] and not transformers_ok_for_kanana:
    fail("  transformers 버전이 낮아 Kanana(deepseek_v3) 로딩 불가 — pip install -U \"transformers>=4.51.0\"")
    feasible["D"] = False

# ────────────────────────────────────────────────────────────
# 8. 최종 요약
# ────────────────────────────────────────────────────────────
header("8. 최종 요약 및 권장 실행안")

print(f"\n  {'항목':<25} {'상태'}")
print(f"  {'─'*45}")
print(f"  {'GPU 수':<25} {num_gpus}개")
for g in gpu_info:
    print(f"  {'GPU '+str(g['index'])+' ('+g['name']+')':<25} "
          f"여유 {g['free_vram']:.1f}/{g['total_vram']:.1f} GB")
print(f"  {'CUDA 사용 가능':<25} {'✔' if cuda_available else '✘'}")
print(f"  {'RAM 가용':<25} {ram_avail:.1f} GB")
print(f"  {'디스크 여유':<25} {free_disk:.1f} GB")
print()

recommended = [k for k, v in feasible.items() if v]
if recommended:
    print(f"  {GREEN}{BOLD}✔ 실행 가능한 실행안: {', '.join(recommended)}{RESET}")
    # D(Kanana-2-30B-A3B, 현재 프로젝트가 실제로 쓰는 설정)를 우선 권장. 미가능하면 다른 안 중 첫 번째.
    best = "D" if "D" in recommended else recommended[0]
    labels = {
        "A": "Qwen2.5-7B BF16",
        "B": "Qwen2.5-14B 4bit",
        "C": "Llama-3.1-8B",
        "D": "Kanana-2-30B-A3B 4bit (현재 config/model.yaml 설정)",
    }
    print(f"  {GREEN}권장 실행안: {best}  ({labels[best]}){RESET}")
    if best != "D":
        warn("현재 프로젝트(config/model.yaml)는 Kanana-2-30B-A3B를 쓰도록 설정되어 있습니다 — "
             "실행안 D가 불가능하면 위 D 항목의 실패 사유를 먼저 해결하세요.")
else:
    print(f"  {RED}{BOLD}✘ 현재 서버 환경에서는 어떤 실행안도 그대로 실행 불가{RESET}")
    print(f"  {YELLOW}→ VRAM/패키지/디스크 부족 항목을 위 결과에서 확인하세요.{RESET}")

print()
print(f"  {CYAN}※ VRAM 요구량은 추정치입니다. 실제 로딩 시 ±1~2 GB 차이 발생 가능.{RESET}")
print(f"  {CYAN}※ 이미 다른 프로세스가 GPU를 점유 중이면 여유 VRAM이 줄어듭니다.{RESET}")
print()
