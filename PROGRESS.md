# 작업 정리 (soobin 브랜치)

한국어 정치 토론 LLM 시스템의 구축 이력 정리. 진보(LEFT)·보수(RIGHT) 페르소나 두 에이전트가
사용자(사회자)의 질문을 놓고 토론하며, SFT → GRPO 강화학습으로 성향 일관성을 강화해온 과정을 기록한다.

---

## 1. 베이스 토론 시스템

- **모델**: Qwen2.5-14B-Instruct를 4-bit NF4 양자화로 2회 로드, GPU당 1인스턴스 고정
  (`device_map={"": gpu_id}` — RTX 3090 24GB × 2)
- **페르소나**:
  - LEFT — 이진영 교수 (42세, 서울대 사회학, Berkeley PhD, 진보 시민단체 공동대표)
  - RIGHT — 박민준 교수 (55세, 연세대 경제학, Chicago PhD, 시장경제연구원장)
- **인터페이스**:
  - `debate.py` — 터미널 스트리밍 UI (`l`/`r`/`q`/`vram`/`reset` 명령)
  - `app.py` — Streamlit 채팅 UI (CSS `:has()` + 숨김 마커로 좌/우/사회자 색상 구분,
    `st.write_stream` 토큰 스트리밍, `@st.cache_resource` 모델 캐시)
- **핵심 흐름**: `core/session.py`의 `_build_message()`가 상대의 직전 발언을 다음 에이전트의
  user 메시지에 주입 → LEFT 먼저, RIGHT가 LEFT의 방금 발언을 보고 응답
- 토론 로그는 `logs/debate_YYYYMMDD_HHMMSS.json`으로 저장

## 2. Phase 1 — SFT (완료, Qwen 기준)

**동기**: 논문 *"Systematic Biases in LLM Simulations of Debates"* — 시스템 프롬프트만으로는
토론이 길어질수록 에이전트가 중립으로 회귀(neutral drift). 자기 생성 데이터 QLoRA 파인튜닝으로
가중치 수준에서 성향을 고정.

- **질문**: `sft/questions.yaml` 100개 (13개 토픽 카테고리, 수동 작성 — LLM 자동 생성은 품질 문제로 폐기)
- **데이터**: 질문당 20회 독립 생성(temp 1.0) → 사이드당 2,000 examples
- **학습**: QLoRA r=64, alpha=128, 1 epoch, lr 2e-4, effective batch 32 → `adapters/left`, `adapters/right`
- **검증 설계**: 10라운드 토론 후 라운드별 0~10 스탠스 자가 평가(5 수렴 여부) + 중립 표현 블랙리스트 카운트

## 3. Phase 2 — GRPO 강화학습 파이프라인 (`rl/`)

SFT 이후에도 남는 중립 회귀·반박 품질·자기 반복 문제를 보상으로 직접 최적화.
상세 설계와 변경 이력은 `rl/PLAN.md` 참고.

**파이프라인**: `build_prompts.py` (오프라인 미니 토론 → 라운드별 프롬프트 슬라이싱) →
`train_grpo.py` (trl GRPOTrainer 1.5.1, QLoRA) → `eval_grpo.py` (스탠스 곡선 / 중립 표현 / before-after).
`judge.py`는 로컬 judge 모델이 stance(−1~+1)와 rebuttal(1~5)을 JSON 1회 호출로 배치 채점하고,
`run_all.sh` (tmux 순차 실행) + `monitor_metrics.py` (메트릭 CSV·경고 감시)가 실행을 보조한다.

### 3-1. 모델·보상 설계 (2026-07-04)

- **베이스 모델 교체**: kakaocorp/**kanana-2-30b-a3b-instruct** (DeepseekV3 아키텍처, MLA+MoE).
  jaeeun 브랜치의 Kanana 기준 SFT LoRA 어댑터(LEFT loss 2.05→0.17, RIGHT 2.03→0.14)를 도입,
  GRPO는 이 SFT 어댑터를 **이어받아** 학습 (`--init-adapter none`이면 새 LoRA)
- **`rl/model_utils.py`**: transformers 버전 가드 + MoE dtype 패치 + LoRA 대상 모듈 자동 탐색 + 멀티GPU 배치
- **보상 8종으로 고도화** (`rl/rewards.py`, 수빈 원안 6종 계보):
  | 보상 | 방식 |
  |---|---|
  | stance | judge 채점 (LEFT: −S, RIGHT: +S) |
  | rebuttal | judge, **pairwise Bradley-Terry** 기본 (arXiv:2605.28313) |
  | engagement | 상대 발언 키워드 커버리지 (규칙 기반, 신규) |
  | persona | 페르소나 정합도, 임베딩 기반 (arXiv:2511.00222, 신규) |
  | antirep | 자기 과거 발언과의 Jaccard 반복 패널티 |
  | semantic_echo | 의미 반복/동조 임베딩 패널티 (ko-sroberta, 신규) |
  | neutral_phrase | 중립 표현 블랙리스트 패널티 |
  | format | 길이/미종결/목록 형식 패널티 |
  - korean_ratio는 삭제 (Kanana는 한국어 네이티브)
  - judge 불확실성 라우팅(API 재채점, arXiv:2510.20369)은 `rl/config.yaml judge_routing.enabled`로 opt-in (기본 off)
  - 실검증: pairwise BT가 정면반박 1.00 > 논점회피 0.01 > 구호나열 0.00 판별, persona 정합 0.655 > 이탈 0.301

### 3-2. GRPO 학습 완료 (2026-07-04, vast.ai RTX PRO 6000 96GB)

- LEFT/RIGHT 각 **55스텝** 완료 (스텝당 ~200초, VRAM 피크 32GB, 정책+judge 동일 GPU)
- 데이터: 50문항 트랜스크립트 → 사이드당 150 프롬프트 (Kanana 토크나이저, max 1,019토큰) → `rl/data_kanana/`
- **결과**: 반복 패널티(antirep/semantic_echo) 80%+ 감소, 통합 보상 상승(RIGHT 0.92→1.05),
  stance는 초기부터 높게 유지(SFT 초기화 효과), judge 파싱 실패 0~3%, KL ~0.02
- 산출 어댑터: `adapters/left_grpo`, `adapters/right_grpo` (각 172MB, KL ref 사본 포함 — 용량상 git 제외)

## 4. 알려진 제약 / 남은 과제

- **transformers==4.57.6 고정 필수**: 5.x는 Kanana MoE expert를 fused Parameter로 구현해
  bitsandbytes 4bit 양자화가 안 걸림 → bf16 그대로 83~94GB 로드되어 OOM.
  `rl/model_utils.py`가 버전 가드
- **이 서버(3090 24GB×2)로는 Kanana 학습 불가** — 실행은 대용량 GPU 서버에서.
  GPU 배치는 `rl/config.yaml`의 `left_gpus/right_gpus/judge_gpu`
- `rl/data/*_prompts.jsonl`은 Qwen 토크나이저 산출물 — Kanana 기준 재생성 필요 (data_kanana가 최신)
- **eval_grpo 미실행** (예산상 보류) — GRPO 어댑터 정량 평가가 다음 과제
- 로컬 추론(`debate.py`/`app.py`)의 Kanana 4bit 동작은 미검증 (4bit 가중치 ~16GB로 24GB 1장에 들어갈 여지)

## 5. 파일 구조 (이 브랜치 기준 주요 변경)

```
rl/                     # GRPO 파이프라인 전체 (신규)
├── PLAN.md             # 설계 문서 + 변경 이력
├── build_prompts.py    # 오프라인 미니 토론 → 프롬프트 JSONL
├── judge.py            # 로컬 judge (stance/rebuttal 배치 채점)
├── rewards.py          # 보상 8종 팩토리
├── train_grpo.py       # GRPO 학습 (--side left|right, SFT 어댑터 이어받기 기본)
├── eval_grpo.py        # 3종 평가 (스탠스 곡선 / 중립 표현 / before-after)
├── model_utils.py      # Kanana 로딩: 버전 가드·MoE 패치·LoRA 타겟 탐색·멀티GPU
├── monitor_metrics.py  # 학습 메트릭 감시
├── run_all.sh          # tmux 전체 파이프라인
├── config.yaml         # RL 하이퍼파라미터 + 보상 가중치 + GPU 배치
└── data_kanana/        # Kanana 기준 프롬프트/트랜스크립트 (학습에 실사용)

agents/base_agent.py    # transformers 4.x/5.x chat_template 호환 처리
config/model.yaml       # Kanana 전환, SFT 어댑터 경로(adapters/left·right) 활성화
CLAUDE.md               # GRPO 파이프라인 실행법 반영
```
