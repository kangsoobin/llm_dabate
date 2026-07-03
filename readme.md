conda activate debate

python debate.py

streamlit run app.py --server.address 0.0.0.0 --server.port 8501

최저임금 올려야 하나?	LEFT → RIGHT 순서로 양쪽 응답
l 재원은 어떻게 마련하죠?	LEFT(이진영 교수)에게만
r 낙수효과가 없다는 증거는?	RIGHT(박민준 교수)에게만


UI 동작 방식
항목	구현
색상	CSS :has() + 숨김 마커 span으로 LEFT=파란, RIGHT=빨간, 사회자=회색
스트리밍	st.write_stream(agent.generate_iter(...)) — 토큰 단위 실시간 출력
히스토리	st.session_state.messages 에 저장, 새로고침해도 유지
상대 발언 주입	기존 session._build_message() 그대로 재사용
모델 캐시	@st.cache_resource — 앱 수명 동안 GPU 메모리에 1회만 로드
새로운 주제	🔄 버튼 → session_state + agent history 모두 초기화


Phase 1 — SFT 파이프라인

> **2026-07-02 모델 교체:** base model을 Qwen2.5-14B-Instruct → **Kanana-2-30B-A3B-Instruct**
> (카카오, 최신 한국어 모델, `transformers>=4.51.0` 필요)로 변경. 이전에 학습해둔 Qwen 기반
> `adapters/left`, `adapters/right`는 아키텍처가 달라(DeepseekV3ForCausalLM, MLA+MoE) 새 모델에
> 로드할 수 없으므로 **SFT를 처음부터 다시 해야 함**. 다만 `sft/data/{left,right}_train.jsonl`
> (팀원 공유분, 페르소나 프롬프트+응답 텍스트)은 모델과 무관해 그대로 재사용 — `generate_data.py`를
> 다시 돌릴 필요 없이 `sft/train.py`만 재실행하면 됨. `sft/train.py`의 LoRA `target_modules`는
> 하드코딩 대신 로드된 모델을 순회해 자동 탐색하도록 바뀌었다 (MoE 모듈 이름이 Qwen과 다름).

sft/questions.yaml: 시드 질문 10개 (경제·노동·환경·안보·사회)
sft/generate_data.py: 질문당 20회 응답 생성 → JSONL 저장
sft/train.py: QLoRA + SFTTrainer 1 epoch → adapters/{side}/ 저장 (LoRA 대상 모듈 자동 탐색)
agents/base_agent.py: adapter_path 파라미터 추가, load() 끝에 PeftModel 조건부 적용
config/model.yaml: left_adapter / right_adapter 항목 추가 (Kanana 재학습 전까지 null)
SFT 실행 순서 (Kanana, 데이터 재사용):


pip install "transformers>=4.51.0" peft trl datasets
# sft/data/left_train.jsonl, right_train.jsonl 은 팀원 공유분 재사용 (이미 sft/data/에 있음)
python sft/train.py --side left
python sft/train.py --side right
# model.yaml에서 left_adapter/right_adapter 경로 활성화 후 기존대로 실행

(처음부터 새 질문/데이터로 다시 만들고 싶다면 기존대로 generate_data.py부터: 아래 "실행 순서" 참고)

---

## SFT (Self Fine-Tuning) 상세 기록

### 배경 및 목적

논문 *"Systematic Biases in LLM Simulations of Debates"* 의 핵심 발견:
- LLM 에이전트는 시스템 프롬프트로 정치 페르소나를 부여받아도, 토론이 길어질수록 **기저 모델의 내재 편향** 쪽으로 수렴함
- 즉 진보 에이전트가 라운드를 거듭할수록 중립적 발언("양쪽 다 일리가 있다")이 늘어나는 현상 발생

해결책: **자기 생성 데이터로 QLoRA 파인튜닝** → 모델 가중치 자체를 특정 정치 성향으로 조정 → 시스템 프롬프트의 일시적 효과가 아닌 지속적인 성향 유지


---

### 훈련 데이터 구성

**질문 (sft/questions.yaml)**
- 총 100개: 시드 10개 + 직접 작성 90개
- 13개 토픽 카테고리: 경제·세금 / 노동·복지 / 부동산·주거 / 교육 / 환경·에너지 / 안보·외교 / 젠더·가족 / 의료·보건 / 사법·검찰·정치 / 언론·디지털·AI / 이민·다문화 / 지방·행정 / 청년·세대
- 이전 실행에서 LLM 자동 생성 방식을 시도했으나 품질이 낮아(반복·단조로운 패턴) 수동 작성으로 교체

**응답 생성 (sft/generate_data.py)**
- 각 질문마다 `temperature=1.0`, 히스토리 없이 독립 생성으로 다양성 확보
- 100 질문 × 20회 = **2,000 examples** per agent
- 포맷: `{"messages": [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]}`
- LEFT(GPU 0) / RIGHT(GPU 1) 동시 병렬 실행
- 출력: `sft/data/left_train.jsonl`, `sft/data/right_train.jsonl`

**QLoRA 설정 (sft/train.py)**
- 베이스 모델: Qwen2.5-14B-Instruct (4-bit NF4 양자화)
- LoRA: `r=64`, `lora_alpha=128`, `lora_dropout=0.05`
- 대상 모듈: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- 훈련: 1 epoch, `lr=2e-4`, `batch=4`, `grad_accum=8` (effective batch=32)
- 출력: `adapters/left/`, `adapters/right/`

---

### 실행 순서

```bash
conda activate debate
pip install peft trl datasets

# 1. 훈련 데이터 생성 (GPU 0, 1 동시 실행)
tmux split-window -h
# pane 0:
python sft/generate_data.py --side left
# pane 1:
python sft/generate_data.py --side right

# 2. QLoRA 파인튜닝 (한 side씩 — 동시 실행 시 VRAM 부족)
python sft/train.py --side left
python sft/train.py --side right

# 3. config/model.yaml 어댑터 경로 활성화
#   left_adapter:  "adapters/left"
#   right_adapter: "adapters/right"

# 4. 기존대로 실행 (어댑터 자동 로드)
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

---

### 검증 방법

**정량 검증 (논문 방식)**

동일 주제로 10라운드 토론 후, 각 라운드 종료 시점에 에이전트에게 별도 질문:
```
"현재 주제에 대한 당신의 입장을 0(완전 반대)~10(완전 찬성)으로만 답하라."
```
- 어댑터 없는 베이스라인: 라운드가 길어질수록 LEFT 점수가 5에 수렴하는지 관찰
- 어댑터 적용 후: LEFT는 낮은 점수, RIGHT는 높은 점수를 끝까지 유지하는지 확인

**정성 검증**

토론 로그(`logs/`)에서 아래 중립 표현 출현 빈도 비교:
- "양쪽 다 일리가 있다"
- "균형 잡힌 시각으로 보면"
- "복잡한 문제입니다"
- "일부 동의합니다"

SFT 적용 후 이런 표현이 현저히 줄어들면 성공.

**비교 절차**
```bash
# 베이스라인 (어댑터 null 상태에서 저장)
# config/model.yaml: left_adapter: null, right_adapter: null
python debate.py  # 10라운드 후 q → logs/baseline_*.json 저장

# SFT 적용 후
# config/model.yaml: left_adapter: "adapters/left", right_adapter: "adapters/right"
python debate.py  # 동일 주제 10라운드 → logs/sft_*.json 저장
```

---

### 주의사항

- `train.py`는 두 side를 동시에 실행하면 VRAM 부족 → **반드시 순차 실행**
- `generate_data.py`는 GPU가 분리되어 있어 동시 실행 가능
- LoRA rank: `r=256`은 논문에서 벤치마크 성능 하락 관측 → **r=64가 안전한 기본값**
- 어댑터는 `adapters/` 에 저장되므로 재훈련 없이 재사용 가능
- `peft` 패키지 미설치 시 `adapter_path=null`이면 정상 동작 (어댑터 없이 스킵)

---

## Phase 2 — GRPO 파이프라인 (구현 완료, 실행은 GPU 서버에서 진행 예정)

### 배경 및 목적

SFT만으로는 "페르소나 유지"는 되지만 "라운드를 거치며 좋은 토론이 되는가"(반박 품질, 다양성 유지,
동조 방지)는 보상되지 않는다. `보상 설계.pdf`(강수빈)가 GRPO 기반 보상의 원안을 제시했고,
이를 최근 MAD 실패 유형 논문들과 중간발표 슬라이드의 문제-해결 매핑에 맞춰 재검토·재설계한
문서가 `docs/reward_design_v2.md`다. 두 설계 모두 코드로 구현했고 `config/reward.yaml`의
`version`(또는 `--reward-version` CLI 플래그)으로 즉시 전환할 수 있다.

### 구성 요소

| 파일 | 역할 |
|---|---|
| `rl/rewards/base.py` | 보상 계산에 필요한 턴 컨텍스트(`DebateTurnSample`) 정의 |
| `rl/rewards/utils.py` | judge 없이 쓰는 Jaccard/key-point/임베딩 유틸 |
| `rl/rewards/judge.py` | 외부 Judge 인터페이스 — `LocalJudge`(로컬 7B), `APIJudge`(Anthropic/OpenAI) |
| `rl/rewards/v1_pdf.py` | `보상 설계.pdf` 원안 (성향/반박품질/반복패널티) |
| `rl/rewards/v2_redesign.py` | `docs/reward_design_v2.md` 재설계 (페르소나 일관성/참여도/다양성/반복억제/근거충실도) |
| `rl/rewards/composer.py` | 가중합 + TRL `GRPOTrainer` 연동용 `reward_func` 어댑터 |
| `rl/simulate.py` | LEFT/RIGHT 자기 대국(self-play)으로 멀티라운드 트랜스크립트 생성 |
| `rl/rollout.py` | 트랜스크립트 → 턴 단위 GRPO 학습 데이터셋 변환 |
| `rl/build_anchor.py` | R_persona용 페르소나 기준점(anchor) 텍스트 생성 |
| `rl/train_grpo.py` | GRPO 학습 진입점 (SFT 어댑터를 이어받아 LoRA 계속 학습) |
| `config/reward.yaml` | 보상 버전/가중치/judge backend 설정 |

### 실행 순서 (GPU 서버에서)

```bash
conda activate debate
pip install "trl>=0.24" peft datasets sentence-transformers
# API judge를 쓸 경우: pip install anthropic  (또는 openai)

# 1. SFT 어댑터가 이미 있어야 함 (Phase 1 선행)

# 2. self-play 트랜스크립트 생성
python rl/simulate.py --left-adapter adapters/left --right-adapter adapters/right \
    --n-topics 40 --rounds-per-topic 3 --out rl/data/transcripts.jsonl

# 3. (v2 사용 시) R_persona anchor 텍스트 생성 → config/reward.yaml에 반영
python rl/build_anchor.py --side left
python rl/build_anchor.py --side right

# 4. judge 필요 여부 결정: config/reward.yaml의 judge.backend를 none/local/api 중 선택
#    (v1은 judge 필수, v2는 judge 없이도 동작)

# 5. GRPO 학습 (side별 순차 실행 — VRAM 공유)
python rl/train_grpo.py --side left  --reward-version v2
python rl/train_grpo.py --side right --reward-version v2

# 6. config/model.yaml 에서 left_adapter/right_adapter를 adapters/{side}_grpo 로 교체
```

### 설계 근거

- `docs/reward_design_v2.md` §1~§4: 왜 재설계했는지, 논문 근거, 보상 수식, GRPO 목적함수.
- `docs/reward_design_v2.md` §5: 이번 보상 설계로 다루지 않는 것(Judge Ceiling 근본 해결, Belief
  Entrenchment, Tyranny of Majority) — 각각 왜 범위 밖인지 명시.
- v1/v2 모두 같은 `DebateTurnSample` 인터페이스를 구현하므로 side별로 다른 버전을 실험하거나,
  가중치만 바꿔가며 ablation 하는 것도 `config/reward.yaml` 수정만으로 가능하다.

### 주의사항

- `rl/simulate.py`, `rl/train_grpo.py` 모두 LEFT/RIGHT 14B 모델 로딩이 필요해 GPU 서버 전용.
- judge를 `local`로 쓰면 3번째 GPU(또는 여유 VRAM)가 필요 — 3090 24GB 두 장은 이미 LEFT/RIGHT로
  가득 차 있음 (`CLAUDE.md` 참고).
- `R_grounding`은 근거 검색(RAG) 파이프라인이 없어 기본 가중치 0 — 활성화하려면 `DebateTurnSample.evidence`를
  채워주는 검색 단계와 NLI 스코어러를 먼저 붙여야 함.