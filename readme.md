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


Phase 1 — SFT 파이프라인 (완료)

sft/questions.yaml: 시드 질문 10개 (경제·노동·환경·안보·사회)
sft/generate_data.py: 질문당 20회 응답 생성 → JSONL 저장
sft/train.py: QLoRA (r=64) + SFTTrainer 1 epoch → adapters/{side}/ 저장
agents/base_agent.py: adapter_path 파라미터 추가, load() 끝에 PeftModel 조건부 적용
config/model.yaml: left_adapter / right_adapter 항목 추가 (기본 null)
SFT 실행 순서:


pip install peft trl datasets
python sft/generate_data.py --side left    # ~2~3시간
python sft/generate_data.py --side right   # ~2~3시간
python sft/train.py --side left            # ~10분
python sft/train.py --side right           # ~10분
# model.yaml에서 left_adapter/right_adapter 경로 활성화 후 기존대로 실행

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