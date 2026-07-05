# GRPO 강화학습 계획 — 토론 에이전트 성향 강화

> **변경 이력 (계획 승인 후, 2026-07-02):**
> 1. **SFT 산출물 완전 배제** — 원안은 미니 토론 생성과 평가 상대에 SFT 어댑터를 사용했으나,
>    사용자 결정으로 데이터 생성·평가 상대·배포 설정(config/model.yaml 어댑터 null) 모두
>    순수 base + 페르소나 프롬프트만 사용. 비교 평가도 SFT 대신 base(`--adapter none --tag base`).
> 2. **시드 2회 → 1회** (rl/config.yaml `n_seeds: 1`) — 데이터 생성 시간 단축.
>    프롬프트는 사이드당 ~300행 수준 (150스텝 × 프롬프트 2개/스텝 = 300개 소비에 부합).
> 3. **추가 파일** — `rl/run_all.sh`(tmux 전체 파이프라인 순차 실행),
>    `rl/monitor_metrics.py`(메트릭 CSV `logs/rl_metrics.csv` + 경고 `logs/rl_alerts.log`),
>    `rl/utils.py`(공유 헬퍼).
> 4. **배치 축소** — 생성 배치 16(프롬프트 2×8생성)이 스모크에서 OOM →
>    `per_device_train_batch_size: 1`(생성 배치 8, 스텝당 프롬프트 1개)로 확정. 아래 본문의
>    GRPOConfig 예시(batch 2)는 원안 기록이며 실제 값은 rl/config.yaml이 기준.
> 5. **(2026-07-04) 모델 Kanana 전환 + 보상 고도화** — base를 kakaocorp/kanana-2-30b-a3b-instruct로
>    교체(jaeeun 브랜치의 Kanana SFT 어댑터 도입, GRPO는 이를 이어받아 학습으로 변경).
>    transformers==4.57.6 필수 + MoE dtype 패치는 rl/model_utils.py가 처리. 이 하드웨어(3090
>    24GB×2)로는 Kanana 인스턴스 1개도 OOM(실측)이라 실행은 다른 서버에서 — rl/config.yaml의
>    left_gpus/right_gpus/judge_gpu로 GPU 배치 조정.
>    보상은 8종으로 고도화: korean_ratio 삭제(한국어 네이티브 모델이라 불필요),
>    engagement(키워드 커버리지)·semantic_echo(의미 반복/동조 임베딩 패널티) 추가,
>    그리고 논문 차용 3종 — rebuttal을 pairwise Bradley-Terry로(arXiv:2605.28313),
>    persona 정합도 보상(arXiv:2511.00222), judge 불확실성 라우팅(arXiv:2510.20369, 기본 off).
>    실검증: pairwise BT가 정면반박 1.00 > 논점회피 0.01 > 구호나열 0.00으로 판별,
>    persona 보상 정합 0.655 > 이탈 0.301 (실제 7B judge + ko-sroberta 임베더).

## Context

SFT만으로는 토론이 길어질수록 에이전트가 중립으로 회귀하는 문제(readme.md의 neutral drift)가 남아 있다. 이를 해결하기 위해 **순수 base 모델(Qwen2.5-14B-Instruct, SFT 어댑터 제외) + 새 QLoRA 어댑터**에 GRPO를 적용해 (1) 성향 일관성, (2) 반박 품질, (3) 자기 반복 억제를 보상으로 직접 최적화한다.

- **확정 사항 (사용자 결정):** GRPO(trl GRPOTrainer 1.5.1), 순수 base + 새 LoRA, Judge는 로컬 Qwen2.5-7B-Instruct 4bit(유휴 GPU), 에이전트당 100~150스텝(~하루), LEFT→RIGHT 순차 학습.
- **하드웨어:** RTX 3090 24GB × 2. 풀 파인튜닝 불가 → QLoRA 필수.
- **핵심 검증된 trl 1.5.1 사실:** ① bnb-4bit 모델 + `peft_config` 직접 지원 (`get_peft_model`/`prepare_model_for_kbit_training` 직접 호출 금지 — 트레이너가 처리, fp32 캐스팅 안 해서 VRAM 절약) ② PEFT + `beta>0`이면 어댑터 비활성화로 ref logprob 계산 → 별도 ref 모델 불필요 ③ **`max_prompt_length` 파라미터 없음** → 데이터 빌더에서 프롬프트 길이 직접 제한 필수 ④ reward func는 `f(prompts, completions, completion_ids, **kwargs)` 형태, 데이터셋의 추가 컬럼이 kwargs로 전달됨, `None` 반환 시 해당 컴포넌트 제외 ⑤ 대화형(`[{"role","content"}]`) 프롬프트 지원 ⑥ `generation_batch_size`는 `num_generations`로 나누어떨어져야 함 ⑦ `loss_type` 기본값 `"dapo"` (유지).

## 새 파일 구조 (sft/ 관례를 따름)

```
rl/
├── build_prompts.py   # 프롬프트 데이터셋 빌더 (순수 base로 상대 발언 오프라인 생성)
├── judge.py           # JudgeClient: Qwen2.5-7B 4bit, 배치 JSON 채점 (stance + rebuttal 한 번에)
├── rewards.py         # 보상 함수 팩토리 (judge 기반 2개 + 규칙 기반 4개)
├── train_grpo.py      # GRPO 학습 스크립트 (CLI: --side left|right --steps --resume)
├── eval_grpo.py       # 평가: 스탠스 곡선 / 중립 표현 카운트 / before-after 비교
├── config.yaml        # RL 하이퍼파라미터 + 보상 가중치 (튜닝 한 곳에 집중)
└── data/
    ├── left_prompts.jsonl / right_prompts.jsonl   # ~500개/에이전트
    ├── heldout.yaml                               # 평가용 홀드아웃 질문 10개
    └── transcripts.jsonl                          # 오프라인 미니 토론 원본 (감사용)
adapters/left_grpo/  adapters/right_grpo/          # 학습 결과물
```

## 1단계 — 프롬프트 데이터셋 (`rl/build_prompts.py`)

`sft/questions.yaml` 100개 중 90개 사용, 10개는 토픽별 층화 추출로 홀드아웃.

**오프라인 미니 토론 생성:** 기존 SFT 어댑터(adapters/left·right)를 각 GPU에 로드해 90개 질문 × 2 시드(temp 1.0/0.8)로 3라운드 토론을 시뮬레이션. 메시지 포맷은 `core/session.py`의 `_build_message()`(L86-93) 문자열을 **그대로 재사용**해 학습/배포 분포를 일치시킴. `max_new_tokens=400`.

**트랜스크립트 → 프롬프트 슬라이싱 (사이드당 ~500개):**
- **1라운드 (~90):** `[system(페르소나), user(질문)]` — 상대 발언 없음, `own_history=[]`
- **2라운드 (~180):** 상대의 1라운드 발언이 주입된 user 메시지
- **3라운드 (~230):** `[system, user(2R 메시지), assistant(자기 2R 발언), user(3R 메시지)]` + `own_history=[자기 2R 발언]` → **자기 반복 패널티가 실제 히스토리 H를 갖게 됨**

**JSONL 스키마** (추가 컬럼은 reward func kwargs로 그대로 전달됨):
```json
{"prompt": [...], "side": "left", "qid": 17, "round": 3, "question": "...",
 "opponent_statement": "...", "own_history": ["..."]}
```

**길이 제한 (필수):** 주입되는 상대 발언은 앞 3문장으로 트림, 자기 과거 발언은 최대 1턴. 최종 템플릿 적용 후 토크나이즈해 **1,400 토큰 초과 행은 재트림/제외** (trl 1.5.1은 프롬프트를 절대 안 자름).

## 2단계 — Judge (`rl/judge.py`)

```python
class JudgeClient:
    def __init__(self, model_id="Qwen/Qwen2.5-7B-Instruct", gpu=1)  # 4bit NF4, ~5.5GB
    def score_batch(self, items) -> list[{"stance": float, "rebuttal": int, "parse_ok": bool}]
```
- **호출 1회로 두 점수 동시 반환** (JSON): stance −1.0(뚜렷한 진보)~+1.0(뚜렷한 보수), rebuttal 1~5. 한국어 few-shot 2~3개 포함 — 그중 하나는 "키워드만 나열하고 논점 회피" 사례를 낮게 채점(Judge 게이밍 방지).
- Judge는 질문/상대발언/답변만 봄 (정책의 시스템 프롬프트는 안 보여줌 — 자기 라벨링으로 점수 못 땀).
- Greedy 디코딩, `max_new_tokens=48`, 16개 completion을 left-padding 배치로 한 번에 채점 (~20-40초/스텝).
- 파싱 실패 시 `{stance: 0.0, rebuttal: 3}` 기본값(중립 → advantage 기여 ≈ 0) + `log_metric`으로 실패율 기록. 5% 초과 시 few-shot 보강.
- **GPU 배치:** 학습 프로세스 안에서 judge를 `device_map={"": 1}`로 로드. `CUDA_VISIBLE_DEVICES`로 물리 GPU 스왑 — LEFT: `0,1`, RIGHT: `1,0` (코드는 항상 학습=0, judge=1).

## 3단계 — 보상 함수 (`rl/rewards.py`)

각 컴포넌트를 **개별 reward func**로 등록하고 `GRPOConfig(reward_weights=...)`로 가중 → 트레이너 로그에서 컴포넌트별 곡선 자동 기록. stance+rebuttal은 judge 1회 호출을 공유하도록 `id(completions)` 키 메모 캐시 사용.

| # | 함수 | 정의 | 범위 | 가중치 |
|---|---|---|---|---|
| 1 | `reward_stance` | LEFT: −S, RIGHT: +S | [−1,1] | **1.0** |
| 2 | `reward_rebuttal` | (Q−1)/4; 상대 발언 없으면 `None` 반환(1라운드 행 제외) | [0,1] | **0.7** |
| 3 | `reward_antirep` | −1.5·max(0, maxⱼ Jaccard(y, Hⱼ) − 0.25); H 없으면 0 | [−1.1,0] | **1.0** |
| 4 | `reward_neutral_phrase` | 블랙리스트 히트당 −0.5, 하한 −1.5 ("양쪽 다 일리", "균형 잡힌 시각", "복잡한 문제입니다", "일부 동의", "절충", "양측 모두", "중립적" — readme.md 검증 프로토콜 + prompts.yaml 금지목록에서 추출) | [−1.5,0] | **1.0** |
| 5 | `reward_korean_ratio` | 한글/전체문자 비율 페널티 −2·max(0, 0.85−ratio); 한자 3연속 이상 추가 −1.0 (Qwen 중국어 드리프트 방지) | ≤0 | **1.0** |
| 6 | `reward_format` | 80자 미만 −1.0; 미종결(토큰캡 도달+문장 안 끝남) −0.5; 불릿/번호목록 −0.5 | [−2,0] | **0.5** |

기본 `scale_rewards="group"` + `sum_then_normalize` 유지 (가중합 후 그룹 내 z-score → 상대 가중치만 유효). 4~6번 shaping 페널티는 stance를 압도하지 않고 동점을 가르는 크기로 설정됨.

## 4단계 — 학습 스크립트 (`rl/train_grpo.py`)

`sft/train.py` 구조를 따르되: base를 4bit NF4 + `device_map={"": 0}`으로 로드(어댑터 없음, `prepare_model_for_kbit_training` 호출 안 함), 새 `LoraConfig(r=16, alpha=32, dropout=0.0, 타겟 모듈 7개 동일)`을 `peft_config`로 트레이너에 전달.

**GRPOConfig 권장값:**
```python
max_steps=150, learning_rate=1e-5(LoRA 스케일; 기본 1e-6은 너무 느림),
lr_scheduler_type="constant_with_warmup", warmup_steps=10,
per_device_train_batch_size=2, gradient_accumulation_steps=8,   # → generation_batch_size=16
num_generations=8, max_completion_length=400,
temperature=0.9, top_p=0.95, repetition_penalty=1.0(정책 확률 왜곡 방지 — 반복은 보상으로 처리),
beta=0.02(base 대비 KL — 어댑터 비활성화 트릭, 추가 VRAM 0),
loss_type="dapo", mask_truncated_completions=True,
optim="paged_adamw_8bit", bf16=True, gradient_checkpointing=True,
save_steps=25, save_total_limit=3, log_completions=True, reward_weights=[...]
```

**VRAM 검산 (24GB):** 14B NF4 ~10.5GB + LoRA/옵티마이저 <0.5GB + 생성 KV캐시(16seq×~1,700tok) ~5.2GB + logprob/학습 forward ~3GB + 여유 ~1.5GB = **피크 ~20GB ✅**. OOM 시 폴백: `per_device_train_batch_size=1`(생성 배치 8), 프롬프트 캡 1,200. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 설정.

**시간 추정:** 스텝당 ~4-5분(생성 2-3분 + judge 0.5분 + logprob/역전파 1.5분) × 150스텝 ≈ **10-12시간/에이전트**. 너무 느리면 100스텝 또는 `max_completion_length=320`으로 축소.

## 5단계 — 평가 (`rl/eval_grpo.py`)

readme.md 검증 프로토콜 재사용:
1. **스탠스 곡선:** 홀드아웃 토픽 3개로 10라운드 토론(테스트 어댑터 vs 상대 SFT 어댑터), 매 라운드 후 "입장을 0~10으로만 답하라"(greedy). {base, SFT, GRPO} 3개 조건 비교 — LEFT는 낮게, RIGHT는 높게 유지되고 5로 수렴하지 않으면 성공.
2. **중립 표현 카운트:** 위 트랜스크립트에 보상 #4와 동일한 블랙리스트 적용.
3. **Before/After 비교:** 홀드아웃 질문 10개에 대해 SFT vs GRPO 답변을 나란히 생성 + judge 채점 테이블 (markdown 출력).

## 실행 순서 (Runbook)

```bash
conda activate debate && cd /home/nlplab9/Desktop/debate
python rl/build_prompts.py                                       # ~2-3h, 양 GPU 동시
CUDA_VISIBLE_DEVICES=0,1 python rl/train_grpo.py --side left     # ~10-12h → adapters/left_grpo
CUDA_VISIBLE_DEVICES=1,0 python rl/train_grpo.py --side right    # ~10-12h → adapters/right_grpo
python rl/eval_grpo.py --side left --adapter adapters/left_grpo
python rl/eval_grpo.py --side right --adapter adapters/right_grpo
```
배포: `config/model.yaml`의 `left_adapter/right_adapter`를 `adapters/{side}_grpo`로 변경 (base_agent.py 코드 수정 불필요).

## 리스크와 대응

- **보상 해킹(구호 반복, 키워드 스터핑):** rebuttal 보상 + judge few-shot에 "공허한 키워드 나열" 저점 예시 + `log_completions`로 25스텝마다 샘플 육안 확인 + 25스텝 체크포인트로 롤백 가능.
- **엔트로피 붕괴/구호 고착:** beta=0.02 KL + temp 0.9 + 실제 own_history 기반 반복 페널티. 트레이너의 `frac_reward_zero_std` 메트릭 감시 — 그룹 보상 분산이 0에 가까워지면 학습 신호 사망 → temp/가중치 조정.
- **언어 드리프트(중국어/영어):** 한글 비율 보상 + 로그에서 육안 확인.
- **프롬프트 길이 초과:** 빌더에서 1,400 토큰 하드캡 (trl에 안전망 없음).

## 검증 방법

1. `build_prompts.py` 후: JSONL 행 수(~500/사이드), 토큰 길이 p95 ≤ 1,200, 라운드 분포 확인.
2. 본 학습 전 **smoke test**: `--steps 3`으로 LEFT 실행 → OOM 없음, judge 파싱 성공률 >95%, 컴포넌트별 보상 로그 정상 출력 확인.
3. 학습 중: 25스텝마다 `log_completions` 샘플 + `rewards/*/mean` 곡선 상승 확인.
4. 학습 후: `eval_grpo.py` 3종 평가 → 스탠스 곡선이 SFT 대비 개선(수렴 없음), 중립 표현 감소.
