# HANDOFF — 작업 인계 문서

이 저장소를 새 세션(특히 GPU가 있는 연구실 서버에서 실행되는 Claude Code)이 이어받을 때
가장 먼저 읽는 문서. 지금까지 무엇이 됐고, 뭐가 안 됐고, 다음에 뭘 해야 하는지가 여기 있다.
**작업을 진행할 때마다 이 문서의 체크리스트를 직접 갱신할 것** — §6 참고.

**2026-07-04 업데이트: SFT가 LEFT/RIGHT 둘 다 실제 GPU 서버에서 끝까지 돌아갔다.** LEFT
60/60 스텝(`adapters/left`, loss 2.05→0.17, mean_token_accuracy 0.96), RIGHT도 60/60 스텝
(`adapters/right`, loss 2.03→0.14, mean_token_accuracy 0.97) 완료 — 둘 다 GPU 0/GPU 3에 병렬로
각각 약 6~6.5시간 걸렸다. `config/model.yaml`에 두 경로 모두 반영됨. 아래 §1/§4는 그 과정에서
확인된 내용으로 갱신했다. GRPO 쪽(§3)은 여전히 미검증이니 그 부분은 아래 리스크를 계속 참고할 것.
다음 할 일은 §2의 6번(정성 확인, `python debate.py`)부터.

## 1. 지금까지 진행 상황

| 단계 | 상태 | 위치 |
|---|---|---|
| Streamlit/터미널 토론 시스템 (LEFT vs RIGHT, 상대 발언 주입) | ✅ 완료, Qwen 기준 검증됨. ⚠️ Kanana 어댑터로는 아직 실행 안 해봄(§2-6) | `debate.py`, `app.py`, `core/session.py` |
| SFT 데이터 (페르소나 프롬프트 + 질문 100개 × 응답) | ✅ 팀원 공유분 재사용, `sft/data/*.jsonl`에 있음 | `sft/data/` (git 미추적, `.gitignore`) |
| **base model 교체**: Qwen2.5-14B-Instruct → Kanana-2-30B-A3B-Instruct | ✅ 실제 GPU에서 로딩·학습 검증 완료 | `config/model.yaml` |
| SFT용 LoRA 어댑터 (Kanana 기준) | ✅ LEFT/RIGHT 둘 다 완료, git에 커밋/push됨 | `adapters/left`, `adapters/right` |
| `sft/train.py` — LoRA 대상 모듈 자동 탐색으로 리팩터링 | ✅ 실제 Kanana 모델로 검증 완료(아래 §4 참고, 원래 코드에서 몇 군데 수정 필요했음) | `sft/train.py` |
| GRPO 보상 설계 v1(PDF 원안) / v2(재설계) | ✅ 완료 + 2026-07-04 정비(kiwi 형태소·bge-m3·anchor 임베딩평균·syc coverage 게이트) — **sanity check 전 항목 PASS** | `docs/reward_design_v2.md`, `rl/rewards/`, `rl/check_reward_sanity.py` |
| GRPO 학습 파이프라인 (self-play rollout → GRPOTrainer) | ✅ **LEFT/RIGHT 둘 다 실제 학습 완료** (2026-07-04, side당 15스텝/~65분, vLLM 서버 모드). 어댑터는 `adapters/{left,right}_grpo` (bf16 변환본), `model.yaml`에 반영됨. ⚠️ debate.py 정성 검증은 아직 | `rl/simulate_vllm.py`, `rl/rollout.py`, `rl/train_grpo.py` |
| `check_server.py` — Kanana 기준 실행 가능 여부 체크 | ✅ Plan D로 추가됨, ⚠️ 실제 서버에서 실행 안 해봄 | `check_server.py` |

## 2. 지금 당장 서버에서 할 일 (순서대로)

> **2026-07-03 갱신: 이 서버엔 conda가 없었다.** 아래는 `uv venv`로 실제로 돌려서 검증한 순서다.
> conda가 있는 다른 서버라면 `conda activate debate` 대신 그걸 써도 무방하지만, **transformers
> 버전만큼은 아래처럼 4.57.6으로 고정할 것** (이유는 §4-3 참고 — 5.x는 MoE 구현이 바뀌어서 못 씀).

```bash
# 0. 저장소 최신화 (jaeeun 브랜치)
git pull origin jaeeun

# 1. 환경 설치 (uv 사용 — conda 없는 서버 기준. 이미 .venv 있으면 생략)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "transformers==4.57.6" peft trl datasets accelerate bitsandbytes torch pyyaml
#   ⚠️ "transformers>=4.51.0"로 설치하면 최신 5.x가 잡히는데, 5.x는 DeepSeek-V3 MoE를
#   fused nn.Parameter로 재구현해서 bnb 4bit 양자화도 안 먹히고 peft LoRA도 못 붙는다.
#   반드시 4.57.6(또는 5.0 미만 최신 4.x)으로 고정할 것.

# 2. GPU 확인 — 공용 서버라 다른 사람이 쓰는 GPU가 있을 수 있음 (nvidia-smi로 빈 GPU 확인)
nvidia-smi

# 3. SFT 학습 데이터 확인 — 팀원 공유분이 sft/data/에 있는지
ls sft/data/   # left_train.jsonl, right_train.jsonl 두 개가 보여야 함
#   없으면: 팀원에게 다시 받아서 sft/data/ 에 넣기 (git에는 안 들어있음, .gitignore 처리됨)

# 4. SFT 실행 (side별로 별도 GPU 지정 — 둘 다 GPU 0을 쓰면 순차 실행됨, 빈 GPU가 2장 있으면
#    --gpu만 다르게 줘서 동시 실행 가능. 아래는 순차 실행 예시)
python sft/train.py --side left  --gpu 0
python sft/train.py --side right --gpu 0   # 다른 빈 GPU가 있으면 --gpu 3처럼 바꿔서 병렬 실행
#   출력 중 "LoRA 대상 모듈: N개 (예: [...])" 줄을 꼭 확인 —
#   attention(q_proj/kv_a_proj_with_mqa/kv_b_proj/o_proj)과 shared_experts가 보여야 정상.
#   "gate"(라우터)는 의도적으로 제외됨 (§4-6 참고, LoRA 붙이면 크래시남).
#   ⚠️ 실측 소요시간: 스텝당 약 355~400초, 총 60 스텝 → side당 약 6시간 (§4-7 참고, 이유는
#   transformers의 MoE forward가 128 expert를 Python for-loop로 도는 비최적화 구현이기 때문).

# 5. config/model.yaml에 어댑터 경로 반영 — 2026-07-04, 이미 반영되어 있음(둘 다 완료)
#   left_adapter:  "adapters/left"
#   right_adapter: "adapters/right"

# 6. 정성 확인 — 실제로 토론이 되는지
python debate.py
```

## 3. SFT 이후 — GRPO 파이프라인 (2026-07-04 정비 완료, 아래 순서로 실행)

2026-07-04에 GRPO 파이프라인을 실전 투입 가능하게 정비했다 (보상 설계 리뷰 + 코드 수정 상세는
git log와 [`docs/reward_design_v2.md`](docs/reward_design_v2.md) §3.1 갱신분 참고). 핵심 변경:

- **생성은 vLLM으로**: transformers의 DeepSeek-V3 MoE는 128 expert를 Python loop로 돌아
  (§4-8) GRPO의 대량 생성(스텝당 64회 × 600토큰)이 사실상 불가능. vLLM(0.23, `.venv`에 설치됨,
  fused MoE 커널)으로 self-play 생성과 GRPO 롤아웃 생성을 모두 처리한다. GPU가 97GB라
  bf16(61GB)이 통째로 올라간다 — 학습도 4bit 대신 bf16 기본으로 바꿈(`--precision`으로 변경 가능).
- **보상 정비**: kiwipiepy 형태소 기반 key-point 추출(조사 붙은 어절 문제 해결), 임베딩 모델
  bge-m3 교체(ko-sroberta는 128토큰 잘림), anchor를 텍스트 결합 → 샘플별 임베딩 평균으로 수정,
  sycophancy 페널티에 coverage 게이트 추가, 컴포넌트별 값이 `reward_components/*`로 로깅됨.
- **`rl/check_reward_sanity.py`**: 학습 전 go/no-go 게이트. 2026-07-04 실행 결과 전 항목 PASS
  (persona 분리: own 0.76 > opposite 0.70 > neutral 0.60 / coverage 분리: 0.70 vs 0.19).

```bash
source .venv/bin/activate   # vllm/kiwipiepy/sentence-transformers 이미 설치됨

# 1) self-play 트랜스크립트 생성 (vLLM multi-LoRA, GPU 1장, 수십 분 예상)
python rl/simulate_vllm.py --gpu 0 --n-topics 40 --rounds-per-topic 3

# 2) anchor 생성 (GPU 불필요, 완료됨 — rl/data/anchors/*.json 이미 존재)
python rl/build_anchor.py --side left && python rl/build_anchor.py --side right

# 3) 보상 sanity check (완료됨 — 전 항목 PASS, 재실행해도 됨)
python rl/check_reward_sanity.py

# 4) GRPO 학습 — 터미널 2개
#    [터미널 A] vLLM 롤아웃 서버 (빈 GPU 하나):
CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model kakaocorp/kanana-2-30b-a3b-instruct \
    --dtype bfloat16 --max_model_len 4096 --gpu_memory_utilization 0.85
#    [터미널 B] 학습 (다른 빈 GPU). 먼저 --max-steps 2로 스모크 후 본 학습:
python rl/train_grpo.py --side left --gpu 0 --use-vllm --max-steps 2   # 스모크
python rl/train_grpo.py --side left --gpu 0 --use-vllm                 # 본 학습
python rl/train_grpo.py --side right --gpu 0 --use-vllm

# 5) (권장) iterative self-play 1~2회 반복 — MAPoRL(ACL 2025)식 co-training의 경량 근사.
#    frozen transcript로 1회만 학습하면 정책이 갱신될수록 학습 컨텍스트가 실제 분포와 어긋난다.
python rl/simulate_vllm.py --gpu 0 --left-adapter adapters/left_grpo --right-adapter adapters/right_grpo \
    --out rl/data/transcripts_iter2.jsonl
python rl/train_grpo.py --side left --gpu 0 --use-vllm --transcripts rl/data/transcripts_iter2.jsonl
```

`config/reward.yaml`의 `judge.backend`는 기본 `"none"`이다 (v2는 judge 없이도 동작하도록 설계됨,
[`docs/reward_design_v2.md`](docs/reward_design_v2.md) §1 참고). judge는 보상보다는 **학습 후
평가**에 쓰는 것을 권장 (Abdulhai et al., NeurIPS 2025의 prompt-to-line consistency 방식 —
컨퍼런스 발표용 정량 지표로도 활용 가능).

미구현으로 남긴 것: Diversity Pruning(inference-time, 중간발표 슬라이드 11) — 컨퍼런스에서
질문 가능성 있으니 "향후 과제"로 정리해둘 것. R_grounding은 AI-Hub 국회 회의록 데이터에 BM25만
붙이면 활성화 가능하나 일정상 보류.

### 2026-07-04 GRPO 실행 결과 (완료)

| 항목 | 실측 |
|---|---|
| 트랜스크립트 생성 (240턴 = 40토픽×3라운드×2side) | **~8분** (vLLM multi-LoRA 배치) |
| GRPO 학습 | side당 15 optimizer step, 스텝당 ~255-270초 → **side당 ~65분** |
| 보상 추이 | LEFT 1.0→1.26 범위 등락(약한 상승 경향), RIGHT ~1.0 부근 유지. 1 epoch·lr 1e-5·KL β 0.04의 보수적 설정이라 큰 변화는 아님 — 성향 강화가 부족하면 iterative self-play(§3의 5번) 또는 epoch/lr 상향 검토 |
| KL / clip | kl 0.009~0.015, clip 비율 2~4% — 발산 없음 |
| 산출물 | `adapters/left_grpo`, `adapters/right_grpo` (fp32 저장분을 bf16로 변환 — GitHub 100MB 제한 + 추론 dtype 일치. `ref/` 하위 폴더는 TRL이 저장한 KL 기준 어댑터 사본으로 git 제외) |

**새로 발견한 관측 (§4에 추가할 리스크 성격):**
- **vLLM(생성)과 HF 트레이너(logprob 재계산) 사이 토큰당 로그확률 차이가 큼** (mean ~0.7-0.9).
  MoE는 커널 구현 간 미세한 수치 차이로 expert 라우팅이 갈리기 때문으로 보이며, TRL의
  vLLM importance sampling correction이 보정한다 (kl·보상 추이 정상 확인). MoE 모델 RL의
  실무적 난점 사례로 발표에서 언급할 만함.
- 생성의 92~100%가 `max_new_tokens=600` 상한에 걸림(clipped) — 토론 발언이 원래 긴 스타일.
  보상 계산에는 문제없으나, 완결된 발언을 원하면 max_new_tokens 상향 고려.

### 2026-07-04 정성/정량 검증 + Synthesizer 하이브리드 (완료)

**SFT vs GRPO 비교** (`docs/eval/20260704_sft_vs_grpo_report.md`, 원문 md 포함):
- 중립·양시론 표현 0회(양쪽 모두), anchor 유사도 8라운드 내내 ~0.77 → **중립 회귀 문제는
  SFT 단계에서 이미 해결** (프로젝트 1차 목표 달성 증거)
- GRPO 기여: 논점 커버리지 개선(R2 0.547→0.672 등), 상대 프레임 직접 인용 반박 경향
- 한계 발견: 라운드 5+ 자기반복 심함(Jaccard 0.8~0.9, 사실상 복붙) — 학습이 3라운드까지만
  본 것 + 같은 질문 반복 주입 구조가 원인

**Synthesizer 하이브리드** (`core/synthesizer.py`, `docs/eval/20260704_grpo_vs_hybrid_report.md`):
- 중립 종합자(base 모델, 어댑터 없음)가 ① 3라운드마다 개입해 쟁점 정리 + 미다룬 하위 쟁점으로
  사회자 질문 교체, ② 종료 시 최종 종합(목업 패널 구조) 생성. 판정은 하지 않음(Judge Ceiling 대응).
- `python rl/simulate_vllm.py ... --synthesizer-every 3 --synthesizer-final`로 실행.
  **학습 데이터 생성 시에는 켜지 말 것** (보상 문맥이 달라짐).
- 실측: 개입 직후 자기반복 급감 (R4 0.593→0.250, R7 0.819→0.274) — 반복 루프 차단 확인.
  DPO 학습 버전은 향후 과제.

**다음 세션이 할 일 (후보):** ① base(어댑터 없음) 추가한 3-way 비교로 발표용 그래프 완성,
② UI 연결(팀 준비분)에 Synthesizer 훅 연동, ③ 필요시 iterative self-play (rounds-per-topic 6
권장 — 후반 라운드를 학습 분포에 포함).

## 4. 알려진 리스크 / 이 세션이 검증하지 못한 것

**2026-07-03: 아래 1~3번은 실제 GPU 서버에서 확인해서 해결했다** (LEFT SFT 60/60 스텝 완료
기준). 4~5번은 여전히 미검증. 새로 발견한 이슈(6~8번)도 추가했다.

1. **확인됨: LoRA 대상 모듈 자동 탐색이 맞게 잡힘.** 실제 Kanana 모델에서
   `q_proj`/`kv_a_proj_with_mqa`/`kv_b_proj`/`o_proj`(attention), `mlp.gate_proj/up_proj/down_proj`
   (dense층), `shared_experts.*`까지 336개 모듈이 정상적으로 잡혔다. 단, **`gate`(라우터)는 코드에서
   제외했다** — §6 참고.
2. **확인됨: VRAM 문제없음.** transformers를 4.57.6으로 고정하고 나니(§3 참고) 4bit 양자화가
   제대로 먹혀서 GPU 0 한 장에 48GB 정도만 쓰고 학습이 끝까지 돌아갔다(97GB 중 절반).
   `per_device_train_batch_size`/`gradient_accumulation_steps` 조정 없이 기본값(1/32) 그대로 OK.
3. **원인 확인됨 + 해결: bitsandbytes 4bit 양자화가 transformers 5.x의 MoE 구현과 호환 안 됨.**
   실제로 겪은 문제였다. **transformers 5.x**는 `DeepseekV3MoE`의 128개 routed expert를
   개별 `nn.Linear`가 아니라 `DeepseekV3NaiveMoe`라는 클래스 안의 **fused `nn.Parameter`**
   (`gate_up_proj`/`down_proj`, shape `[128, ...]`)로 구현한다. bnb의 `load_in_4bit`은 `nn.Linear`만
   골라 치환하므로 이 fused 텐서는 양자화가 전혀 안 되고 bf16 그대로 로드되어 GPU 0 한 장에
   ~83~94GB를 먹고 OOM이 났다. **해결책: transformers를 5.x보다 낮은 4.57.6으로 고정**했다 —
   이 버전은 `DeepseekV3MoE.experts = nn.ModuleList([...])`로 expert마다 진짜 `nn.Linear`를 쓰기
   때문에 bnb가 정상적으로 4bit 양자화한다. `sft/train.py` 설치 안내를
   `"transformers>=4.51.0"`이 아니라 `"transformers==4.57.6"`으로 명시해뒀다.
4. **미검증: `apply_chat_template`가 system role을 지원하는지.** `agents/base_agent.py`,
   `sft/train.py`(SFTTrainer) 둘 다 `[{"role":"system",...}, ...]` 형식을 그대로 넘기는데, Kanana의
   chat template jinja가 system role을 안 받으면 에러가 난다. SFT는 `SFTTrainer`가 내부적으로
   처리해서 문제없이 지나갔지만, `debate.py`/`app.py`(실시간 생성)는 아직 실행해보지 않았다.
5. **GRPO 쪽은 전부 미검증.** `rl/` 이하 코드는 문법 검증 + 로직 단위테스트(더미 객체)만 했고,
   실제 GRPOTrainer 학습 루프를 한 번도 돌려본 적이 없다.
6. **새로 발견: MoE 라우터(`mlp.gate`)에 LoRA를 못 붙임.** `DeepseekV3TopkRouter`는 라우터
   가중치를 `nn.Linear`가 아니라 `self.weight`라는 raw `nn.Parameter`로 갖고 있다. peft가 이를
   `ParamWrapper`로 감싸는데 (a) `lora_dropout != 0`을 지원 안 하고, (b) 더 치명적으로 라우팅
   로직이 참조하는 `self.gate.e_score_correction_bias`(로드밸런싱 보정값) 속성을 `ParamWrapper`가
   프록시해주지 않아 forward에서 `AttributeError`로 죽는다. **해결책: `sft/train.py`의
   `_LORA_LEAF_NAMES`에서 `"gate"`를 제외**했다 — attention/MLP/shared_experts 대비 파라미터
   비중도 작아서 손해는 미미하다.
7. **새로 발견: transformers 4.57.6의 DeepSeek-V3 MoE forward 자체에 dtype 버그가 있음.**
   `DeepseekV3MoE.moe()`가 `final_hidden_states`를 `hidden_states.dtype`이 아니라
   `topk_weights.dtype`으로 만드는데, 실제 학습 중(4bit + gradient checkpointing 조합)
   `expert_output * expert_weights`가 다른 dtype으로 나와서
   `index_add_(): self와 source의 scalar type이 달라야 한다`는 RuntimeError로 첫 스텝에서 죽었다.
   해당 함수 자체에 `"CALL FOR CONTRIBUTION! I don't have time to optimise this right now"`라는
   주석이 달려 있을 만큼 미완성 코드. **해결책: `sft/train.py`에 런타임 몽키패치
   (`_patch_deepseek_v3_moe_dtype_bug()`)를 추가**해서 `index_add_` 직전에 명시적으로 dtype을
   맞춰준다. site-packages를 직접 고치지 않고 우리 코드에서 패치하므로 재설치해도 유지된다.
8. **새로 발견: 학습 속도가 예상(20~40분/side)보다 훨씬 느림 — side당 약 6시간.** 원인은
   transformers 4.x/5.x 공통으로 DeepSeek-V3 MoE forward가 128개 expert를 Python for-loop로
   순회하는 비최적화 구현이기 때문(위 7번의 그 함수, 라이브러리 자체가 미완성이라고 인정한 부분).
   `gradient_checkpointing`이 이 느린 loop을 backward 때 한 번 더 돌려서 사실상 2배가 된다.
   DeepSpeed-MoE 같은 진짜 최적화 프레임워크로 바꾸려면 이미 학습된 체크포인트의 가중치 레이아웃을
   변환하는 별도 엔지니어링이 필요해 비현실적이라 판단하고 보류했다. `gradient_checkpointing=False`
   + `per_device_train_batch_size` 상향으로 2~3배 개선 여지는 있으나(메모리 47GB 여유 있음),
   아직 시도 안 함 — 다음 세션에서 시간 여유 있으면 시도해볼 만하다.
9. **공용 서버라 다른 유저의 GPU를 건드리지 않게 주의할 것.** `sft/train.py`는 `--gpu`로 지정한
   물리 GPU만 프로세스에 보이도록 `CUDA_VISIBLE_DEVICES`를 내부에서 설정한다 — 이걸 안 하면
   HuggingFace `Trainer`가 `torch.cuda.device_count()`로 "보이는 GPU 전체"를 세서
   `nn.DataParallel`로 자동 래핑하면서 우리가 고르지 않은(=다른 사람이 쓰는) GPU에까지 모델을
   복제하려다 그쪽에서 OOM을 낸 적이 있다. 이 서버엔 conda도 없어서 `uv venv`로 새로 세팅했다
   (§2 참고).

## 5. 참고 문서

- [`CLAUDE.md`](CLAUDE.md) — 아키텍처 개요, 모듈 책임, 알려진 이슈
- [`readme.md`](readme.md) — Phase 1(SFT)/Phase 2(GRPO) 실행 순서, 검증 방법
- [`docs/reward_design_v2.md`](docs/reward_design_v2.md) — 보상 설계 v1 vs v2 근거, 논문 인용
- [`보상 설계.pdf`](../보상%20설계.pdf), [`투빅스_컨퍼런스 중간발표_NLP.pdf`](../투빅스_컨퍼런스%20중간발표_NLP.pdf), [`평가 데이터셋.pdf`](../평가%20데이터셋.pdf) — 프로젝트 방향성 원본 자료

## 6. 이 문서 갱신 방법

작업을 진행하면서:
- §1 표의 상태를 ✅/❌/⚠️로 갱신하고, 검증됐으면 "⚠️ 미검증" 문구를 지울 것.
- §4의 리스크 중 확인된 것은 "확인됨: 정상 동작" 또는 "확인됨: 실제로는 X라서 Y로 고침" 으로
  바꾸고, 새로 발견한 리스크가 있으면 추가할 것.
- SFT/GRPO가 끝나면 §2/§3의 체크리스트를 "완료, 결과는 logs/... 참고"처럼 갱신할 것.
- 이 문서가 실제 상태와 어긋나기 시작하면(예: 어댑터가 이미 있는데 표는 "없음"이라고 되어 있으면)
  다음 세션이 헷갈리니, 작업 후 바로바로 고쳐둘 것.
