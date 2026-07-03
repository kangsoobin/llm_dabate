# HANDOFF — 작업 인계 문서

이 저장소를 새 세션(특히 GPU가 있는 연구실 서버에서 실행되는 Claude Code)이 이어받을 때
가장 먼저 읽는 문서. 지금까지 무엇이 됐고, 뭐가 안 됐고, 다음에 뭘 해야 하는지가 여기 있다.
**작업을 진행할 때마다 이 문서의 체크리스트를 직접 갱신할 것** — §6 참고.

이 문서를 쓰는 시점(로컬, GPU 없는 환경)까지의 작업은 코드/설정/문서 작성까지만 되어 있고,
**실제 GPU에서 실행해서 검증된 것은 하나도 없다.** 아래 리스크(§4)를 꼭 먼저 읽을 것.

## 1. 지금까지 진행 상황

| 단계 | 상태 | 위치 |
|---|---|---|
| Streamlit/터미널 토론 시스템 (LEFT vs RIGHT, 상대 발언 주입) | ✅ 완료, Qwen 기준 검증됨 | `debate.py`, `app.py`, `core/session.py` |
| SFT 데이터 (페르소나 프롬프트 + 질문 100개 × 응답) | ✅ 팀원 공유분 재사용, `sft/data/*.jsonl`에 있음 | `sft/data/` (git 미추적, `.gitignore`) |
| **base model 교체**: Qwen2.5-14B-Instruct → Kanana-2-30B-A3B-Instruct | ✅ 설정/코드는 반영됨, ⚠️ **GPU에서 실행 검증 안 됨** | `config/model.yaml` |
| SFT용 LoRA 어댑터 (Qwen 기준) | ❌ Kanana에는 로드 불가 (아키텍처 다름) — 재학습 필요 | `adapters/left`, `adapters/right` (현재 없음, `model.yaml`에서 null) |
| `sft/train.py` — LoRA 대상 모듈 자동 탐색으로 리팩터링 | ✅ 코드 작성 + 가짜 모듈 트리로 로직 단위테스트 완료, ⚠️ **실제 Kanana 모델로는 미검증** | `sft/train.py` |
| GRPO 보상 설계 v1(PDF 원안) / v2(재설계) | ✅ 문서 + 코드 둘 다 완료 | `docs/reward_design_v2.md`, `rl/rewards/` |
| GRPO 학습 파이프라인 (self-play rollout → GRPOTrainer) | ✅ 코드 작성 완료, ⚠️ **GPU에서 전혀 실행 안 해봄** | `rl/simulate.py`, `rl/rollout.py`, `rl/train_grpo.py` |
| `check_server.py` — Kanana 기준 실행 가능 여부 체크 | ✅ Plan D로 추가됨, ⚠️ 실제 서버에서 실행 안 해봄 | `check_server.py` |

## 2. 지금 당장 서버에서 할 일 (순서대로)

```bash
# 0. 저장소 최신화 (jaeeun 브랜치)
git pull origin jaeeun

# 1. 서버 사양/환경 체크 — Plan D(Kanana-2-30B-A3B 4bit)가 ✔ 인지 확인
python check_server.py
#   ✘ 뜨면: transformers 버전(>=4.51.0), GPU 여유 VRAM(권장 18GB+/GPU), 디스크(65GB+) 확인

# 2. 환경 설치/업그레이드 (기존 conda env가 있다면 transformers만 업그레이드해도 됨)
conda activate debate
pip install "transformers>=4.51.0" peft trl datasets accelerate bitsandbytes

# 3. SFT 학습 데이터 확인 — 팀원 공유분이 sft/data/에 있는지
ls sft/data/   # left_train.jsonl, right_train.jsonl 두 개가 보여야 함
#   없으면: 팀원에게 다시 받아서 sft/data/ 에 넣기 (git에는 안 들어있음, .gitignore 처리됨)

# 4. SFT 실행 (side별 순차로 — 동시 실행 시 VRAM 부족 가능성 큼, 30B라 Qwen 14B보다 훨씬 빠듯)
python sft/train.py --side left
#   출력 중 "LoRA 대상 모듈: N개 (예: [...])" 줄을 꼭 확인 —
#   attention(q_proj/kv_a_proj_with_mqa/kv_b_proj/o_proj)과 MoE 라우터(mlp.gate),
#   shared_experts가 보여야 정상. 10개 미만이거나 attention이 하나도 안 보이면
#   Kanana 실제 모듈 이름이 예상과 다른 것 — sft/train.py의 discover_lora_target_modules()
#   주석 참고해서 _LORA_LEAF_NAMES 집합을 실제 모델에 맞게 고쳐야 함.
python sft/train.py --side right

# 5. config/model.yaml에 어댑터 경로 반영
#   left_adapter:  "adapters/left"
#   right_adapter: "adapters/right"

# 6. 정성 확인 — 실제로 토론이 되는지
python debate.py
```

## 3. SFT 이후 — GRPO 파이프라인 (코드는 있지만 아직 한 번도 안 돌려봄)

SFT가 끝나고 어댑터가 정상 동작하는 걸 확인한 뒤에 진행. 순서와 상세는
[`readme.md`](readme.md)의 "Phase 2 — GRPO 파이프라인" 섹션에 이미 정리되어 있다. 요약:

```bash
pip install "trl>=0.24" sentence-transformers   # (선택) API judge면 anthropic 또는 openai도

python rl/simulate.py --left-adapter adapters/left --right-adapter adapters/right \
    --n-topics 40 --rounds-per-topic 3 --out rl/data/transcripts.jsonl

python rl/build_anchor.py --side left
python rl/build_anchor.py --side right
# → 출력된 anchor 텍스트를 config/reward.yaml의 v2.anchor_texts.{left,right}에 붙여넣기

python rl/train_grpo.py --side left  --reward-version v2
python rl/train_grpo.py --side right --reward-version v2
```

`config/reward.yaml`의 `judge.backend`는 기본 `"none"`이다 (v2는 judge 없이도 동작하도록 설계됨,
[`docs/reward_design_v2.md`](docs/reward_design_v2.md) §1 참고). judge를 쓰고 싶으면 `local`(여유
GPU 필요, Kanana가 이미 24GB를 많이 쓰고 있어 3번째 GPU 권장) 또는 `api`로 바꿀 것.

## 4. 알려진 리스크 / 이 세션이 검증하지 못한 것

GPU가 없는 환경에서 작성했기 때문에, 아래는 **코드 리뷰·단위테스트로만 확인했고 실제 모델로는
확인 못한** 항목이다. 서버에서 문제가 생기면 여기부터 의심할 것.

1. **LoRA 대상 모듈 자동 탐색이 진짜 Kanana 모델에서도 맞게 잡히는지.** `sft/train.py`의
   `discover_lora_target_modules()`는 Kanana의 공개 `config.json`(model_type: `deepseek_v3`,
   `q_lora_rank: null` → `q_proj` 직접 사용, `kv_a_proj_with_mqa`/`kv_b_proj`, MoE
   `experts.N.*`/`shared_experts.*`/`gate`)을 근거로 작성했고 가짜 모듈 트리로는 검증했지만,
   실제 HF에 올라간 모델 구현이 100% 이 이름을 쓰는지는 실물로 확인 못했다.
2. **VRAM이 실제로 24GB 한 장에 들어가는지.** 4bit 기준 가중치만 ~17~18GB로 추정했는데, QLoRA
   학습 시 옵티마이저 상태 + gradient checkpointing + 활성화 메모리까지 더하면 빠듯하거나 넘칠 수
   있음. OOM 나면 `sft/train.py --gpu`로 여유 GPU 지정하거나, `per_device_train_batch_size`/
   `gradient_accumulation_steps`(현재 1/32)를 조정.
3. **bitsandbytes 4bit 양자화가 Kanana의 MoE 레이어 구조와 호환되는지.** HF 레퍼런스 구현이
   expert마다 개별 `nn.Linear`를 쓰는 구조라면 문제없지만, 만약 실제로는 fused/batched 텐서로
   구현되어 있다면 bnb의 `Linear4bit` 자동 치환이 안 먹힐 수 있음.
4. **`apply_chat_template`가 system role을 지원하는지.** `agents/base_agent.py`,
   `sft/train.py`(SFTTrainer) 둘 다 `[{"role":"system",...}, ...]` 형식을 그대로 넘기는데, Kanana의
   chat template jinja가 system role을 안 받으면 에러가 난다.
5. **GRPO 쪽은 전부 미검증.** `rl/` 이하 코드는 문법 검증 + 로직 단위테스트(더미 객체)만 했고,
   실제 GRPOTrainer 학습 루프를 한 번도 돌려본 적이 없다.

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
