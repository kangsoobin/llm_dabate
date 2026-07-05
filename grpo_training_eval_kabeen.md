# Kanana-2-30B-A3B-Instruct SFT Adapter 기반 GRPO 강화학습 및 GovReport 평가 정리

> 작성 목적: `kabeen` 브랜치에 포함할 강화학습 설계, 실제 학습 설정, 산출 adapter, 평가 방법과 결과를 팀원이 재현/검토할 수 있도록 정리한다.  
> 기준 경로: `/root/kb/llm_dabate_kabeen_grpo_v3`  
> 기준 모델: `kakaocorp/kanana-2-30b-a3b-instruct`

## 1. 전체 요약

이번 작업은 기존 `kanana-2-30b-a3b-instruct` 기반 SFT adapter에서 출발해, LEFT/RIGHT 토론 에이전트를 각각 GRPO로 추가 강화학습한 것이다.

핵심 목표는 단순히 문체를 바꾸는 것이 아니라 다음 세 가지 토론 능력을 강화하는 것이다.

1. **정치적 성향 일관성 유지**
   - LEFT는 진보적 관점, RIGHT는 보수적 관점을 유지한다.
   - 단, 상대 주장에 무조건 동조하거나 중립으로 회귀하지 않도록 한다.

2. **상대 논점에 대한 정면 반박 품질 향상**
   - 상대 발언의 핵심 논점을 잡고, 회피하지 않고 직접 반박하도록 보상한다.
   - 단순 키워드 언급만으로 보상을 받는 것을 줄이기 위해 반박 cue와 coverage를 함께 본다.

3. **자기 반복 감소 및 토론 전개 능력 강화**
   - 후반 라운드에서 같은 말을 반복하는 문제를 줄인다.
   - 자기 과거 발언과 다른 새 하위 논점을 제시하도록 유도한다.

최종 적용 adapter는 다음과 같다.

| side | 최종 적용 adapter | 선택 이유 |
|---|---|---|
| LEFT | `adapters/left_grpo_v3_left_long_g9/checkpoint-112` | 보존된 checkpoint 중 reward 균형이 가장 좋음. direct rebuttal, issue coverage, argument advancement가 높음. |
| RIGHT | `adapters/right_grpo_v3_right_long_g8/checkpoint-120` | 보존된 checkpoint 중 stance/direct/coverage/advancement 균형이 가장 좋음. |

현재 `config/model.yaml`도 위 adapter를 바라보도록 설정되어 있다.

```yaml
left_adapter:  "adapters/left_grpo_v3_left_long_g9/checkpoint-112"
right_adapter: "adapters/right_grpo_v3_right_long_g8/checkpoint-120"
```

## 2. 기반 모델과 SFT adapter

### 2.1 Base model

- 모델 ID: `kakaocorp/kanana-2-30b-a3b-instruct`
- 구조: DeepseekV3 계열 MoE 구조
- 프로젝트 내 권장 transformers 버전: `transformers==4.57.6`
- 학습/평가 로딩 방식: 4bit NF4 quantization 기반

### 2.2 SFT adapter

GRPO는 base model에서 바로 시작하지 않고, 기존 SFT adapter를 이어받아 수행했다.

| side | SFT adapter |
|---|---|
| LEFT | `adapters/left` |
| RIGHT | `adapters/right` |

중요한 점은 `rl/train_grpo.py` 실행 시 `--init-adapter adapters/left` 또는 `--init-adapter adapters/right`를 명시했다는 것이다. 따라서 이번 GRPO는 다음 흐름이다.

```text
Kanana base model
  -> LEFT/RIGHT SFT adapter
  -> LEFT/RIGHT GRPO adapter
```

즉, GRPO는 SFT로 만들어진 토론 페르소나와 기본 응답 형식을 유지한 상태에서 보상 기반 미세 조정을 추가한 것이다.

## 3. 기존 보상 설계와 v3 확장

처음 제안된 보상 설계는 다음 세 축이었다.

```text
R_total(y_i) = w1 * R_stance(y_i) + w2 * R_rebuttal(y_i) + w3 * R_anti_rep(y_i)
```

- `R_stance`: 정치 성향 일관성 보상
- `R_rebuttal`: 상대 논점에 대한 반박 품질 보상
- `R_anti_rep`: 자기 반복 패널티

이번 구현에서는 이 원안을 그대로 버리지 않고, 실제 multi-agent 토론에서 더 안정적으로 작동하도록 v3 보상으로 확장했다.

## 4. v3 보상 설계

v3 보상은 원래 세 축을 아래처럼 더 세분화했다.

```text
R_total
= R_stance_axis
+ R_rebuttal_axis
+ R_dynamics_axis
+ R_grounding_axis
```

실제 구현 컴포넌트는 `rl/rewards/v3_multiagent.py`에 있다.

### 4.1 원안 보상과 v3 컴포넌트 매핑

| 원안 축 | v3 컴포넌트 | 설명 |
|---|---|---|
| 성향 일관성 | `stance_alignment` | LEFT/RIGHT의 페르소나 anchor와 현재 답변의 의미 유사도를 보상한다. |
| 반박 품질 | `issue_coverage` | 상대 발언의 핵심 논점을 얼마나 포함했는지 본다. |
| 반박 품질 | `direct_rebuttal` | 상대 논점을 단순 언급하는 것이 아니라 반박 문맥에서 다루는지 본다. |
| 자기 반복 억제 | `lexical_anti_repetition` | 자기 과거 발언과의 단어 Jaccard 반복을 패널티로 준다. |
| 자기 반복 억제 | `late_loop_penalty` | 후반 라운드 반복 루프를 더 강하게 벌한다. |
| 토론 전개 | `semantic_distinctiveness` | 상대/자기 과거 발언과 의미적으로 너무 비슷해지는 것을 줄인다. |
| 토론 전개 | `argument_advancement` | 자기 과거 발언 대비 새 논점/내용어를 제시하면 보상한다. |
| 근거 충실도 | `evidence_grounding` | RAG/NLI가 붙기 전까지 0으로 비활성화했다. |

### 4.2 실제 reward weight

`config/reward.yaml`의 v3 설정은 다음과 같다.

```yaml
v3:
  weights:
    stance_alignment: 1.0
    issue_coverage: 1.1
    direct_rebuttal: 0.5
    semantic_distinctiveness: 1.0
    argument_advancement: 0.6
    lexical_anti_repetition: 0.4
    late_loop_penalty: 0.7
    evidence_grounding: 0.0
```

### 4.3 성향 유지: `stance_alignment`

`stance_alignment`는 SFT adapter가 만든 정치적 정체성을 유지하도록 하는 보상이다.

- LEFT는 진보 성향 anchor와 가까울수록 보상이 높다.
- RIGHT는 보수 성향 anchor와 가까울수록 보상이 높다.
- 상대 의견을 다루는 과정에서 너무 동조하거나 중립으로 흐르는 현상을 줄인다.

원래 제안했던 LLM-as-a-Judge 기반 `S(y) in [-1, 1]` 방식과 대응시키면 다음과 같다.

```text
LEFT  : R_stance = -S(y)
RIGHT : R_stance =  S(y)
```

다만 이번 실제 학습에서는 judge backend를 `none`으로 두었다. 즉 OpenAI API나 외부 judge 모델 없이, embedding/anchor 기반의 judge-free reward를 사용했다.

```yaml
judge:
  backend: "none"
```

따라서 이번 GRPO에는 OpenAI API가 필요하지 않았다.

### 4.4 논점 커버리지: `issue_coverage`

`issue_coverage`는 상대 발언의 핵심 논점을 현재 답변이 얼마나 다루는지 보는 보상이다.

원래 설계의 반박 품질 보상은 judge가 1~5점으로 평가하고 다음처럼 정규화하는 방식이었다.

```text
R_rebuttal(y_i) = (Q(y_i) - 1) / 4
```

이번 v3에서는 외부 judge 없이도 학습이 가능하도록, 상대 발언과 현재 답변의 핵심 표현/의미 overlap을 기반으로 `issue_coverage`를 계산했다. 이 값은 상대 논점 회피를 줄이는 역할을 한다.

### 4.5 직접 반박성: `direct_rebuttal`

단순히 상대 키워드를 언급하는 것만으로는 좋은 반박이라고 보기 어렵다. 그래서 v3에는 `direct_rebuttal`을 추가했다.

이 보상은 다음 두 요소를 함께 본다.

1. 상대 핵심 논점 coverage
2. 반박 cue 사용 여부

반박 cue 예시는 다음과 같다.

```text
그러나, 하지만, 반면, 오히려, 그 주장은, 문제는, 사실과 다르다, 간과한다, 역효과
```

개념식은 다음과 같다.

```text
R_direct_rebuttal = (1 - a) * coverage(opponent, y) + a * cue(y)
```

기본 설정은 `cue_weight = 0.35`다.

### 4.6 자기 반복 억제: `lexical_anti_repetition`

원안의 자기 반복 패널티는 다음과 같았다.

```text
J(y_i, H) = 현재 답변과 과거 자기 발언 히스토리 간 단어 단위 Jaccard 유사도

R_anti_rep = 0                       if J <= tau
R_anti_rep = -gamma * (J - tau)       if J > tau
```

이번 구현에서도 같은 아이디어를 유지했다.

```yaml
lexical_anti_repetition:
  tau: 0.25
  gamma: 1.5
```

### 4.7 후반 반복 루프 억제: `late_loop_penalty`

토론이 길어질수록 후반 라운드에서 같은 프레임을 반복하는 문제가 있었다. 그래서 4라운드 이후에는 더 엄격한 반복 패널티를 추가했다.

```yaml
late_loop_penalty:
  late_start: 4
  tau: 0.18
  gamma: 2.5
```

개념적으로는 다음과 같다.

```text
round < 4 이거나 J < 0.18이면 패널티 없음
round >= 4 이고 J >= 0.18이면 -2.5 * (J - 0.18)
```

### 4.8 토론 전개: `argument_advancement`

`argument_advancement`는 현재 답변이 자기 과거 발언에 비해 새로운 논점이나 새 내용어를 포함하는지 본다.

단, 완전히 엉뚱한 새 이야기를 하는 것을 막기 위해 상대 논점 coverage로 gate를 건다.

```text
new_ratio = |terms(y) - terms(H_self)| / |terms(y)|
gate = coverage_floor + (1 - coverage_floor) * coverage(opponent, y)
R_argument_advancement = new_ratio * gate
```

설정값은 다음과 같다.

```yaml
argument_advancement:
  min_round: 2
  coverage_floor: 0.25
```

### 4.9 근거 충실도: `evidence_grounding`

`evidence_grounding`은 현재 0으로 비활성화했다.

이유는 아직 RAG 검색, citation matching, NLI 기반 factuality 검증 파이프라인이 붙지 않았기 때문이다. 근거 검색 없이 evidence reward를 켜면 reward hacking 가능성이 커지므로 이번 학습에서는 제외했다.

## 5. GRPO 학습 방식

### 5.1 GRPO 개념

GRPO는 하나의 prompt에 대해 여러 개의 completion을 생성한 뒤, 같은 그룹 안에서 보상이 높은 completion과 낮은 completion을 비교해 정책을 업데이트한다.

이번 학습에서는 다음 흐름으로 수행했다.

```text
1. 토론 transcript에서 특정 side의 턴을 prompt로 구성
2. 현재 policy가 같은 prompt에 대해 G개 답변 생성
3. 각 답변에 v3 reward 계산
4. 그룹 내부 상대 advantage 계산
5. KL penalty로 SFT/base 분포에서 과도하게 멀어지는 것을 억제
6. LoRA adapter만 업데이트
```

이번 학습은 LoRA/PEFT 기반이므로 base model 전체를 업데이트하지 않고 adapter만 업데이트했다.

### 5.2 데이터

학습 transcript는 다음 파일을 사용했다.

```text
rl/data/eval_sft.jsonl
```

구성은 다음과 같았다.

| 항목 | 값 |
|---|---:|
| 전체 row | 64 |
| LEFT turn | 32 |
| RIGHT turn | 32 |
| 라운드 | 1~8 |
| 주제 수 | 4 |

prompt 길이는 후반 라운드로 갈수록 길어져서, RIGHT 쪽이 LEFT보다 약간 더 긴 편이었다. 이 때문에 RIGHT는 group size를 LEFT보다 하나 낮게 설정했다.

### 5.3 공통 학습 설정

| 항목 | 값 |
|---|---|
| precision | 4bit NF4 |
| optimizer 대상 | LoRA adapter |
| reward version | v3 |
| judge backend | none |
| max steps | 128 |
| max completion length | 96 |
| learning rate | 5e-6 |
| gradient accumulation | 1 |
| save steps | 8 |
| save total limit | 3 |
| GPU | A100-SXM4-80GB 1장 |
| vLLM | 사용하지 않음 |

vLLM은 학습에 필수는 아니다. 이번 작업에서는 `kanana-2-30b-a3b-instruct` 모델을 transformers로 직접 로드해서 4bit로 학습했다. vLLM은 생성 가속용 선택지일 뿐이며, 이번 최종 장기 학습은 no-vLLM 방식으로 수행했다.

### 5.4 GPU 사용량 최적화

처음에는 group size를 더 크게 잡아 GPU를 더 쓰려고 했다. 실제로 다음 설정을 시도했다.

| side | 시도 | 결과 |
|---|---|---|
| LEFT | G=12, completion=96 | OOM 발생 |
| LEFT | G=10, completion=96 | 80GB에 너무 근접해 장기 학습 위험 |
| LEFT | G=9, completion=96 | 안정적으로 128 steps 완료 |
| RIGHT | G=9, completion=96 | RIGHT prompt가 더 길어 OOM 위험 |
| RIGHT | G=8, completion=96 | 안정적으로 128 steps 완료 |

따라서 최종 설정은 다음과 같다.

| side | group size | generation batch size | 이유 |
|---|---:|---:|---|
| LEFT | 9 | 9 | 단일 A100 80GB에서 안정적으로 가능한 상한 |
| RIGHT | 8 | 8 | RIGHT prompt가 더 길어 한 단계 낮춤 |

### 5.5 실제 실행 명령

LEFT:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/tmp/kb_test_deps:/root/kb/.venvs/llm_debate_eval/lib/python3.10/site-packages python3 rl/train_grpo.py   --side left   --precision 4bit   --reward-version v3   --output-suffix grpo_v3_left_long_g9   --init-adapter adapters/left   --gpu 0   --transcripts rl/data/eval_sft.jsonl   --max-steps 128   --save-steps 8   --group-size 9   --generation-batch-size 9   --gradient-accumulation-steps 1   --max-completion-length 96   --learning-rate 5e-6
```

RIGHT:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/tmp/kb_test_deps:/root/kb/.venvs/llm_debate_eval/lib/python3.10/site-packages python3 rl/train_grpo.py   --side right   --precision 4bit   --reward-version v3   --output-suffix grpo_v3_right_long_g8   --init-adapter adapters/right   --gpu 0   --transcripts rl/data/eval_sft.jsonl   --max-steps 128   --save-steps 8   --group-size 8   --generation-batch-size 8   --gradient-accumulation-steps 1   --max-completion-length 96   --learning-rate 5e-6
```

## 6. GRPO 학습 결과

### 6.1 LEFT 학습 결과

| 항목 | 값 |
|---|---:|
| 총 step | 128 |
| runtime | 약 6시간 42분 |
| group size | 9 |
| final reward, step 128 | 1.8954 |
| final KL, step 128 | 0.00733 |
| 최종 adapter | `adapters/left_grpo_v3_left_long_g9` |
| 최종 적용 checkpoint | `checkpoint-112` |

LEFT에서 보존된 주요 checkpoint 비교는 다음과 같다.

| step | reward | KL | stance | direct_rebuttal | issue_coverage | advancement | lexical_penalty | late_penalty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 112 | 2.1833 | 0.00730 | 0.6491 | 0.8826 | 0.8194 | 0.4056 | -0.2506 | 0.0000 |
| 120 | 1.4961 | 0.00762 | 0.6126 | 0.5069 | 0.3611 | 0.1751 | -0.0292 | 0.0000 |
| 128 | 1.8954 | 0.00733 | 0.6858 | 0.7653 | 0.6389 | 0.1262 | -0.0056 | -0.0952 |

최종 128 step까지 학습은 완료했지만, 보존된 checkpoint 중 실제 토론 품질 보상 균형은 `checkpoint-112`가 가장 좋았다. 특히 direct rebuttal, issue coverage, argument advancement가 높았다. 따라서 실제 적용 경로는 `checkpoint-112`로 설정했다.

### 6.2 RIGHT 학습 결과

| 항목 | 값 |
|---|---:|
| 총 step | 128 |
| runtime | 약 6시간 32분 |
| group size | 8 |
| final reward, step 128 | 1.8176 |
| final KL, step 128 | 0.00445 |
| 최종 adapter | `adapters/right_grpo_v3_right_long_g8` |
| 최종 적용 checkpoint | `checkpoint-120` |

RIGHT에서 보존된 주요 checkpoint 비교는 다음과 같다.

| step | reward | KL | stance | direct_rebuttal | issue_coverage | advancement | lexical_penalty | late_penalty |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 112 | 1.7576 | 0.01025 | 0.5929 | 0.5164 | 0.3906 | 0.3119 | -0.0108 | 0.0000 |
| 120 | 2.0336 | 0.00515 | 0.6689 | 0.7430 | 0.6719 | 0.2810 | -0.0283 | 0.0000 |
| 128 | 1.8176 | 0.00445 | 0.7011 | 0.7563 | 0.6250 | 0.1204 | -0.0127 | -0.1522 |

RIGHT는 `checkpoint-120`이 보존된 checkpoint 중 가장 균형이 좋았다. stance, direct rebuttal, issue coverage, advancement가 모두 준수했고 KL도 안정적이었다. 따라서 실제 적용 경로는 `checkpoint-120`으로 설정했다.

### 6.3 학습 결과 해석

- GRPO는 SFT adapter에서 출발했기 때문에 기존 LEFT/RIGHT 토론 페르소나를 완전히 잃지 않았다.
- KL 값이 0.004~0.010 수준으로 유지되어, SFT 분포에서 과도하게 멀어지지는 않았다.
- v3 보상은 특히 `direct_rebuttal`, `issue_coverage`, `argument_advancement`를 명시적으로 로그에 남기기 때문에 어떤 방향으로 개선되는지 추적하기 쉽다.
- 후반 반복에 대해서는 `lexical_anti_repetition`과 `late_loop_penalty`가 실제로 음수 보상으로 작동했다.
- RIGHT에서는 일부 짧은 종료 답변이 관측되었다. 2차 개선 시에는 `substance_length_floor` 또는 논거 밀도 보상을 추가하는 것이 좋다.

## 7. GovReport 평가

### 7.1 평가 목적

GRPO의 직접 목표는 토론 성향 유지, 반박 품질, 반복 억제다. GovReport 평가는 이 강화학습이 일반적인 장문 정책 보고서 요약 능력을 크게 망가뜨렸는지 확인하기 위한 보조 평가다.

중요한 한계는 GovReport가 영어 데이터라는 점이다. 따라서 이 결과를 “한국어 요약 성능”이라고 해석하면 안 된다. 더 정확한 표현은 다음과 같다.

```text
GovReport English long-document government report summarization transfer evaluation
```

즉, 영어 장문 정부 보고서 요약에 대한 cross-domain/cross-lingual 보존 평가로 보는 것이 적절하다.

### 7.2 평가 데이터

- 데이터셋: GovReport
- split: official test split
- 샘플 수: 20개
- seed: 42
- 구성: CRS/GAO가 섞이도록 균형 샘플링
- 전체 test set을 모두 평가하지 않은 이유: Kanana 30B 4bit transformers 추론이 문서당 약 2분 정도 걸리므로, 전체 973개 test 문서를 평가하면 며칠 단위가 걸림

GovReport 데이터 구조는 CRS와 GAO가 다르므로 각각 다르게 처리했다.

| source | 입력 문서 | reference summary |
|---|---|---|
| CRS | `reports` section tree를 펼친 텍스트 | `summary` |
| GAO | `report` section list를 펼친 텍스트 | `highlight` |

### 7.3 Prompt

모든 variant에 동일한 영어 prompt를 사용했다.

```text
Summarize the following U.S. government report in English.
Focus on the report purpose, major findings, evidence, and policy implications.
Keep the summary factual, concise, and understandable to an informed citizen.

Title: ...
Report ID: ...

Report text:
...

Summary:
```

### 7.4 긴 문서 처리

GovReport 문서는 매우 길기 때문에 전체 문서를 그대로 넣지 않았다.

- `max_input_tokens = 3072`
- tokenizer 기준으로 문서를 자름
- 앞부분과 뒷부분을 절반씩 유지
- 중간 생략 note를 추가

따라서 이번 평가는 full-context GovReport 평가가 아니라 다음 조건의 평가다.

```text
GovReport sampled test set, n=20,
head+tail truncation,
max input 3072 tokens,
max output 192 tokens
```

### 7.5 생성 조건

모든 variant에 동일한 decoding 설정을 적용했다.

| 항목 | 값 |
|---|---:|
| max_new_tokens | 192 |
| temperature | 0 |
| top_p | 0.9 |
| repetition_penalty | 1.05 |
| precision | 4bit |

`temperature=0`으로 deterministic decoding을 사용했기 때문에 같은 환경에서는 결과 재현성이 높다.

### 7.6 평가 variant

| variant | 설명 |
|---|---|
| `base` | Kanana base model, adapter 없음 |
| `sft_left` | `adapters/left` |
| `sft_right` | `adapters/right` |
| `sft_macro` | `sft_left`, `sft_right` 평균 |
| `grpo_left` | `adapters/left_grpo_v3_left_long_g9/checkpoint-112` |
| `grpo_right` | `adapters/right_grpo_v3_right_long_g8/checkpoint-120` |
| `grpo_macro` | `grpo_left`, `grpo_right` 평균 |

### 7.7 평가지표

이번 평가는 외부 `rouge_score` 패키지가 아니라 `eval/govreport_eval.py` 안에 구현한 lightweight ROUGE를 사용했다. 따라서 최종 논문/보고서에는 “in-house ROUGE-style overlap metric”이라고 쓰는 것이 정확하다.

| 지표 | 의미 |
|---|---|
| ROUGE-1 F1 | 모델 요약과 reference 요약 사이의 단어 1개 단위 overlap. 핵심 키워드 포함 정도에 가깝다. |
| ROUGE-2 F1 | 연속 2단어 overlap. 핵심 표현/구 단위 일치도를 본다. 일반적으로 가장 낮게 나온다. |
| ROUGE-L F1 | Longest Common Subsequence 기반. 요약의 단어 순서와 흐름 유사도를 본다. |
| words | 모델이 생성한 요약문의 평균 영어 단어 수. |
| compression | 생성 요약 단어 수 / 원문 문서 단어 수. 요약 압축률. |
| bigram_repetition | `1 - 고유 bigram 수 / 전체 bigram 수`. 반복 표현이 많을수록 높다. |

### 7.8 최종 GovReport 평가 결과

평가 산출물은 로컬 기준 다음 경로에 있다.

```text
eval_outputs/govreport/test_n20_all/report.md
eval_outputs/govreport/test_n20_all/metrics.json
eval_outputs/govreport/test_n20_all/predictions.jsonl
```

단, `eval_outputs/`는 용량 및 재생성 가능성 때문에 `.gitignore`에 포함했다. GitHub에는 이 문서의 요약 표만 포함한다.

최종 결과는 다음과 같다.

| variant | n | ROUGE-1 | ROUGE-2 | ROUGE-L | words | compression | bigram repetition |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 20 | 0.2251 | 0.0808 | 0.1315 | 131.5 | 0.0206 | 0.0577 |
| sft_left | 20 | 0.2312 | 0.0800 | 0.1323 | 136.7 | 0.0214 | 0.0542 |
| sft_right | 20 | 0.2306 | 0.0812 | 0.1315 | 135.3 | 0.0211 | 0.0601 |
| sft_macro | 40 | 0.2309 | 0.0806 | 0.1319 | 136.0 | 0.0213 | 0.0572 |
| grpo_left | 20 | 0.2284 | 0.0791 | 0.1327 | 137.0 | 0.0214 | 0.0582 |
| grpo_right | 20 | 0.2310 | 0.0825 | 0.1318 | 136.3 | 0.0214 | 0.0579 |
| grpo_macro | 40 | 0.2297 | 0.0808 | 0.1323 | 136.6 | 0.0214 | 0.0580 |

### 7.9 평가 결과 해석

요약하면 다음과 같다.

| 비교 | 해석 |
|---|---|
| base -> SFT | ROUGE-1이 `0.2251 -> 0.2309`로 소폭 상승했다. SFT가 영어 GovReport 요약 overlap을 약간 개선하거나 적어도 유지했다. |
| SFT -> GRPO | ROUGE-1은 `0.2309 -> 0.2297`로 아주 조금 낮아졌고, ROUGE-2는 `0.0806 -> 0.0808`, ROUGE-L은 `0.1319 -> 0.1323`으로 거의 동일하거나 소폭 상승했다. |
| GRPO vs base | GRPO macro는 ROUGE-1과 ROUGE-L에서 base보다 높고, ROUGE-2는 거의 동일하다. |
| repetition | bigram repetition은 base 0.0577, SFT macro 0.0572, GRPO macro 0.0580으로 큰 차이가 없다. |

따라서 이번 GovReport 평가에서 가장 안전한 결론은 다음이다.

```text
SFT는 base 대비 영어 장문 정부 보고서 요약 overlap을 소폭 개선했다.
GRPO는 토론 보상 최적화를 수행했음에도 GovReport 요약 성능을 크게 손상시키지 않았고, SFT 수준을 거의 유지했다.
```

GRPO의 목적은 GovReport 요약 최적화가 아니라 토론 성향/반박/반복 억제 강화였기 때문에, GovReport에서 큰 상승이 없다고 해서 GRPO가 실패했다고 해석하면 안 된다. 오히려 토론 품질 보상을 추가한 뒤에도 일반 장문 요약 지표가 크게 무너지지 않은 점이 중요하다.

## 8. 재현 방법

### 8.1 보상 sanity check

```bash
PYTHONPATH=/tmp/kb_test_deps:/root/kb/.venvs/llm_debate_eval/lib/python3.10/site-packages python3 rl/check_reward_sanity.py
```

확인된 결과는 다음과 같다.

```text
[PASS] R_direct_rebuttal: 정면반박 샘플 > 회피 샘플
[PASS] R_argument_advancement: 새 논점 전개 > 자기 반복
[PASS] R_late_loop_penalty: 5라운드 반복 발언 패널티
모든 항목 PASS
```

### 8.2 GovReport 평가

GovReport 평가는 다음 스크립트로 수행했다.

```text
eval/govreport_eval.py
```

base, SFT, GRPO를 한 번에 돌릴 때는 다음 명령을 사용한다.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=/tmp/kb_test_deps:/root/kb/.venvs/llm_debate_eval/lib/python3.10/site-packages python3 eval/govreport_eval.py   --limit 20   --variants base,sft_left,sft_right,grpo_left,grpo_right   --run-name test_n20_all   --max-input-tokens 3072   --max-new-tokens 192   --temperature 0   --local-files-only   --gpu 0
```

중간에 중단되면 `--resume`을 붙여 이어서 실행할 수 있다.

```bash
python3 eval/govreport_eval.py   --limit 20   --variants sft_right   --run-name test_n20_all   --max-input-tokens 3072   --max-new-tokens 192   --temperature 0   --local-files-only   --gpu 0   --resume
```

## 9. GitHub 업로드 시 주의사항

### 9.1 브랜치

현재 작업용 브랜치는 다음 이름으로 정리했다.

```text
kabeen
```

기존 `jaeeun` 문자열은 작업 디렉터리와 문서에서 제거했다.

### 9.2 대용량 파일

다음 경로는 `.gitignore`에 포함했다.

```text
adapters/
eval_data/
eval_outputs/
rl/data/
sft/data/
```

이유는 다음과 같다.

- `adapters/`: LoRA adapter 가중치가 커서 GitHub 일반 push 대상이 아님
- `eval_data/`: GovReport 원본 및 추출 데이터가 큼
- `eval_outputs/`: predictions와 결과 파일은 재생성 가능
- `rl/data/`, `sft/data/`: 학습/rollout 데이터는 재생성 가능하고 용량이 커질 수 있음

GitHub에는 코드, 설정, 보상 구현, 평가 스크립트, 문서만 올리는 것이 안전하다.

### 9.3 adapter 공유 방식

adapter를 팀원과 공유해야 한다면 GitHub repo에 직접 넣기보다 다음 중 하나를 권장한다.

1. Hugging Face Hub model repo
2. 별도 cloud storage
3. release asset
4. 사내/팀 공유 스토리지

현재 문서에는 adapter 경로와 학습 설정을 기록했으므로, 같은 환경에서는 재학습 또는 별도 adapter 다운로드 후 동일 경로에 배치해 사용할 수 있다.

## 10. 남은 개선 과제

1. **한국어 정책 문서 평가 추가**
   - GovReport는 영어 데이터이므로 한국어 정책 토론 모델의 메인 평가로는 한계가 있다.
   - AI Hub 문서요약, 국회입법조사처/KDI/국회예산정책처 보고서 기반 한국어 평가셋을 추가하면 더 설득력 있다.

2. **공식 ROUGE 패키지로 재평가**
   - 현재는 in-house lightweight ROUGE다.
   - 최종 보고서용 숫자는 `rouge_score` 또는 `evaluate` 기반으로 다시 산출하는 것이 좋다.

3. **BGE-m3 semantic similarity 평가 추가**
   - 한국어/영어 모두 n-gram ROUGE만으로는 의미적 유사도를 충분히 반영하기 어렵다.
   - BGE-m3 embedding cosine similarity를 함께 넣으면 요약 의미 보존 평가가 더 안정적이다.

4. **LLM-as-a-Judge 평가 추가**
   - 논점 반영, 사실성, 시민 친화성, 정책적 함의 설명력을 1~5점으로 평가할 수 있다.
   - 다만 외부 API를 쓰면 비용과 재현성 이슈가 생기므로 별도 표로 분리하는 것이 좋다.

5. **2차 GRPO 개선**
   - RIGHT 일부 샘플에서 짧은 종료 답변이 관측되었다.
   - `substance_length_floor`, `claim_evidence_density`, `minimum_argument_units` 같은 보상을 추가하면 토론 답변 밀도를 더 높일 수 있다.

## 11. 최종 결론

이번 작업의 핵심 결론은 다음과 같다.

1. Kanana-2-30B-A3B-Instruct 기반 LEFT/RIGHT SFT adapter에서 출발해 GRPO v3 장기 학습을 완료했다.
2. 원래 제안한 성향 일관성, 반박 품질, 자기 반복 억제 보상식을 유지하면서, 실제 multi-agent 토론에 맞게 `stance_alignment`, `issue_coverage`, `direct_rebuttal`, `argument_advancement`, `late_loop_penalty` 등으로 확장했다.
3. 최종 적용 adapter는 LEFT `checkpoint-112`, RIGHT `checkpoint-120`이다.
4. GovReport n=20 sampled evaluation에서 SFT는 base 대비 ROUGE-1을 소폭 개선했고, GRPO는 SFT 수준의 요약 성능을 거의 유지했다.
5. 따라서 GRPO는 영어 장문 요약 능력을 크게 손상시키지 않으면서, 토론 보상 방향으로 policy를 조정한 것으로 해석할 수 있다.
