# 보상 설계 v2 — 논문 리서치 기반 재설계

> 원안: [`보상 설계.pdf`](../../보상%20설계.pdf) (강수빈, 성향/반박품질/반복패널티 3종 + GRPO)
> 방향성 근거: [`투빅스_컨퍼런스 중간발표_NLP.pdf`](../../투빅스_컨퍼런스%20중간발표_NLP.pdf) 03. 프로젝트 설계
> 이 문서는 코드에서 `config/reward.yaml`의 `version: v2`로 선택해 쓸 수 있다 (`version: v1`이면 PDF 원안 그대로 사용). 구현은 [`rl/rewards/`](../rl/rewards).
>
> **base model 무관 설계.** 이 문서를 작성할 당시 LEFT/RIGHT의 base model은 Qwen2.5-14B-Instruct였고,
> 이후 2026-07-02에 Kanana-2-30B-A3B-Instruct로 교체되었다(`config/model.yaml`, `CLAUDE.md` 참고).
> 보상 설계 자체(§2~§4)는 어떤 base model을 debate에 쓰든 동일하게 적용된다 — 임베딩/키워드/Jaccard
> 기반 컴포넌트는 텍스트만 보고, judge 모델(§7 LocalJudge 예시의 Qwen2.5-7B)도 LEFT/RIGHT와 독립적으로
> 교체 가능하다. GRPO 학습 대상(`rl/train_grpo.py`)만 Kanana 기준 SFT 어댑터를 이어받도록 하면 된다.

## 1. 왜 다시 설계하는가

원안(v1)은 GRPO에 필요한 스칼라 보상을 정의했다는 점에서 출발점으로 유효하지만, 두 가지 지점에서 중간발표가 정리한 MAD 실패 유형(슬라이드 10~13) 및 최근 논문들의 지적과 어긋난다.

1. **성향 점수(`S(y)`)를 그대로 보상으로 쓰면 "극단적일수록 유리"해진다.** LEFT는 `-S(y)`, RIGHT는 `S(y)`를 직접 보상으로 받으므로, 모델이 실제 논증 품질과 무관하게 더 극단적인 어휘를 쓰도록 reward hacking할 유인이 생긴다. `PReSS`(정치 스탠스 안정성 평가, arXiv:2504.17052)와 `Generative Exaggeration in LLM Social Agents`(arXiv:2507.00657)는 페르소나 프롬프팅이 모델의 내재 편향과 충돌할 때 응답이 과장되거나 불안정해짐을 보였다. 우리가 막으려는 것은 "중립으로의 회귀"이지 "논증 없는 극단화"가 아니다 — 이 둘을 같은 보상으로 섞으면 안 된다.
2. **보상 계산을 전부 "외부 LLM Judge"에 의존한다.** 중간발표 슬라이드 10, 12가 지적한 **Judge Ceiling**(Judge 능력이 시스템 천장, Judge가 약하면 토론도 약함) 문제를 그대로 GRPO 학습 신호에 들여오게 된다. `Judging with Many Minds`(arXiv:2505.19477)는 멀티에이전트 판정에서도 같은 backbone을 공유하면 편향이 사라지지 않고 오히려 증폭될 수 있음을 보였고, "The Confident Liar"(arXiv:2606.10296)는 LLM judge가 논리적으로 틀린 자신감 있는 답변에 속기 쉬움을 지적한다. Judge 하나에 보상 전체가 걸리는 구조는 취약하다.

v2는 같은 3가지 문제의식(성향 유지, 반박 품질, 반복 억제)을 계승하되, **가능한 구성 요소는 judge-free(임베딩/어휘 매칭)로 옮기고, judge는 보조 신호로만 남긴다.** 그리고 슬라이드 13의 표에 나온 R1(다양성)·R2(근거 충실도)·R3(페르소나 일관성) 설계를 구체적인 수식으로 채운다.

## 2. 문제 → 보상 매핑 (슬라이드 13 확장)

| 실패 유형 | 관련 연구 | v1 원안의 한계 | v2 대응 보상 |
|---|---|---|---|
| Sycophancy (동조) | Peacemaker or Troublemaker (arXiv:2509.23055), The Silicon Mirror (arXiv:2604.00478), Too Polite to Disagree (arXiv:2604.02668) | 없음 — 반박 품질(R_rebuttal)이 간접적으로만 완화 | **R_persona** — 상대 쪽으로의 스탠스 이동(Δ)을 직접 패널티 |
| Echo Chamber / DoT (조기 수렴) | Diversity Collapse in Multi-Agent LLM Systems (arXiv:2604.18005), Encouraging Divergent Thinking (arXiv:2305.19118), Estornell & Liu "tyranny of the majority" (관련: Debate or Vote arXiv:2508.17536) | R_antirep가 자기 반복만 봄. 상대와 수렴하는 것은 못 잡음 | **R_diversity** — 상대 발언·자기 과거 발언 모두와의 임베딩 유사도 패널티 |
| 근거 없는 주장(Hallucination) 전파 | DebateSum(ACL 2020 argmining), VERI-DPO 스타일 claim verification (arXiv:2603.10494) | 없음 | **R_grounding** — 근거문서 대비 주장 검증 (RAG 연동 전까지는 optional, weight=0) |
| 논점 회피 | ArgKP/KPA (ACL 2021 argmining), CONSENSAGENT (ACL 2025 findings) | 1~5점 LLM judge 단일 신호 | **R_engagement** — 상대 핵심 논점(key point) 어휘 커버리지 + judge는 보조 |
| 문자 그대로의 자기 반복 | — | Jaccard 자체는 유효 | **R_novelty** — v1의 Jaccard 패널티를 경량 안전망으로 유지 |
| Judge Ceiling | Judging with Many Minds (arXiv:2505.19477), The Confident Liar (arXiv:2606.10296) | 전 보상이 judge 1개에 의존 | GRPO 보상에서는 **judge 의존도를 최소화**(위 4개 중 judge가 필요한 건 R_engagement의 보조 신호뿐). Judge Ceiling 자체의 정공법은 슬라이드대로 **Synthesizer 분리 DPO**(GRPO 범위 밖) |
| Tyranny of Majority / Voting Collapse | Debate or Vote (arXiv:2508.17536), The Consensus Trap (arXiv:2604.17139) | 해당 없음 (우리 시스템은 2-agent 대립구조, 다수결 없음) | 아키텍처 차원에서 "최종 Voting 안함"으로 이미 해결(슬라이드 13). 보상 설계로 재해결하지 않음 |
| Belief Entrenchment | From Belief Entrenchment to Robust Reasoning (arXiv:2503.16814), DReaMAD | 슬라이드에서 "직접 다루지 않음"으로 명시 | v2도 정면 대응은 범위 밖. 단, R_diversity(상대 발언과의 유사도 패널티)가 "공유된 오류를 그대로 반복 지지"하는 정도를 부분적으로 억제하는 부산물 효과는 있음 — 정식 해결책 아님을 §5에 명시 |

## 3. 보상 구성 요소

아래 4개가 기본 활성 컴포넌트다 (`R_grounding`은 근거문서/RAG가 없는 현재 시스템에서는 기본 비활성).

### 3.1 R_persona — 페르소나 일관성 / 항(抗)동조 보상

**막으려는 것:** 라운드가 진행될수록 base model의 중립 성향으로 회귀하는 것(슬라이드 7, 16의 Default Bias) *그리고* 상대 논증에 설득되어 입장이 흔들리는 것(sycophancy). v1의 `S(y)` 직접보상 방식은 "얼마나 진보/보수적인 단어를 썼는가"만 측정해 후자를 못 잡고 전자도 극단화로 왜곡될 수 있다.

임베딩 모델 `E(·)` (judge 불필요, 기본 `BAAI/bge-m3` — 초안의 ko-sroberta는 max_seq_length 128로
600토큰 발언이 앞부분만 잘리는 문제가 있어 2026-07-04 교체)로 각 발언을 벡터화한다.

- `anchor` = 해당 side의 SFT 단계 seed 응답 **임베딩 평균** (페르소나의 "기준점",
  `rl/build_anchor.py`가 `sft/data/{side}_train.jsonl`에서 샘플을 뽑아 `rl/data/anchors/{side}.json`으로
  저장하고, 학습 시 샘플별 임베딩을 평균해 1회 계산·캐싱. 텍스트를 이어붙여 한 번에 임베딩하면
  모델 max_seq_length에서 잘리므로 반드시 샘플별 임베딩 후 평균할 것)
- `drift_i = cos_sim(E(y_i), anchor)` — 기준 페르소나에서 얼마나 멀어졌는가
- `sycophancy_i = cos_sim(E(y_i), E(opponent_now)) - cos_sim(E(y_{i-1}), E(opponent_now))` —
  직전 자기 발언 대비, 이번 발언이 **상대의 이번 주장** 쪽으로 얼마나 더 가까워졌는가
  (초안은 `E(opponent_prev)` 기준이었으나 "상대의 이번 주장에 흔들렸는가"를 재는 데는
  현재 상대 발언 기준이 더 직접적이라 구현과 함께 이 정의로 통일)

```
coverage_i     = |KP(opponent) ∩ terms(y_i)| / |KP(opponent)|      # §3.2와 동일
gate_i         = (1 - coverage_i)                                   # syc_coverage_gate=true일 때
R_persona(y_i) = clip(drift_i, 0, 1) - λ_syc · max(0, sycophancy_i) · gate_i
```

`λ_syc` (기본 1.0)는 동조 패널티 강도. `drift_i`는 [0,1]로 clip해 "기준점에서 너무 멀어지지 않았는가"만 보상하고, 더 멀어질수록 추가 보상을 주지 않는다 (v1처럼 극단화를 계속 밀어주지 않음).

`gate_i`(coverage 게이트, 2026-07-04 추가)는 초안에서 "R_engagement가 낮으면서 sycophancy가
양수인 경우로 교차 검증"이라고만 적고 구현하지 않았던 부분의 실제 구현이다: 상대 논점을 정면으로
다루면서 가까워진 것(반박·교전, coverage↑)은 페널티를 완화하고, 논점을 회피하며 가까워진 것(동조)만
강하게 처벌한다. 상대 쪽으로의 모든 스탠스 수렴이 동조는 아니며 정당한 설득·정정이 섞여 있다는
지적(*Not All Flips Are Conformity: Decomposing Stance Convergence in Multi-Agent LLM Debate*,
arXiv:2606.00820)에 대한 대응이기도 하다. `config/reward.yaml`의 `persona.syc_coverage_gate`로 끌 수 있다.

> 선택적 보조 신호: judge 기반 `S(y) ∈ [-1,1]`을 `0.2` 가중치로 더할 수 있음 (`config/reward.yaml`의 `persona.judge_aux_weight`). 기본값 0 — judge 없이도 동작.

### 3.2 R_engagement — 반박 참여도 (논점 회피 방지)

**막으려는 것:** 상대 논점을 정면으로 다루지 않고 자기 할 말만 하는 것. v1은 이를 전적으로 judge 1~5점에 위임했다(Judge Ceiling 리스크).

1차 신호는 **key-point 커버리지**: `ArgKP/KPA`(ACL 2021 argmining-1.16) 방식으로, 상대 발언에서 핵심 명사구/논점 키워드 집합 `KP(opponent)`를 추출(형태소 분석기 + TF-IDF 상위 n-gram, judge 불필요)하고, 현재 응답이 그 키워드들을 실제로 언급·반박하는지 어휘 중첩으로 측정한다.

```
coverage_i = |KP(opponent) ∩ terms(y_i)| / |KP(opponent)|
Rengagement_judge_free(y_i) = coverage_i          # ∈ [0,1]
```

이것만으로는 "키워드만 나열하고 반박은 안 함"을 못 잡으므로, 경량 judge(로컬 7B급 또는 API)로 1~5점 정면반박 평가를 보조 신호로 유지한다 — 단 **v1처럼 유일한 신호가 아니라 평균**으로 섞는다.

```
Q(y_i) ∈ [1,5]  (judge 평가, v1과 동일 프롬프트 재사용)
R_engagement(y_i) = 0.6 · coverage_i + 0.4 · (Q(y_i) - 1) / 4
```

judge를 못 쓰는 환경(judge=disabled)에서는 `coverage_i` 단독으로 자동 fallback (`config/reward.yaml`의 `engagement.judge_weight=0`).

### 3.3 R_diversity — 다양성 / 반-에코챔버 보상 (슬라이드 13의 R1)

**막으려는 것:** Round를 거치며 두 agent가 같은 말을 반복하거나 서로 수렴하는 것(Echo Chamber, DoT). `Diversity Collapse in Multi-Agent LLM Systems`(arXiv:2604.18005)가 보고한 "Artificial Hivemind" 현상, `Encouraging Divergent Thinking`(arXiv:2305.19118)의 "tit-for-tat" 전제(서로 다른 관점이 유지되어야 교정 효과가 생김)를 근거로 한다. Judge 불필요, 임베딩만 사용.

```
sim_opponent_i = cos_sim(E(y_i), E(opponent_response))
sim_self_i     = max_{j<i, same side} cos_sim(E(y_i), E(y_j))   # 자기 과거 발언과의 최대 유사도

R_diversity(y_i) = 1 - max(sim_opponent_i, sim_self_i)
```

상대와 너무 비슷해지는 것(조기 합의/동조성 수렴)과 자기 자신을 semantic하게 반복하는 것 모두를 하나의 보상으로 잡는다. v1의 Jaccard 반복 패널티는 "정확히 같은 단어의 반복"만 잡고 "다른 단어로 같은 말을 반복"하는 것은 못 잡았는데, 임베딩 기반은 이를 보완한다.

### 3.4 R_novelty — 표면적 반복 억제 (v1 계승, 안전망)

v1의 `R_antirep`을 그대로 유지한다. 임베딩 모델이 없거나 실패하는 상황에서도 동작하는 값싼 안전망이며, R_diversity(의미 기반)와 상호보완적이다 (완전히 다른 단어로 새 문장처럼 보이지만 사실상 같은 주장을 반복하는 경우는 R_diversity가, 같은 문구를 그대로 복붙하는 경우는 R_novelty가 잡는다).

```
J(y_i, H) ∈ [0,1]   # 과거 자기 발언 히스토리와의 단어 단위 Jaccard 유사도
τ = 0.25, γ = 1.5   # v1과 동일 기본값

R_novelty(y_i) = 0                         if J(y_i, H) < τ
R_novelty(y_i) = -γ · (J(y_i, H) - τ)      if J(y_i, H) >= τ
```

### 3.5 R_grounding — 근거 충실도 (기본 비활성, RAG 연동 후 사용)

`[0627]평가데이터셋_v2`에서 제시된 `DebateSum`(evidence-claim 구조) 및 목업의 "근거문서" 패널은, 이 시스템이 궁극적으로 라운드마다 근거 문서를 검색해 agent에게 제공하는 방향임을 시사한다. 근거 검색(RAG)이 아직 파이프라인에 없으므로, R_grounding은 인터페이스만 정의하고 기본 가중치 0으로 둔다.

```
# evidence 문서가 주어졌을 때만 계산 (context["evidence"] 존재 시)
entail_i = NLI_entailment_score(premise=evidence, hypothesis=claims(y_i))
R_grounding(y_i) = entail_i   # ∈ [0,1], NLI 모델 사용 — LLM judge 불필요
```

`VERI-DPO`(arXiv:2603.10494)가 임상 요약에서 쓴 claim-verification 패턴을 참고했다. 가벼운 NLI 모델(예: `xlm-roberta` 기반 다국어 NLI)로 계산 가능해 judge 의존이 없다. RAG 파이프라인이 생기면 `config/reward.yaml`에서 `grounding.weight`를 0보다 크게 주면 즉시 활성화된다.

## 4. 최종 통합 보상과 GRPO 목적함수

```
R_total(y_i) = w1·R_persona(y_i) + w2·R_engagement(y_i) + w3·R_diversity(y_i)
             + w4·R_novelty(y_i) + w5·R_grounding(y_i)
```

기본 가중치(`config/reward.yaml`): `w1=1.0, w2=1.0, w3=0.8, w4=0.5, w5=0.0`. R_novelty는 R_diversity와 목적이 겹치므로 과대평가되지 않도록 낮게, R_grounding은 비활성이므로 0.

GRPO 손실은 원안과 동일한 형태를 그대로 쓴다 (클리핑 + KL 페널티):

```
L_GRPO(θ) = (1/M) Σ_i [ min( r_i(θ)·A_i, clip(r_i(θ), 1-ε, 1+ε)·A_i ) ] − β·D_KL(π_θ ‖ π_ref)

r_i(θ) = π_θ(y_i|x) / π_θ_old(y_i|x)
A_i    = (R_total(y_i) − mean(R_total(y_1..G))) / std(R_total(y_1..G))     # 그룹 내 정규화
```

LoRA(PEFT) 기반 학습이므로 `π_ref`는 별도 모델 복제 없이 **어댑터를 비활성화한 base 모델**로 대체한다(TRL `GRPOTrainer`의 PEFT 표준 방식) — 3090 24GB 두 장이라는 VRAM 제약에서 ref 모델 복제 비용을 없앤다.

## 5. 명시적으로 다루지 않는 것 (Scope)

- **Judge Ceiling 자체의 근본 해결**: R_engagement의 보조 judge 신호를 줄였을 뿐, 최종 결정 단계의 Judge Ceiling은 슬라이드가 이미 지정한 대로 **Synthesizer 분리 + DPO**로 풀어야 한다. GRPO 보상 설계의 범위가 아니다.
- **Belief Entrenchment**: 슬라이드 13과 동일하게 이번 v2도 정면으로 다루지 않는다. R_diversity가 부수적으로 완화 효과를 줄 수 있으나, `From Belief Entrenchment to Robust Reasoning`(arXiv:2503.16814)이 제안하는 정식 개입(초기 belief 자체의 교정)은 별도 과제로 남긴다.
- **Tyranny of Majority**: 우리 시스템은 2-agent 대립 구조이고 최종 투표를 하지 않으므로(슬라이드 13) 해당 없음.
- **R_grounding 실사용**: RAG/근거 검색 파이프라인이 붙기 전까지는 가중치 0으로 비활성 상태 유지.

## 6. 코드 연동

- `config/reward.yaml` → `version: v1 | v2`, 컴포넌트별 가중치·하이퍼파라미터.
- `rl/rewards/v1_pdf.py` — `보상 설계.pdf` 원안 그대로 (stance/rebuttal/anti-rep).
- `rl/rewards/v2_redesign.py` — 본 문서의 5개 컴포넌트.
- `rl/rewards/composer.py` — `version`에 따라 v1/v2 컴포넌트를 로드해 가중합.
- 두 버전 모두 같은 `DebateTurnSample` 인터페이스를 구현하므로 `rl/train_grpo.py --reward-version v1|v2`로 즉시 전환 가능.

## 7. 참고 문헌

- Estornell & Liu, *Multi-LLM Debate: Framework, Principals, and Interventions* — tyranny of the majority. https://openreview.net/pdf?id=sy7eSEXdPC
- *Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?*, arXiv:2508.17536
- Liang et al., *Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate*, arXiv:2305.19118
- *Diversity Collapse in Multi-Agent LLM Systems: Structural Coupling and Collective Failure in Open-Ended Idea Generation*, arXiv:2604.18005
- *Peacemaker or Troublemaker: How Sycophancy Shapes Multi-Agent Debate*, arXiv:2509.23055
- *The Silicon Mirror: Dynamic Behavioral Gating for Anti-Sycophancy in LLM Agents*, arXiv:2604.00478
- *Too Polite to Disagree: Understanding Sycophancy Propagation in Multi-Agent Systems*, arXiv:2604.02668
- CONSENSAGENT, *Towards Efficient and Effective Consensus in Multi-Agent LLM Interactions Through Sycophancy Mitigation*, ACL 2025 Findings. https://aclanthology.org/2025.findings-acl.1141/
- *Judging with Many Minds: Do More Perspectives Mean Less Prejudice? On Bias Amplifications and Resistance in Multi-Agent Based LLM-as-Judge*, arXiv:2505.19477
- *The Confident Liar: Diagnosing Multi-Agent Debate with Log-Probabilities and LLM-as-Judge*, arXiv:2606.10296
- *From Belief Entrenchment to Robust Reasoning in LLM Agents*, arXiv:2503.16814
- Zhang, Park et al., *MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning*, ACL 2025 / arXiv:2502.18439
- *PReSS: A Black-Box Framework for Evaluating Political Stance Stability in LLMs via Argumentative Pressure*, arXiv:2504.17052
- *Generative Exaggeration in LLM Social Agents: Consistency, Bias, and Toxicity*, arXiv:2507.00657
- Bar-Haim et al., *Key Point Analysis (KPA) 2021 Shared Task*, ACL argmining-1.16. https://aclanthology.org/2021.argmining-1.16/
- *DebateSum*, ACL argmining-1.1 (2020). https://aclanthology.org/2020.argmining-1.1/
- *VERI-DPO: Evidence-Aware Alignment for Clinical Summarization via Claim Verification and Direct Preference Optimization*, arXiv:2603.10494
- Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, arXiv:2305.18290
