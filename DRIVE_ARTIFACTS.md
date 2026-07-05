# Google Drive Runtime Artifacts

이 저장소의 `kabeen` 브랜치는 코드와 가벼운 설정 파일을 GitHub에 두고, Git에서 제외한 대용량 실행 산출물은 Google Drive에 둔다.

Drive 폴더:

```text
https://drive.google.com/drive/folders/14chwR5azV67Xv2ZsMQglseFtDfLdv4gl
```

상위 공유 폴더:

```text
https://drive.google.com/drive/folders/1pCMROcO7hW5_xuf9uscbaA7y92RN3gbG
```

## 포함된 실행 산출물

Drive 하위 폴더 `llm_dabate_kabeen_runtime_artifacts_20260705` 안에는 다음 파일이 있다.

```text
README_runtime_artifacts.md
llm_dabate_kabeen_runtime_artifacts_20260705.tar.part-00
llm_dabate_kabeen_runtime_artifacts_20260705.tar.part-01
llm_dabate_kabeen_runtime_artifacts_20260705.tar.part-02
llm_dabate_kabeen_runtime_artifacts_20260705.tar.part-03
```

Part 파일 4개를 합치면 다음 runtime artifact tar가 된다.

```text
llm_dabate_kabeen_runtime_artifacts_20260705.tar
```

이 tar 안에는 GitHub clone만으로는 복원되지 않는 다음 경로가 들어 있다.

```text
adapters/left_grpo_v3_left_long_g9/checkpoint-112
adapters/right_grpo_v3_right_long_g8/checkpoint-120
rl/data/eval_sft.jsonl
rl/data/eval_grpo.jsonl
rl/data/eval_grpo_synth.jsonl
```

## 필요한 이유

`config/model.yaml`은 최종 GRPO v3 adapter를 아래 경로로 참조한다.

```yaml
left_adapter:  "adapters/left_grpo_v3_left_long_g9/checkpoint-112"
right_adapter: "adapters/right_grpo_v3_right_long_g8/checkpoint-120"
```

하지만 `.gitignore`에서 `adapters/`와 `rl/data/`를 제외하므로, 위 최종 adapter와 재학습용 데이터는 GitHub에 직접 올라가지 않는다.

## 복원 방법

GitHub 저장소를 clone한다.

```bash
git clone -b kabeen git@github.com:kangsoobin/llm_dabate.git
cd llm_dabate
```

Drive에서 part 파일 4개를 같은 로컬 폴더에 받은 뒤 합친다.

```bash
cat llm_dabate_kabeen_runtime_artifacts_20260705.tar.part-* > llm_dabate_kabeen_runtime_artifacts_20260705.tar
sha256sum llm_dabate_kabeen_runtime_artifacts_20260705.tar
```

원본 tar SHA256:

```text
6e7f890f0cb8599c8beed9dfadf38e4afa7f59ea95a10760a2cf6b2e59fe82e5  llm_dabate_kabeen_runtime_artifacts_20260705.tar
```

SHA256이 일치하면 저장소 루트에서 tar를 푼다.

```bash
tar -xf /path/to/llm_dabate_kabeen_runtime_artifacts_20260705.tar
```

압축을 풀면 `config/model.yaml`의 adapter 경로와 맞는 위치에 파일이 복원된다.

## 제외한 항목

- `eval_data/`: GovReport 원본 데이터. 약 1.5GB이며 재다운로드/재구성이 가능하다.
- `eval_outputs/`: 평가 산출물. 최종 수치는 `grpo_training_eval_kabeen.md`에 정리되어 있다.
- `logs/`: 학습 로그. 실행 필수 파일이 아니다.
- 중간 실험 adapter/checkpoint: 최종 적용 checkpoint가 아닌 smoke/run/OOM 실험 결과는 제외했다.
