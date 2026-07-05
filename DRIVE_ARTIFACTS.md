# Google Drive Final GRPO Adapters

이 저장소의 `kabeen` 브랜치는 코드와 가벼운 설정 파일을 GitHub에 두고, Git에서 제외한 최종 GRPO adapter checkpoint는 Google Drive에 둔다.

최종 adapter-only Drive 폴더:

```text
https://drive.google.com/drive/folders/1qXqNy5tdpgb_HyjlIm1ykgwDaNZRI-z_
```

상위 공유 폴더:

```text
https://drive.google.com/drive/folders/1pCMROcO7hW5_xuf9uscbaA7y92RN3gbG
```

## 포함된 파일

Drive 하위 폴더 `llm_dabate_kabeen_final_grpo_adapters_only_20260705` 안에는 최종 실행에 필요한 두 checkpoint만 있다.

```text
README_final_grpo_adapters_only.md
left_grpo_v3_checkpoint-112.tar.part-00
left_grpo_v3_checkpoint-112.tar.part-01
right_grpo_v3_checkpoint-120.tar.part-00
right_grpo_v3_checkpoint-120.tar.part-01
```

Part 파일을 합치면 각각 다음 tar가 된다.

```text
left_grpo_v3_checkpoint-112.tar
right_grpo_v3_checkpoint-120.tar
```

각 tar 안에는 다음 경로가 들어 있다.

```text
adapters/left_grpo_v3_left_long_g9/checkpoint-112
adapters/right_grpo_v3_right_long_g8/checkpoint-120
```

중간 실험 checkpoint, smoke checkpoint, GovReport 원본 데이터, 평가 출력물, `rl/data`는 포함하지 않았다.

## 필요한 이유

`config/model.yaml`은 최종 GRPO v3 adapter를 아래 경로로 참조한다.

```yaml
left_adapter:  "adapters/left_grpo_v3_left_long_g9/checkpoint-112"
right_adapter: "adapters/right_grpo_v3_right_long_g8/checkpoint-120"
```

하지만 `.gitignore`에서 `adapters/`를 제외하므로, 위 최종 adapter checkpoint는 GitHub에 직접 올라가지 않는다.

## 복원 방법

GitHub 저장소를 clone한다.

```bash
git clone -b kabeen git@github.com:kangsoobin/llm_dabate.git
cd llm_dabate
```

Drive에서 part 파일 4개를 같은 로컬 폴더에 받은 뒤 LEFT/RIGHT를 각각 합친다.

```bash
cat left_grpo_v3_checkpoint-112.tar.part-* > left_grpo_v3_checkpoint-112.tar
cat right_grpo_v3_checkpoint-120.tar.part-* > right_grpo_v3_checkpoint-120.tar
```

checksum을 확인한다.

```bash
sha256sum left_grpo_v3_checkpoint-112.tar right_grpo_v3_checkpoint-120.tar
```

원본 tar SHA256:

```text
4288e5280fdad393c55e6d03093d16741dd5ab250bad959ec76e79e334a717ff  left_grpo_v3_checkpoint-112.tar
bf771e41d2cc4d6b75f3b496107398291dc2d3696a01a925173d028bbdad0dc2  right_grpo_v3_checkpoint-120.tar
```

SHA256이 일치하면 저장소 루트에서 tar를 푼다.

```bash
tar -xf /path/to/left_grpo_v3_checkpoint-112.tar
tar -xf /path/to/right_grpo_v3_checkpoint-120.tar
```

압축을 풀면 `config/model.yaml`의 adapter 경로와 맞는 위치에 파일이 복원된다.

## 참고

각 checkpoint 안의 `ref/`는 GRPO 학습 과정에서 저장된 reference adapter다. 순수 inference에는 없어도 되지만, checkpoint 디렉터리 전체를 보존하기 위해 함께 포함했다.
