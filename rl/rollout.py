"""
rl/rollout.py
───────────────
rl/simulate.py가 생성한 트랜스크립트(JSONL)를 GRPOTrainer용 datasets.Dataset으로 변환한다.

각 행은 TRL의 conversational 포맷(`prompt` = 메시지 딕셔너리 리스트)을 따르며,
reward_func가 kwargs로 받을 수 있도록 side/question/opponent_response/own_history/
round_num/evidence 컬럼을 함께 담는다 (rl/rewards/composer.py의 as_trl_reward_fn 참고).
"""

from __future__ import annotations

import json


def load_transcript_rows(path: str, side: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["side"] != side:
                continue
            rows.append(row)
    return rows


def build_grpo_dataset(path: str, side: str):
    """
    Parameters
    ----------
    path : rl/simulate.py가 만든 transcripts.jsonl 경로
    side : "left" | "right" — 이 side의 턴만 골라 학습 데이터셋을 만든다

    Returns
    -------
    datasets.Dataset
        컬럼: prompt(list[dict]), side, question, opponent_response, own_history, round_num, evidence
    """
    from datasets import Dataset

    rows = load_transcript_rows(path, side)
    if not rows:
        raise ValueError(
            f"{path}에 side={side} 데이터가 없습니다. rl/simulate.py를 먼저 실행하세요."
        )

    records = [
        {
            "prompt": row["prompt_messages"],
            "side": row["side"],
            "question": row["question"],
            "opponent_response": row["opponent_response"],
            "own_history": row["own_history"],
            "round_num": row["round_num"],
            "evidence": row.get("evidence"),
        }
        for row in rows
    ]
    return Dataset.from_list(records)
