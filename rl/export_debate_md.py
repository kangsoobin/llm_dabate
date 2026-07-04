#!/usr/bin/env python3
"""
rl/export_debate_md.py — 토론 트랜스크립트(jsonl)를 읽기 좋은 markdown으로 변환
════════════════════════════════════════════════════════════════════════════

사용법:
    python rl/export_debate_md.py --in rl/data/eval_grpo.jsonl --out docs/eval/20260704_debate_grpo.md \
        --title "GRPO 어댑터 토론 (4주제 × 8라운드)"
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIDE_LABEL = {
    "left": "🔵 LEFT (이진보)",
    "right": "🔴 RIGHT (김보수)",
    "synthesizer": "⚪ SYNTHESIZER (중간 개입)",
    "synthesizer_final": "⚪ SYNTHESIZER (최종 종합)",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--title", default="토론 트랜스크립트")
    args = parser.parse_args()

    inp = args.inp if os.path.isabs(args.inp) else os.path.join(BASE_DIR, args.inp)
    out = args.out if os.path.isabs(args.out) else os.path.join(BASE_DIR, args.out)

    by_topic: dict[str, list[dict]] = defaultdict(list)
    topic_order: list[str] = []
    with open(inp, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["question"] not in by_topic and row["side"] == "left" and row["round_num"] == 1:
                topic_order.append(row["question"])
            # 개입 이후 질문이 바뀌어도 원 주제로 묶기 위해 topic 필드를 우선 사용
            key = row.get("topic", row["question"])
            if key not in by_topic:
                topic_order.append(key) if key not in topic_order else None
            by_topic[key].append(row)

    lines = [f"# {args.title}", "", f"> 원본: `{args.inp}` — 총 {sum(len(v) for v in by_topic.values())}턴", ""]
    for topic in topic_order:
        rows = by_topic.get(topic)
        if not rows:
            continue
        lines += [f"\n## 주제: {topic}", ""]
        rows_sorted = sorted(rows, key=lambda r: (r["round_num"], 0 if r["side"] == "left" else (2 if r["side"].startswith("synthesizer") else 1)))
        current_round = None
        for r in rows_sorted:
            if r["round_num"] != current_round:
                current_round = r["round_num"]
                lines.append(f"\n### 라운드 {current_round}")
                if r.get("question") != topic:
                    lines.append(f"\n_(사회자 질문: {r['question']})_")
            label = SIDE_LABEL.get(r["side"], r["side"])
            lines += [f"\n**{label}**", "", r["response_ref"].strip(), ""]

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
