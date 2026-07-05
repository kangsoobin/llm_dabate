#!/usr/bin/env python3
"""
rl/monitor_metrics.py — GRPO 학습 메트릭 감시
═══════════════════════════════════════════════

run_all.sh 로그에서 매 스텝 메트릭을 파싱해:
  - logs/rl_metrics.csv 에 누적 저장 (스텝별 한 행)
  - 임계값 위반 시 logs/rl_alerts.log 에 경고 기록
      · judge/parse_fail_rate > 0.05        → Judge few-shot 보강 필요
      · frac_reward_zero_std 최근 10스텝 평균 > 0.5 → 학습 신호 약화 (temperature/가중치 조정)
      · Traceback / CUDA OOM                → 파이프라인 중단
  - 60초마다 콘솔에 최신 상태 한 줄 출력

사용법: python rl/monitor_metrics.py   (run_all.sh와 별도 tmux 창에서)
"""

from __future__ import annotations

import csv
import glob
import os
import re
import sys
import time
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_GLOB = os.path.join(BASE_DIR, "logs", "rl_run_all_*.log")
CSV_PATH = os.path.join(BASE_DIR, "logs", "rl_metrics.csv")
ALERT_PATH = os.path.join(BASE_DIR, "logs", "rl_alerts.log")

FIELDS = [
    "side", "step", "reward", "reward_std", "frac_reward_zero_std",
    "stance", "rebuttal", "engagement", "persona", "antirep", "semantic_echo",
    "neutral_phrase", "format",
    "judge_parse_fail", "pairwise_parse_fail", "routed_frac",
    "kl", "entropy", "completion_len", "clipped_ratio", "grad_norm",
]

_METRIC_KEYS = {
    "reward": r"'reward': '([-\d.e]+)'",
    "reward_std": r"'reward_std': '([-\d.e]+)'",
    "frac_reward_zero_std": r"'frac_reward_zero_std': '([-\d.e]+)'",
    "stance": r"'rewards/reward_stance/mean': '([-\d.e]+)'",
    "rebuttal": r"'rewards/reward_rebuttal/mean': '([-\d.e]+)'",
    "engagement": r"'rewards/reward_engagement/mean': '([-\d.e]+)'",
    "persona": r"'rewards/reward_persona/mean': '([-\d.e]+)'",
    "antirep": r"'rewards/reward_antirep/mean': '([-\d.e]+)'",
    "semantic_echo": r"'rewards/reward_semantic_echo/mean': '([-\d.e]+)'",
    "neutral_phrase": r"'rewards/reward_neutral_phrase/mean': '([-\d.e]+)'",
    "format": r"'rewards/reward_format/mean': '([-\d.e]+)'",
    "judge_parse_fail": r"'judge/parse_fail_rate': '([-\d.e]+)'",
    "pairwise_parse_fail": r"'judge/pairwise_parse_fail_rate': '([-\d.e]+)'",
    "routed_frac": r"'judge/routed_frac': '([-\d.e]+)'",
    "kl": r"'kl': '([-\d.e]+)'",
    "entropy": r"'entropy': '([-\d.e]+)'",
    "completion_len": r"'completions/mean_length': '([-\d.e]+)'",
    "clipped_ratio": r"'completions/clipped_ratio': '([-\d.e]+)'",
    "grad_norm": r"'grad_norm': '([-\d.e]+)'",
}
_SIDE_RE = re.compile(r"\[(LEFT|RIGHT)\] GRPO 학습 시작")
_ERROR_RE = re.compile(r"Traceback|CUDA out of memory|torch\.OutOfMemoryError")


def alert(msg: str, seen: set) -> None:
    if msg in seen:
        return
    seen.add(msg)
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(f"\n⚠️  {line}", flush=True)
    with open(ALERT_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def parse_log(path: str) -> tuple[list[dict], bool]:
    rows, side, step, has_error = [], "?", 0, False
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _SIDE_RE.search(line)
            if m:
                side, step = m.group(1), 0
                continue
            if _ERROR_RE.search(line):
                has_error = True
            if "'rewards/reward_stance/mean'" not in line:
                continue
            step += 1
            row = {"side": side, "step": step}
            for key, pattern in _METRIC_KEYS.items():
                m = re.search(pattern, line)
                row[key] = float(m.group(1)) if m else None
            rows.append(row)
    return rows, has_error


def main() -> None:
    print(f"메트릭 감시 시작 — CSV: {CSV_PATH}, 경고: {ALERT_PATH}")
    seen_alerts: set = set()
    last_count = 0
    while True:
        logs = sorted(glob.glob(LOG_GLOB), key=os.path.getmtime)
        if not logs:
            time.sleep(60)
            continue
        rows, has_error = parse_log(logs[-1])

        if rows and len(rows) != last_count:
            last_count = len(rows)
            with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)

        if has_error:
            alert(f"로그에 Traceback/OOM 발견 — {logs[-1]} 확인 필요", seen_alerts)

        if rows:
            latest = rows[-1]
            tail = rows[-10:]
            frac_mean = sum(r["frac_reward_zero_std"] or 0 for r in tail) / len(tail)
            if (latest["judge_parse_fail"] or 0) > 0.05:
                alert(f"{latest['side']} step {latest['step']}: judge parse_fail_rate="
                      f"{latest['judge_parse_fail']:.2f} > 0.05 — Judge few-shot 보강 필요", seen_alerts)
            if len(tail) >= 10 and frac_mean > 0.5:
                alert(f"{latest['side']} step {latest['step']}: frac_reward_zero_std 최근 10스텝 평균 "
                      f"{frac_mean:.2f} > 0.5 — 학습 신호 약화 (temperature/가중치 점검)", seen_alerts)
            stance = latest["stance"]
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {latest['side']} step {latest['step']} | "
                  f"stance={stance:+.3f} reward={latest['reward']:+.3f} "
                  f"zero_std={latest['frac_reward_zero_std']:.2f} "
                  f"parse_fail={latest['judge_parse_fail'] if latest['judge_parse_fail'] is not None else '-'} "
                  f"kl={latest['kl']}", flush=True)
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 학습 스텝 대기 중 "
                  f"(현재 단계: 데이터 생성 또는 평가)", flush=True)
        time.sleep(60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
