#!/usr/bin/env bash
# Retrain the paper checkpoints, sequentially:
#   - the benchmark matrix: 3 algos x 2 tasks x 2 planes x SEEDS (default 0-4, 60 runs)
#   - the ground-effect study's altitude policies on the other two airframes (PPO, seed 0)
# Idempotent: a checkpoint that already exists is skipped, so an interrupted queue resumes.
#
#   CKPT=checkpoints RESULTS=results FALCONS=.venv/bin/falcons SEEDS="0 1 2 3 4" scripts/train_seeds.sh
#
# Each run's start, end and wall-clock seconds are appended to $RESULTS/_logs/train_timings.tsv.
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT=${CKPT:-checkpoints}
RESULTS=${RESULTS:-results}
FALCONS=${FALCONS:-.venv/bin/falcons}
SEEDS=${SEEDS:-"0 1 2 3 4"}
mkdir -p "$CKPT" "$RESULTS/_logs"
LOG="$RESULTS/_logs/train_timings.tsv"
[ -f "$LOG" ] || printf "algo\ttask\tplane\tseed\tstart\tend\tseconds\trc\n" > "$LOG"

run() {   # algo task plane seed
  local ck="$CKPT/${1}_${2}_${3}_s${4}.pt"
  if [ -f "$ck" ]; then echo "SKIP $ck"; return 0; fi
  local t0 t1 s rc=0
  t0=$(date +%s); s=$(date -Is)
  echo "=== $1 $2 $3 s$4  $s"
  "$FALCONS" train --checkpoints "$CKPT" --results "$RESULTS" \
    --algo "$1" --task "$2" --plane "$3" --seed "$4" || rc=$?
  t1=$(date +%s)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$1" "$2" "$3" "$4" "$s" "$(date -Is)" "$((t1 - t0))" "$rc" >> "$LOG"
  return $rc
}

# one run of every (algo, task) first, so the cost of each kind is known early
for algo in ppo sac td3; do for task in altitude attitude; do
  run "$algo" "$task" Airship_V7 0
done; done
for seed in $SEEDS; do for algo in ppo sac td3; do for task in altitude attitude; do
  for plane in Airship_V7 Volantex_Ranger; do run "$algo" "$task" "$plane" "$seed"; done
done; done; done
for plane in Airship_A0S Navion; do run ppo altitude "$plane" 0; done
echo "ALL DONE $(date -Is)"
