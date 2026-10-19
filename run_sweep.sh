#!/bin/bash
# Timing sweep with GPU clock telemetry sampled alongside, so a throttled run
# is visible in the record. Writes results/sweep_<date>.jsonl and .clocks.csv.
set -euo pipefail
cd "$(dirname "$0")"
tag=${1:-$(date +%Y%m%d_%H%M)}
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active --format=csv -lms 500 > results/sweep_$tag.clocks.csv &
smi=$!
./build/bench "${@:2}" > results/sweep_$tag.jsonl 2> results/sweep_$tag.log || true
kill $smi
echo "results/sweep_$tag.jsonl"
