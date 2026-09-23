#!/usr/bin/env bash
set -euo pipefail
bash scripts/run_local.sh python scripts/run_contact.py --plane --seconds 8 --output evaluation/final_plane > evaluation/final_plane.log 2>&1
bash scripts/run_local.sh python scripts/run_contact.py --offsets-mm 1 2 5 --directions xp --xy-pulse --seconds 12 --output evaluation/final_handoff > evaluation/final_handoff.log 2>&1
bash scripts/run_local.sh python scripts/run_contact.py --seconds 12 --set fast_speed=0.001 --output evaluation/conservative_approach > evaluation/conservative_approach.log 2>&1
bash scripts/run_local.sh python scripts/check_gpu_reset.py > evaluation/gpu_reset.log 2>&1
bash scripts/run_local.sh python scripts/diagnose_contact.py --output evaluation/final_cpu_gpu_audit --config evaluation/final_offsets/config.json > evaluation/final_cpu_gpu_audit.log 2>&1
