#!/usr/bin/env bash
set -euo pipefail
bash scripts/run_local.sh python scripts/run_contact.py --seconds 12 --output evaluation/final_repeat2 > evaluation/final_repeat2.log 2>&1
bash scripts/run_local.sh python scripts/run_contact.py --seconds 12 --output evaluation/final_repeat3 > evaluation/final_repeat3.log 2>&1
