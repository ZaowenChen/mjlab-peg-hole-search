#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="${MPHS_RUNTIME:-${HOME}/.local/share/mphs/runtime-20260917}"
cuda_lib="$("${runtime_dir}/bin/python" -c 'import site; from pathlib import Path; print(next((str(Path(p)/"nvidia/cu13/lib") for p in site.getsitepackages() if (Path(p)/"nvidia/cu13/lib/libnvrtc-builtins.so.13.0").is_file()), ""))')"
export PATH="${runtime_dir}/bin:${PATH}"
export PYTHONPATH="${project_root}/src"
export LD_LIBRARY_PATH="${cuda_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/contact20_mpl
export WARP_CACHE_PATH=/tmp/contact20_warp
cd -- "${project_root}"
exec "$@"
