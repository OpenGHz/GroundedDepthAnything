#!/usr/bin/env bash
# Reproducible H200/B300 bootstrap. This remains a normal Bash script, while
# `ai run scripts/setup-gpu.sh` caches each successful step for safe resume.

set -euo pipefail

#ai >>>
if [[ -f pixi.toml ]]; then
  :
elif [[ -f ../pixi.toml ]]; then
  cd ..
else
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
fi
#ai <<<

#ai >>>
if ! command -v pixi >/dev/null 2>&1 && [[ -x "${HOME}/.pixi/bin/pixi" ]]; then
  export PATH="${HOME}/.pixi/bin:${PATH}"
fi
if ! command -v pixi >/dev/null 2>&1; then
  echo "pixi is required: https://pixi.sh/latest/installation/" >&2
  exit 1
fi
#ai <<<

#ai: name verify-sibling-repositories
python3 scripts/check-workspace.py

#ai: name ensure-sam2-checkpoint
python3 scripts/ensure-sam2-checkpoint.py

#ai: name detect-gpu-platform
#ai >>>
if [[ -z "${GDA_PIXI_PLATFORM:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && \
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -qi 'B300'; then
    export GDA_PIXI_PLATFORM=b300
  else
    export GDA_PIXI_PLATFORM=h200
  fi
fi
echo "using Pixi platform: ${GDA_PIXI_PLATFORM}"
#ai <<<

#ai: name install-locked-environment
pixi install --platform "${GDA_PIXI_PLATFORM}" --locked

#ai: name build-sam2-cuda-extension
pixi run --platform "${GDA_PIXI_PLATFORM}" --locked build-sam2

#ai: name verify-gpu-runtime
pixi run --platform "${GDA_PIXI_PLATFORM}" --locked --skip-deps doctor

#ai: name run-unit-tests
pixi run --platform "${GDA_PIXI_PLATFORM}" --locked test

#ai: name run-linter
pixi run --platform "${GDA_PIXI_PLATFORM}" --locked lint

echo "${GDA_PIXI_PLATFORM} Pixi environment is ready."
