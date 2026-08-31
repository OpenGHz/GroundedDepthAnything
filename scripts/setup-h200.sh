#!/usr/bin/env bash
# Resumable H200 bootstrap. This is also a normal Bash script; `ai run` caches
# successful expensive steps and resumes after the first failure.

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

export GDA_PIXI_PLATFORM=h200
export GDA_SETUP_INPUTS_SHA256=c89ff6925a7c3e495298efe44b3a09a22cb25235cb5d4b7053c04c3b03b897e2

#ai: always
#ai: name validate-setup-inputs
#ai >>>
actual_setup_fingerprint=$(python3 scripts/setup-fingerprint.py)
if [[ "${actual_setup_fingerprint}" != "${GDA_SETUP_INPUTS_SHA256}" ]]; then
  echo "setup inputs changed: expected ${GDA_SETUP_INPUTS_SHA256}, found ${actual_setup_fingerprint}" >&2
  echo "update GDA_SETUP_INPUTS_SHA256 in both platform manifests" >&2
  exit 1
fi
#ai <<<

# The two targets share Pixi's default prefix, so a checkout must never switch.
#ai: always
#ai: name bind-checkout-to-h200
python3 scripts/check-setup-platform.py h200

# Validate canonical upstream URLs before allowing Git to sync or fetch them.
#ai: always
#ai: name verify-submodule-metadata
python3 scripts/check-workspace.py --metadata-only

#ai: always
#ai: name initialize-pinned-submodules
#ai >>>
git submodule sync --recursive
git submodule update --init --recursive
#ai <<<

#ai: always
#ai: name verify-pinned-submodules
python3 scripts/check-workspace.py

# Cache location is an external runtime input, so verify/download every time.
#ai: always
#ai: name ensure-sam2-checkpoint
python3 scripts/ensure-sam2-checkpoint.py

#ai: name install-locked-h200-environment
pixi install --platform h200 --locked

# Cache location is an external runtime input, so verify/download every time.
#ai: always
#ai: name ensure-sam3-checkpoint
pixi run --platform h200 --locked ensure-sam3

#ai: name build-h200-sam2-cuda-extension
pixi run --platform h200 --locked build-sam2

# The current GPU may change even when the checkout and cached build do not.
#ai: always
#ai: name verify-h200-gpu-runtime
pixi run --platform h200 --locked --skip-deps doctor

#ai: name run-unit-tests
pixi run --platform h200 --locked test

#ai: name run-format-check
pixi run --platform h200 --locked format-check

#ai: name run-linter
pixi run --platform h200 --locked lint

echo "h200 Pixi environment is ready."
