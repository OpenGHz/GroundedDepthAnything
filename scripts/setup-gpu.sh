#!/usr/bin/env bash
# Reproducible H200/B300 bootstrap dispatcher.
#
# Run this file directly with Bash for automatic GPU selection. For resumable
# anchored-install caching, invoke setup-h200.sh or setup-b300.sh explicitly so
# incompatible Pixi targets never share a manifest cache.

set -euo pipefail

if [[ -f pixi.toml ]]; then
  :
elif [[ -f ../pixi.toml ]]; then
  cd ..
else
  cd "$(dirname "${BASH_SOURCE[0]}")/.."
fi

if [[ -n "${GDA_PIXI_PLATFORM:-}" ]]; then
  platform="${GDA_PIXI_PLATFORM}"
else
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cannot detect a supported GPU: nvidia-smi is unavailable; set GDA_PIXI_PLATFORM=h200 or b300 explicitly" >&2
    exit 2
  fi

  if ! gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null); then
    echo "cannot detect a supported GPU: nvidia-smi failed; set GDA_PIXI_PLATFORM=h200 or b300 explicitly" >&2
    exit 2
  fi

  has_h200=0
  has_b300=0
  if grep -Eqi '(^|[^[:alnum:]])H200([^[:alnum:]]|$)' <<<"${gpu_names}"; then
    has_h200=1
  fi
  if grep -Eqi '(^|[^[:alnum:]])B300([^[:alnum:]]|$)' <<<"${gpu_names}"; then
    has_b300=1
  fi

  case "${has_h200}${has_b300}" in
    10) platform=h200 ;;
    01) platform=b300 ;;
    11)
      echo "multiple supported GPU families detected (H200 and B300); set GDA_PIXI_PLATFORM explicitly" >&2
      exit 2
      ;;
    *)
      echo "unsupported GPU(s): ${gpu_names//$'\n'/, }; set GDA_PIXI_PLATFORM=h200 or b300 explicitly" >&2
      exit 2
      ;;
  esac
fi

case "${platform}" in
  h200 | b300) ;;
  *)
    echo "GDA_PIXI_PLATFORM must be h200 or b300, found: ${platform}" >&2
    exit 1
    ;;
esac

echo "dispatching setup for Pixi platform: ${platform}"
exec bash "scripts/setup-${platform}.sh"
