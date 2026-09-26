#!/usr/bin/env bash
# Retired 64-QA listed smoke. Kept only so old commands fail loudly.
#
# 64 paper-task QA with accum=512 clamps to 10 update steps. That is not
# listed training. Wire the frozen 3253-QA manifests, then train Random-K.
set -euo pipefail

HNG="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "error: 64-QA / 10-step shared-pool listed smoke is retired." >&2
echo "  That run only proves wiring; it is not a training budget." >&2
echo "  CPU-wire every frozen ID first:" >&2
echo "    bash $HNG/configs/run_wire_shared_pool_samplers.sh" >&2
echo "  Then train Random-K on the full paper task file (tmux, accum=512):" >&2
echo "    TMUX_SESSION=shared-pool-random-k-full-20260926 \\" >&2
echo "      bash $HNG/configs/run_shared_pool_random_k_full.sh" >&2
echo "  Do not rescore 3253x198. Do not expand the 64-QA shot overnight." >&2
exit 1
