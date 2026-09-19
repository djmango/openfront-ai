#!/usr/bin/env bash
# Reproduce the whole ofcuda_hash verification yardstick from scratch.
#
#   bash scripts/run_hash_yardstick.sh
#
# Steps:
#   1. build the engine-linked oracle (own target dir - never the live trainer's)
#   2. dump the state planes + engine per-tick hashes for two windows
#   3. build the cuda-oxide crate
#   4. run the GPU harness and the CPU companion on both dumps
#
# The live trainer owns rust/target/release/libopenfront_engine.so: nothing here
# writes into openfront-ai/rust/target.
set -euo pipefail

SKG=/opt/data/workspaces/skg
OF=$SKG/ofcuda_hash
REC=$SKG/openfront-ai/records/early-curriculum-parity/curr-b002-s1-pangaea.json.gz
ENV=$SKG/ofcuda_env.sh
export PATH="/nix/store/qmdxxa88bgbdx31dav3qlssb4rghr14c-gcc-wrapper-14.4.0/bin:$PATH"

# ---- 1/2. oracle: state planes + the engine's own hashes -------------------
export CARGO_TARGET_DIR=/tmp/ofhash-oracle-target
( cd "$OF/oracle" && nice -n 10 cargo build --release -j 4 )
ORACLE=/tmp/ofhash-oracle-target/release/ofhash_oracle

nice -n 10 "$ORACLE" --mode parity --ticks-per-decision 1 --decisions 130 \
    --out /tmp/ofhash_parity
nice -n 10 "$ORACLE" --mode record --record "$REC" \
    --replay-ticks 200 --from-tick 500 --out /tmp/ofhash_record_late

# ---- 3. the CUDA yardstick -------------------------------------------------
( cd "$OF" && nice -n 10 "$ENV" cargo oxide build )

# ---- 4. GPU + CPU, on both windows ----------------------------------------
for d in parity record_late; do
    nice -n 10 "$ENV" "$OF/target/release/ofcuda_hash" "/tmp/ofhash_$d" \
        | tee "/tmp/run_${d}_gpu.out"
    nice -n 10 "$ENV" "$OF/target/release/ofcuda_hash_cpu" "/tmp/ofhash_$d" \
        | tee "/tmp/run_${d}_cpu.out"
done

# ---- verdict -----------------------------------------------------------------
grep -H '^verdict\|^per_tick' /tmp/run_parity_gpu.out /tmp/run_record_late_gpu.out
