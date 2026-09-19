#!/usr/bin/env bash
# Inject src/core_impl.rs into src/main.rs between the VERBATIM COPY markers.
#
# cuda-oxide's `#[cuda_module]` attribute macro sees the module before an
# `include!` inside it expands, so the device copy has to be real text in the
# file. This script is the only writer for that region; lib.rs's
# `device_core_is_a_verbatim_copy` test fails if anyone edits one side by hand.
set -euo pipefail
cd "$(dirname "$0")"

CORE=src/core_impl.rs
MAIN=src/main.rs
BEGIN='===== BEGIN VERBATIM COPY of src/core_impl.rs ====='
END='===== END VERBATIM COPY of src/core_impl.rs ====='

grep -q "$BEGIN" "$MAIN" || { echo "no BEGIN marker in $MAIN" >&2; exit 1; }
grep -q "$END" "$MAIN" || { echo "no END marker in $MAIN" >&2; exit 1; }

awk -v core="$CORE" -v end_marker="$END" '
  index($0, "===== BEGIN VERBATIM COPY of src/core_impl.rs =====") {
    print
    while ((getline line < core) > 0) print line
    close(core)
    skip = 1
    next
  }
  index($0, end_marker) { skip = 0 }
  skip { next }
  { print }
' "$MAIN" > "$MAIN.tmp"
mv "$MAIN.tmp" "$MAIN"

echo "synced $CORE -> $MAIN ($(grep -c '' "$CORE") lines)"
