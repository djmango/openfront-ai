#!/usr/bin/env bash
# Re-copy src/core_impl.rs into the device module of src/main.rs, between the
# BEGIN/END VERBATIM COPY markers. cuda-oxide's `#[cuda_module]` attribute macro
# cannot see through an `include!` inside the module (it expands the module
# before the include does), so the device copy has to be literal text - and this
# script plus the `device_core_is_a_verbatim_copy` test in lib.rs are what keep
# the two copies from drifting.
#
# Usage: bash sync_device_core.sh
set -euo pipefail
cd "$(dirname "$0")"
# systemd/agent shells here get a PATH without python3 (see the operator-status
# lesson in the puffer skill), so resolve it explicitly.
PY="$(command -v python3 || true)"
[ -n "$PY" ] || PY=/run/current-system/sw/bin/python3
"$PY" - <<'PY'
core = open("src/core_impl.rs").read().rstrip("\n")
p = "src/main.rs"
s = open(p).read()
b = "    // ===== BEGIN VERBATIM COPY of src/core_impl.rs =====\n"
e = "    // ===== END VERBATIM COPY of src/core_impl.rs =====\n"
i = s.index(b) + len(b)
j = s.index(e)
head = s[:i]
tail = s[j:]
s = head + core + "\n" + tail
open(p, "w").write(s)
print(f"synced {len(core)} bytes of core_impl.rs into the device module")
PY
