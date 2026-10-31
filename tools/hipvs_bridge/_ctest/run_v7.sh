#!/usr/bin/env bash
# Regression for tools/hipvs_bridge/seekdb_cuvs_bridge.c (v7).
#
# The code under test is NOT a copy maintained in this directory: sync_v7.py
# re-extracts the body from the real source file on every run and refuses to
# proceed if an entry point went missing. A suite that silently tests old code
# is worse than no suite.
#
# Four builds:
#   ex   exclusive resource (SEEKDB_CUVS_EXCLUSIVE_RES=1) - legacy v6 lifetime
#   sh   shared resource    (v7 default)
#   cm   MUTANT: cache the resource in the handle again (the v6 bug)   -> MUST fail
#   cf   MUTANT: free the scratch through the live resource instead of the one
#        it was allocated from                                       -> MUST fail
# The two mutants are the point: they show the assertions can tell a fixed
# bridge from a broken one. A green run with dead mutants proves nothing.
#
# Windows/MSVC only: the cuVS stub needs an MSVC CRT to link against. The paths
# below point at a local _ctest/bin toolchain drop. On Linux (the contest box)
# build the bridge with build.sh instead; this harness is a desk check, not a
# contest gate.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PATH="$HERE/bin:$PATH"

PY="${PY:-python}"
INCS="-I. -Iinc -Imsvc_inc -Iuctt_inc -Ium_inc -Ish_inc"
LIBS="-LIBPATH:lib_ucrt -LIBPATH:lib_um -LIBPATH:lib_msvc -LIBPATH:lib_x64"

echo "== [0] sync the harness with ../seekdb_cuvs_bridge.c =="
"$PY" sync_v7.py || exit 2

build() {
  local tag="$1" defs="$2"
  cl -nologo -W0 -MD -O2 $INCS -DSK_NO_STUB_MAIN $defs -c stub_runtime.c -Fo"stub_$tag.obj" > "b_$tag.log" 2>&1 \
    || { echo "STUB COMPILE FAILED ($tag):"; sed -n '1,25p' "b_$tag.log"; return 2; }
  cl -nologo -W0 -MD -O2 $INCS $defs -c bridge_v7_test.c -Fo"br_$tag.obj" >> "b_$tag.log" 2>&1 \
    || { echo "BRIDGE COMPILE FAILED ($tag):"; sed -n '1,25p' "b_$tag.log"; return 2; }
  cl -nologo "br_$tag.obj" "stub_$tag.obj" -Fe"v7_$tag.exe" -link $LIBS >> "b_$tag.log" 2>&1 \
    || { echo "LINK FAILED ($tag):"; sed -n '1,25p' "b_$tag.log"; return 2; }
}

run_one() {
  local tag="$1" defs="$2" label="$3"
  echo
  echo "=================================================================="
  echo "  $label"
  echo "=================================================================="
  build "$tag" "$defs" || return 2
  ./v7_$tag.exe > "run_$tag.log" 2>&1
  local rc=$?
  grep -E "^  (ok|FAIL)|RESULT|created=|uaf=" "run_$tag.log" | tail -n 40
  return $rc
}

run_one ex "-DSEEKDB_CUVS_EXCLUSIVE_RES=1" "A. exclusive resource (legacy v6 lifetime)"
r_ex=$?
run_one sh ""                               "B. shared resource (v7 default)"
r_sh=$?
run_one cm "-DSK_MUTANT_CACHE_RES=1"        "C. MUTANT - handle caches the resource (must fail)"
r_cm=$?
run_one cf "-DSK_MUTANT_FREE_VIA_LIVE=1"    "D. MUTANT - free through the wrong resource (must fail)"
r_cf=$?

echo
echo "=============================================================="
echo "  SUMMARY   (exit code / verdict)"
echo "--------------------------------------------------------------"
verdict() { [ "$1" -eq 0 ] && echo "PASS" || echo "FAIL"; }
printf "  A exclusive        exit=%-3s %s   want exit 0\n" "$r_ex" "$(verdict $r_ex)"
printf "  B shared (v7)      exit=%-3s %s   want exit 0\n" "$r_sh" "$(verdict $r_sh)"
printf "  C mutant:cache     exit=%-3s %s   want NON-ZERO (caught)\n" "$r_cm" "$(verdict $r_cm)"
printf "  D mutant:free      exit=%-3s %s   want NON-ZERO (caught)\n" "$r_cf" "$(verdict $r_cf)"
echo "=============================================================="

[ "$r_ex" -eq 0 ] && [ "$r_sh" -eq 0 ] && [ "$r_cm" -ne 0 ] && [ "$r_cf" -ne 0 ]
