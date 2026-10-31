"""Splice the REAL bridge source into the v7 harness and prove nothing is stale.

A regression suite that silently tests yesterday's code is worse than none. So
this script re-extracts the code under test from the shipped source file on
every run, and refuses to proceed if a required entry point went missing.

It also derives the mutation (cache the resource inside the handle = the v6
bug) by patching the extracted body, so the mutant is guaranteed to differ from
the real code only in that one place.
"""
import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "seekdb_cuvs_bridge.c"
HARNESS = HERE / "bridge_v7_test.c"
BEGIN = "/* ==== BEGIN spliced body (generated - do not edit) ==== */"
END = "/* ==== END spliced body ==== */"

REQUIRED = [
    "seekdb_cuvs_build",
    "seekdb_cuvs_search",
    "seekdb_cuvs_free",
    "seekdb_cuvs_serialize",
    "seekdb_cuvs_deserialize",
    "seekdb_cuvs_cagra_knn",
    "seekdb_cuvs_shared_res_mode",
    "sk_cuvs_handle_res",
]


def patch(body: str, old: str, new: str, label: str) -> str:
    if body.count(old) != 1:
        raise SystemExit("sync: %s anchor matched %d times, expected 1" % (label, body.count(old)))
    return body.replace(old, new, 1)


def main() -> int:
    raw = SRC.read_bytes()
    text = raw.decode("utf-8").replace("\r\n", "\n")
    sha = hashlib.sha256(raw).hexdigest()

    body = text[text.index("/* seekdb <-> hipVS bridge"):]
    body = body.replace('#include "seekdb_cuvs_bridge.h"\n', "", 1)
    # Bridge-side probe: announce which resource search() picked. Compiled
    # only when the build passes -DSK_TRACE_RES -- a debug switch, not a
    # test switch; the shipped code carries no such output.
    _SLOT = "  if (sk_cuvs_scratch_reserve(h, res, nq) != 0) { rc = 12; goto out; }\n"
    body = patch(body, _SLOT,
                 "#ifdef SK_TRACE_RES\n"
                 "  fprintf(stderr, \"[dbg] search res=%p owned=%d cap=%zu\\n\",\n"
                 "          (void*)res, h->res_owned, h->nd_cap);\n"
                 "#endif\n" + _SLOT,
                 "search resource trace")

    # ---- mutant: the v6 lifetime, chosen at compile time -------------------
    m = body
    m = patch(m,
              "static cuvsResources_t sk_cuvs_handle_res(const sk_cuvs_index* h)\n{\n",
              "#ifdef SK_MUTANT_CACHE_RES\n"
              "static cuvsResources_t sk_cuvs_handle_res(const sk_cuvs_index* h)\n"
              "{ return h->res; }\n"
              "#else\n"
              "static cuvsResources_t sk_cuvs_handle_res(const sk_cuvs_index* h)\n{\n",
              "handle_res head")
    close = m.index("\n}\n", m.index("{ return h->res; }")) + 3
    m = m[:close] + "#endif\n" + m[close:]

    m = patch(m,
              "  h->res_owned = shared ? 0 : 1;\n  if (shared) {\n    h->res = NULL;",
              "#ifndef SK_MUTANT_CACHE_RES\n"
              "  h->res_owned = shared ? 0 : 1;\n  if (shared) {\n    h->res = NULL;",
              "build ownership head")
    m = patch(m,
              "    sk_cuvs_res_release(res, shared);\n  }\n  return h;",
              "    sk_cuvs_res_release(res, shared);\n  }\n"
              "#else\n  h->res_owned = shared ? 0 : 1;\n"
              "  if (shared) { sk_cuvs_res_release(res, shared); }\n"
              "#endif\n  return h;",
              "build ownership tail")

    # ---- mutant #2: free through the WRONG resource -------------------------
    # v7 records the resource the scratch was allocated from (h->scratch_res)
    # and frees through it. The mutant frees through the live pool instead,
    # which is what a "just re-resolve everywhere" half-fix would do.
    #
    # The wrong fix has to re-resolve the LIVE resource, not just take `owner`:
    # with res_owned==1 owner IS h->res, i.e. the correct allocator, and the
    # mutant would be a no-op. The mistake worth catching is "the live resource
    # is the shared pool, so free through the pool" -- which is exactly the
    # reasoning v7's first draft used for sk_cuvs_handle_res().
    m2 = patch(m,
               "  scratch = h->scratch_res ? h->scratch_res : owner;\n",
               "#ifdef SK_MUTANT_FREE_VIA_LIVE\n"
               "  scratch = g_res;  /* wrong: the live pool, not the allocator */\n"
               "#else\n"
               "  scratch = h->scratch_res ? h->scratch_res : owner;\n"
               "#endif\n",
               "free resource choice")

    print("   mutant markers: combined=%d free-mutant=%d"
          % (m2.count("SK_MUTANT_CACHE_RES"), m2.count("SK_MUTANT_FREE_VIA_LIVE")))
    pathlib.Path("_mutant_body.c").write_text(m2, encoding="utf-8", newline="\n")
    if m2.count("SK_MUTANT_CACHE_RES") != 2 or m2.count("SK_MUTANT_FREE_VIA_LIVE") != 1:
        raise SystemExit("sync: mutants were not generated as expected")

    # ---- splice ------------------------------------------------------------
    head = (HERE / "harness_prologue.c").read_text(encoding="utf-8")
    tail = (HERE / "harness_epilogue.c").read_text(encoding="utf-8")
    header = ("\n/* source: %s\n * sha256: %s\n * bytes:  %d (disk)\n */\n"
              % (SRC.name, sha, len(raw)))
    # the mutant is a preprocessor branch inside the body, so both lifetimes
    # live in one copy and the -D flag decides which compiles
    spliced = header + m2

    HARNESS.write_text(head + BEGIN + spliced + "\n" + END + tail,
                       encoding="utf-8", newline="\n")

    # ---- guard -------------------------------------------------------------
    missing = [s for s in REQUIRED if s not in spliced]
    if missing:
        raise SystemExit("sync: spliced body is missing " + ", ".join(missing))
    if sha not in spliced:
        raise SystemExit("sync: spliced body does not carry the source hash")
    print("   spliced %d bytes from %s (sha256 %s)" % (len(spliced.encode()), SRC.name, sha[:16]))
    print("   all %d required symbols present" % len(REQUIRED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
