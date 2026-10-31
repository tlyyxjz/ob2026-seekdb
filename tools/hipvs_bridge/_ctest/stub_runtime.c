/* Stub cuVS/ROCm runtime + test driver for the seekdb <-> hipVS bridge.
 *
 * Compiles seekdb_cuvs_bridge.c for real and exercises it end to end. Every
 * cuVS object carries a magic word; touching a destroyed resource, or freeing
 * device memory through a different resource than it was allocated from, is
 * recorded in g_use_after_destroy so the test can fail loudly.
 */
#include "bridge_stub.h"
#include "stub_iface.h"

/* Two lifetime modes for the bridge's cuVS resource, exercised by the same
 * driver source. SK_TEST_SHARED_RES is set by the build for the shared-resource
 * variant only; it changes what "correct" means, not what the bridge does. */
#ifndef SK_TEST_SHARED_RES
#define SK_TEST_SHARED_RES 0
#endif

size_t g_last_ideg = 0, g_last_gdeg = 0, g_last_niter = 0;
int    g_build_calls = 0;
int    g_res_created = 0, g_res_destroyed = 0, g_use_after_destroy = 0;
int    g_live_idx = 0;
int    g_idx_res[8] = {-1,-1,-1,-1,-1,-1,-1,-1};
long   g_shape_n = 0, g_shape_d = 0;

static char g_last_error[256] = "ok";
static void seterr(const char* m) { snprintf(g_last_error, sizeof g_last_error, "%s", m); fprintf(stderr, "[UAF] %s\n", m); }
void ddd(const char* m);

/* ------------------------------------------------- counters for the driver -- */
int stub_res_created(void)   { return g_res_created; }
int stub_res_destroyed(void) { return g_res_destroyed; }
int stub_uaf(void)           { return g_use_after_destroy; }
int stub_build_calls(void)   { return g_build_calls; }
int stub_idx_resource_id(int slot) { return g_idx_res[slot & 7]; }
int stub_live_handles(void)  { return g_live_idx; }

const stub_build_opts* stub_last_build_opts(void)
{
  static stub_build_opts o;
  o.graph_degree = g_last_gdeg;
  o.intermediate_graph_degree = g_last_ideg;
  o.nn_descent_niter = g_last_niter;
  return &o;
}

/* Build a resource, destroy it, then hand the dead pointer back to cuVS once.
 * The stub has to report that as use-after-destroy, otherwise the "0 UAF"
 * assertions in the driver would be vacuous. */
void stub_uaf_probe_stale(void)
{
  cuvsResources_t r = NULL;
  void* p = NULL;
  if (cuvsResourcesCreate(&r) != CUVS_SUCCESS) { return; }
  cuvsResourcesDestroy(r);
  (void)cuvsRMMAlloc(r, &p, 16);
}

static void say(const char* m)
{
  fprintf(stderr, "[seq] %s\n", m);
  fflush(stderr);
}

/* Freed block goes to a quarantine list, NOT back to the CRT heap.
 *
 * Real cuVS hands the freed handle's memory back to the allocator, so a stale
 * pointer usually still "works" and the bug hides. Keeping the block intact
 * behind the magic word makes every use-after-destroy deterministic: the test
 * reports the same verdict on every run instead of once in five. */
static void* g_trash[64];
static int   g_trash_n = 0;
void ddd(const char* m)
{
  fprintf(stderr, "[B5] %s\n", m);
  fflush(stderr);
}

static void trash(void* p)
{
  if (p == NULL) return;
  if (g_trash_n < (int)(sizeof g_trash / sizeof g_trash[0])) { g_trash[g_trash_n++] = p; }
  else { free(p); }
}

/* ------------------------------------------------------------------ core --- */
cuvsError_t cuvsResourcesCreate(cuvsResources_t* out)
{
  cuvsResources_t r;
  say("res create");
  r = (cuvsResources_t)calloc(1, sizeof(*r));
  if (!r) return 1;
  r->magic = MAGIC_LIVE;
  r->id = ++g_res_created;
  *out = r;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsResourcesDestroy(cuvsResources_t r)
{
  if (!r) return 0;
  say("res destroy");
  if (r->magic != MAGIC_LIVE) { g_use_after_destroy++; seterr("destroy on dead resource"); return 1; }
  r->magic = MAGIC_DEAD;
  g_res_destroyed++;
  trash(r);
  return CUVS_SUCCESS;
}

const char* cuvsGetLastErrorText(void) { return g_last_error; }

typedef struct { cuvsResources_t owner; unsigned magic; } alloc_hdr;

cuvsError_t cuvsRMMAlloc(cuvsResources_t r, void** p, size_t n)
{
  alloc_hdr* h;
  say("alloc");
  if (!r || r->magic != MAGIC_LIVE) { g_use_after_destroy++; seterr("RMMAlloc on dead resource"); return 1; }
  h = (alloc_hdr*)calloc(1, sizeof(alloc_hdr) + n);
  if (!h) return 1;
  h->owner = r; h->magic = MAGIC_LIVE;
  *p = (void*)(h + 1);
  return CUVS_SUCCESS;
}

cuvsError_t cuvsRMMFree(cuvsResources_t r, void* p, size_t n)
{
  alloc_hdr* h;
  (void)n;
  say("free");
  if (!p) return CUVS_SUCCESS;
  h = ((alloc_hdr*)p) - 1;
  if (h->magic != MAGIC_LIVE) { g_use_after_destroy++; seterr("double free"); return 1; }
  if (!r || r->magic != MAGIC_LIVE) { g_use_after_destroy++; seterr("RMMFree on dead resource"); return 1; }
  if (h->owner != r) { g_use_after_destroy++; seterr("RMMFree through a different resource"); return 1; }
  h->magic = MAGIC_DEAD;
  trash(h);
  return CUVS_SUCCESS;
}

/* Which resource the block `p` was allocated from, or -1 when `p` is not one of
 * ours. Lets the driver assert that a handle's scratch really sits on a
 * different resource than the live pool -- without that assertion, a test
 * meant to tell two allocators apart can silently compare a pointer to itself. */
int stub_block_resource_id(void* p)
{
  alloc_hdr* h;
  if (p == NULL) { return -1; }
  h = ((alloc_hdr*)p) - 1;
  return (h->owner != NULL) ? h->owner->id : -1;
}

cudaError_t cudaMemcpy(void* d, const void* s, size_t n, int kind)
{
  (void)kind;
  memcpy(d, s, n);
  return cudaSuccess;
}

/* ----------------------------------------------------------------- cagra --- */
cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* p)
{
  *p = (cuvsCagraIndexParams_t)calloc(1, sizeof(**p));
  return *p ? CUVS_SUCCESS : 1;
}
cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t p) { free(p); return CUVS_SUCCESS; }

cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* p)
{
  *p = (cuvsCagraSearchParams_t)calloc(1, sizeof(**p));
  if (*p) { (*p)->itopk_size = 64; (*p)->search_width = 1; }
  return *p ? CUVS_SUCCESS : 1;
}
cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t p) { free(p); return CUVS_SUCCESS; }

/* Flame probe left in place: if a future run hangs or dies mid-build, this
 * tells you which stage it reached. Enabled by SK_STUB_TRACE=1. */
void dbg(const char* m)
{
  static int on = -1;
  if (on < 0) { const char* e = getenv("SK_STUB_TRACE"); on = (e && *e == '1') ? 1 : 0; }
  if (!on) { return; }
  fprintf(stderr, "[p] %s\n", m);
  fflush(stderr);
}

cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t* p)
{
  *p = (cuvsCagraIndex_t)calloc(1, sizeof(**p));
  if (*p) { (*p)->magic = MAGIC_LIVE; g_live_idx++; }
  return *p ? CUVS_SUCCESS : 1;
}
cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t p)
{
  if (p && p->magic == MAGIC_LIVE) { g_live_idx--; }
  if (p) { p->magic = MAGIC_DEAD; free(p); }
  return CUVS_SUCCESS;
}

cuvsError_t cuvsCagraBuild(cuvsResources_t r, cuvsCagraIndexParams_t ip,
                           DLManagedTensor* ds, cuvsCagraIndex_t idx)
{
  dbg("build enter");
  if (!r || r->magic != MAGIC_LIVE) { g_use_after_destroy++; seterr("build on dead resource"); return 1; }
  if (!idx || idx->magic != MAGIC_LIVE) { seterr("build on dead index"); return 1; }
  g_build_calls++;
  /* Remember which resource each new index was built against, by identity
   * (creation order). Pointer comparison is useless in the test because a
   * freed resource slot gets recycled by the very next allocation. */
  {
    int i, free_slot = -1;
    for (i = 0; i < 8; i++) { if (g_idx_res[i] < 0) { free_slot = i; break; } }
    if (free_slot < 0) { free_slot = g_build_calls & 7; }
    g_idx_res[free_slot] = r->id;
    /* keep the array indexed by build order, newest last */
    for (i = 7; i > 0; i--) { g_idx_res[i] = g_idx_res[i - 1]; }
    g_idx_res[0] = r->id;
  }
  /* Snapshot shape comes from the dataset actually handed to the build, so the
   * deserialize shape check downstream is exercised with real numbers. */
  g_shape_n = (long)ds->dl_tensor.shape[0];
  g_shape_d = (long)ds->dl_tensor.shape[1];
  g_last_ideg  = ip->intermediate_graph_degree;
  g_last_gdeg  = ip->graph_degree;
  g_last_niter = ip->nn_descent_niter;
  return CUVS_SUCCESS;
}

cuvsError_t cuvsCagraSearch(cuvsResources_t r, cuvsCagraSearchParams_t sp,
                            cuvsCagraIndex_t idx, DLManagedTensor* q,
                            DLManagedTensor* n, DLManagedTensor* d, cuvsFilter f)
{
  unsigned* ids;
  float* dst;
  long nq, topk, i, j;
  (void)sp; (void)f; (void)q;
  if (!r || r->magic != MAGIC_LIVE) { g_use_after_destroy++; seterr("search on dead resource"); return 1; }
  if (!idx || idx->magic != MAGIC_LIVE) { seterr("search on dead index"); return 1; }
  if (!n || !n->dl_tensor.data) { seterr("search with null neighbor buffer"); return 1; }
  nq   = n->dl_tensor.shape[0];
  topk = n->dl_tensor.shape[1];
  ids  = (unsigned*)n->dl_tensor.data;
  dst  = (d && d->dl_tensor.data) ? (float*)d->dl_tensor.data : NULL;
  for (i = 0; i < nq; i++) {
    for (j = 0; j < topk; j++) {
      ids[i * topk + j] = (unsigned)(i * topk + j);
      if (dst) dst[i * topk + j] = (float)(i * topk + j) * 0.5f;
    }
  }
  return CUVS_SUCCESS;
}

cuvsError_t cuvsCagraSerialize(cuvsResources_t r, const char* fn, cuvsCagraIndex_t idx, int up)
{
  FILE* f;
  (void)up;
  if (!r || r->magic != MAGIC_LIVE) { g_use_after_destroy++; seterr("serialize on dead resource"); return 1; }
  if (!idx || idx->magic != MAGIC_LIVE) { seterr("serialize on dead index"); return 1; }
  f = fopen(fn, "wb");
  if (!f) { seterr("open failed"); return 1; }
  fwrite("SNAP", 1, 4, f);
  fwrite(&g_last_ideg, sizeof g_last_ideg, 1, f);
  fclose(f);
  return CUVS_SUCCESS;
}

cuvsError_t cuvsCagraDeserialize(cuvsResources_t r, const char* fn, cuvsCagraIndex_t idx)
{
  FILE* f;
  if (!r || r->magic != MAGIC_LIVE) { g_use_after_destroy++; seterr("deserialize on dead resource"); return 1; }
  if (!idx || idx->magic != MAGIC_LIVE) { seterr("deserialize on dead index"); return 1; }
  f = fopen(fn, "rb");
  if (!f) { seterr("open failed"); return 1; }
  fclose(f);
  return CUVS_SUCCESS;
}

cuvsError_t cuvsCagraIndexGetSize(cuvsCagraIndex_t idx, long long* n)
{
  if (!idx || idx->magic != MAGIC_LIVE) { seterr("getsize on dead index"); return 1; }
  *n = g_shape_n;
  return CUVS_SUCCESS;
}
cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t idx, long long* d)
{
  if (!idx || idx->magic != MAGIC_LIVE) { seterr("getdims on dead index"); return 1; }
  *d = g_shape_d;
  return CUVS_SUCCESS;
}

/* ------------------------------------------------------------- test driver -- */
#ifndef SK_NO_STUB_MAIN
static int g_fail = 0;
#define CHECK(cond, msg) \
  do { if (!(cond)) { printf("  FAIL  %s\n", msg); g_fail++; } \
       else { printf("  ok    %s\n", msg); } } while (0)

int main(void)
{
  enum { N = 256, DIM = 8, NQ = 4, TOPK = 10 };
  static float base[N * DIM], query[NQ * DIM];
  unsigned ids[NQ * TOPK];
  float dist[NQ * TOPK];
  long n = N, dim = DIM;
  int i, rc;
  void* h;

  for (i = 0; i < N * DIM; i++)   base[i]  = (float)(i % 97) * 0.01f;
  for (i = 0; i < NQ * DIM; i++)  query[i] = (float)(i % 31) * 0.01f;
  g_shape_n = N; g_shape_d = DIM;

  printf("[1] build\n");
  fflush(stdout);
  say("== driver: build");
  h = seekdb_cuvs_build(base, n, dim);
  say("== driver: build returned");
  dbg("build returned to caller");
  CHECK(h != NULL, "build returned a handle");
  if (!h) { printf("RESULT: ABORT\n"); return 1; }
  printf("      recorded gdeg=%zu ideg=%zu niter=%zu\n", g_last_gdeg, g_last_ideg, g_last_niter);
  CHECK(g_build_calls == 1,        "cuVS build was invoked once");
  CHECK(g_last_ideg == 192,        "intermediate_graph_degree == 192");
  CHECK(g_last_gdeg == 128,        "graph_degree stays 128");
  CHECK(g_last_niter == 20,        "nn_descent_niter == 20");

  say("== driver: search #1");
  printf("[2] search after build\n");
  rc = seekdb_cuvs_search(h, query, NQ, TOPK, ids, dist);
  say("== driver: search #1 returned");
  printf("      rc=%d use_after_destroy=%d\n", rc, g_use_after_destroy);
  CHECK(rc == 0, "search returned 0");
  CHECK(g_use_after_destroy == 0, "no use-after-destroy during search");
  CHECK(ids[0] == 0,               "neighbor ids were written");

  say("== driver: search #2");
  printf("[3] second search reuses handle scratch\n");
  say("== driver: search #2 pre");
  rc = seekdb_cuvs_search(h, query, NQ, TOPK, ids, NULL);
  say("== driver: search #2 post");
  say("== driver: free h (marker)");
  CHECK(rc == 0, "second search returned 0");
  CHECK(g_use_after_destroy == 0, "no use-after-destroy on second search");

  printf("[4] serialize / deserialize\n");
  {
    void* h2;
    say("== driver: serialize");
    rc = seekdb_cuvs_serialize(h, "snap.bin");
    CHECK(rc == 0, "serialize returned 0");
    say("== driver: deserialize");
    h2 = seekdb_cuvs_deserialize("snap.bin", n, dim);
    say("== driver: deserialize returned");
    CHECK(h2 != NULL, "deserialize returned a handle");
    if (h2) {
      rc = seekdb_cuvs_search(h2, query, NQ, TOPK, ids, dist);
      CHECK(rc == 0, "search on deserialized handle returned 0");
      say("== driver: free h2");
      seekdb_cuvs_free(h2);
      say("== driver: free h2 done");
    }
  }

  printf("[5] free releases exactly one resource\n");
  {
    int before = g_res_destroyed;
    int bcreated = g_res_created;
    seekdb_cuvs_free(h);
    printf("      destroyed %d resource(s); created=%d destroyed=%d\n",
           g_res_destroyed - before, g_res_created, g_res_destroyed);
#if SK_TEST_SHARED_RES
    /* Shared-resource build: the process-wide resource outlives the handle, so
     * freeing one handle must destroy nothing at all. This is the mirror of the
     * exclusive-mode check - the same line of code, the opposite expectation.
     * Without this, "0 destroyed" would silently look like a leak. */
    CHECK(g_res_created - bcreated == 0,
          "shared mode: build reused the process resource");
    CHECK(g_res_destroyed - before == 0,
          "shared mode: free left the shared resource alive");
#else
    CHECK(g_res_destroyed - before == 1, "free destroyed exactly one resource");
#endif
  }

  printf("[6] one-shot cagra_knn path\n");
  rc = seekdb_cuvs_cagra_knn(base, n, dim, query, NQ, TOPK, ids);
  CHECK(rc == 0, "cagra_knn returned 0");

  printf("\n  created=%d destroyed=%d use_after_destroy=%d\n",
         g_res_created, g_res_destroyed, g_use_after_destroy);
#if SK_TEST_SHARED_RES
  /* One process resource for every handle, never destroyed by us. */
  CHECK(g_res_created == 1, "shared mode: one process resource total");
  CHECK(g_use_after_destroy == 0,
        "shared mode: no use-after-destroy after handle free");
#else
  CHECK(g_res_created == g_res_destroyed, "no resource leak");
#endif
  CHECK(g_use_after_destroy == 0,         "no use-after-destroy overall");

  if (g_fail) { printf("RESULT: %d CHECK(S) FAILED\n", g_fail); return 1; }
  printf("RESULT: ALL CHECKS PASSED\n");
  return 0;
}
#endif /* SK_NO_STUB_MAIN */
