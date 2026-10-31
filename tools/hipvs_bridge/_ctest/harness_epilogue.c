/* ============================================================== scenario ===*/

/* The stub's pool helper needs the bridge's own residency entry points, which
 * are static in the spliced body above. Expose them for the driver only. */
int   stub_res_pool_id(void) { return g_res ? g_res->id : -1; }

void  stub_pool_make_live(void)
{
  int shared = 0;
  cuvsResources_t r = sk_cuvs_res_acquire(&shared);
  if (r != NULL && shared) { sk_cuvs_res_release(r, shared); }
}

static long g_fail = 0;
#define CHECK(c, msg) do { \
  if (c) { printf("  ok    %s\n", msg); } \
  else   { printf("  FAIL  %s   (line %d)\n", msg, __LINE__); g_fail++; } \
} while (0)

static float* make_vecs(long n, long dim, unsigned seed)
{
  long i; float* p = (float*)malloc(sizeof(float) * n * dim);
  srand(seed);
  for (i = 0; i < n * dim; i++) { p[i] = (float)rand() / (float)RAND_MAX; }
  return p;
}

static void run(void)
{
  const long n = 2000, dim = 32, nq = 4, topk = 10;
  float* base = make_vecs(n, dim, 7);
  float* q    = make_vecs(nq, dim, 11);
  unsigned ids[64];
  float dist[64];
  int created0, destroyed0;
  void *h1, *h2, *h3;

  printf("\n  resource mode compiled in: shared=%d\n", seekdb_cuvs_shared_res_mode());

  printf("[1] build + search (first handle)\n");
  created0 = stub_res_created(); destroyed0 = stub_res_destroyed();
  h1 = seekdb_cuvs_build(base, n, dim);
  CHECK(h1 != NULL, "build returned a handle");
  CHECK(stub_build_calls() == 1, "cuVS build was invoked once");
  {
    const stub_build_opts* o = stub_last_build_opts();
    printf("      recorded gdeg=%zu ideg=%zu niter=%zu\n",
           o->graph_degree, o->intermediate_graph_degree, o->nn_descent_niter);
    CHECK(o->graph_degree == 128, "graph_degree stays 128");
    CHECK(o->intermediate_graph_degree == 192, "intermediate_graph_degree == 192");
    CHECK(o->nn_descent_niter == 20, "nn_descent_niter == 20");
  }
  printf("      after build: created=%d (+%d)  destroyed=%d (+%d)\n",
         stub_res_created(), stub_res_created() - created0,
         stub_res_destroyed(), stub_res_destroyed() - destroyed0);
  CHECK(stub_res_created() - created0 == 1, "build created exactly one resource");
  CHECK(stub_res_destroyed() - destroyed0 == 0, "build did not destroy it");

  CHECK(seekdb_cuvs_search(h1, q, nq, topk, ids, dist) == 0, "search on h1 returned 0");
  CHECK(stub_uaf() == 0, "no use-after-destroy during the first search");

  printf("[2] second handle built while h1 is still live\n");
  h2 = seekdb_cuvs_build(base, n, dim);
  CHECK(h2 != NULL, "second build returned a handle");
#if SK_CUVS_SHARED_RES
  CHECK(stub_res_created() - created0 == 1, "shared: second build reused the resource");
#endif
  CHECK(seekdb_cuvs_search(h2, q, nq, topk, ids, dist) == 0, "search on h2 returned 0");
  CHECK(stub_uaf() == 0, "no use-after-destroy building h2 next to h1");

  printf("[3] free h1 while h2 is STILL LIVE (this is the v6 bug)\n");
  seekdb_cuvs_free(h1);
  h1 = NULL;
  CHECK(stub_uaf() == 0, "freeing h1 did not touch a dead resource");
  CHECK(seekdb_cuvs_search(h2, q, nq, topk, ids, dist) == 0,
        "search on h2 AFTER h1 was freed returned 0");
  CHECK(stub_uaf() == 0, "no use-after-destroy: h2 survived h1's free");

  printf("[4] serialize / deserialize after the sibling handle is gone\n");
  CHECK(seekdb_cuvs_serialize(h2, "stub_v7.snapshot") == 0, "serialize h2 returned 0");
  CHECK(stub_uaf() == 0, "no use-after-destroy during serialize");
  h3 = seekdb_cuvs_deserialize("stub_v7.snapshot", n, dim);
  CHECK(h3 != NULL, "deserialize returned a handle");
  CHECK(seekdb_cuvs_search(h3, q, nq, topk, ids, dist) == 0, "search on h3 returned 0");
  CHECK(stub_uaf() == 0, "no use-after-destroy on the deserialized handle");

  printf("[5] free everything\n");
  seekdb_cuvs_free(h2);
  seekdb_cuvs_free(h3);
  printf("      created=%d destroyed=%d uaf=%d live_idx=%d\n",
         stub_res_created(), stub_res_destroyed(), stub_uaf(), stub_live_handles());
#if SK_CUVS_SHARED_RES
  CHECK(stub_res_created() <= 1, "shared: never created more than one resource");
  CHECK(stub_res_destroyed() == 0, "shared: process resource outlived every handle");
#else
  CHECK(stub_res_created() == stub_res_destroyed(), "exclusive: every resource destroyed");
#endif
  CHECK(stub_uaf() == 0, "no use-after-destroy overall");
  CHECK(stub_live_handles() == 0, "no live cuVS index left behind");

  printf("[6] one-shot cagra_knn path\n");
  CHECK(seekdb_cuvs_cagra_knn(base, n, dim, q, nq, topk, ids) == 0, "cagra_knn returned 0");
  CHECK(stub_uaf() == 0, "no use-after-destroy in the one-shot path");

#if SK_CUVS_SHARED_RES
  printf("[7] mixed lifetime: a handle whose allocator is NOT the live pool\n");
  /* The situation the two allocator identities below exist for: the scratch was
   * allocated from resource A, and by the time free() runs the handle's live
   * resource is B. Rather than assume it (which is how the first version of this
   * test compared a pointer to itself and passed on a broken free), build it:
   * take the pool pointer, hand the handle its own resource, and keep the
   * handle's id as the "allocator". The driver then only needs free() to hand
   * the scratch back to A. */
  {
    void* hx;
    cuvsResources_t pool;
    int uaf0;

    stub_pool_make_live();
    pool = g_res;
    CHECK(pool != NULL, "the shared pool is up before the build");

    hx = seekdb_cuvs_build(base, n, dim);
    CHECK(hx != NULL, "build with the pool already up returned a handle");

    /* Warm the scratch once on the pool, then hand the handle a resource of its
     * own and DROP the scratch. Dropping it is the part that matters: nd_cap is
     * sized for the largest nq ever seen, so merely zeroing nd_cap would let the
     * next search hit `nd_cap >= need` and reuse a buffer that still belongs to
     * the pool -- the old tap measure that made this case a comparison of a
     * pointer with itself. The warm-up frees on the pool, which is its owner. */
    CHECK(seekdb_cuvs_search(hx, q, nq, topk, ids, dist) == 0, "first search (pool) returned 0");
    {
      cuvsResources_t own = NULL;
      if (cuvsResourcesCreate(&own) != CUVS_SUCCESS) { own = NULL; }
      CHECK(own != NULL, "created a second resource for the handle");
      CHECK(own != pool, "own resource and pool are two identities");
      cuvsRMMFree(pool, ((sk_cuvs_index*)hx)->nd, ((sk_cuvs_index*)hx)->nd_cap * sizeof(unsigned));
      cuvsRMMFree(pool, ((sk_cuvs_index*)hx)->dd, ((sk_cuvs_index*)hx)->nd_cap * sizeof(float));
      ((sk_cuvs_index*)hx)->nd = NULL;
      ((sk_cuvs_index*)hx)->dd = NULL;
      ((sk_cuvs_index*)hx)->nd_cap = 0;
      ((sk_cuvs_index*)hx)->scratch_res = NULL;
      ((sk_cuvs_index*)hx)->res = own;
      ((sk_cuvs_index*)hx)->res_owned = 1;
    }

    /* search must now allocate the scratch on the allocator, not on the pool:
     * if it ran on the pool this whole case would collapse into comparing a
     * pointer to itself -- which is exactly how the first version of this test
     * passed on a broken free. */
    uaf0 = stub_uaf();
    CHECK(seekdb_cuvs_search(hx, q, nq, topk, ids, dist) == 0, "search on the own-resource handle returned 0");
    printf("      handle #%d  pool #%d  scratch #%d\n",
           ((sk_cuvs_index*)hx)->res->id, pool->id,
           stub_block_resource_id(((sk_cuvs_index*)hx)->nd));
    CHECK(stub_block_resource_id(((sk_cuvs_index*)hx)->nd) == ((sk_cuvs_index*)hx)->res->id,
          "scratch was allocated from the handle's own resource");

    /* Now the pool is the only thing a "just re-resolve the live resource"
     * implementation could see, and it is the WRONG allocator. */
    uaf0 = stub_uaf();
    seekdb_cuvs_free(hx);
    printf("      uaf %d -> %d\n", uaf0, stub_uaf());
    CHECK(stub_uaf() == uaf0, "free used the allocating resource, not the live pool");
    CHECK(g_res != NULL && g_res->magic == MAGIC_LIVE,
          "free did not destroy the pool that other handles still use");
  }
#else
  printf("[7] (skipped: no shared pool in this build, nothing to mix)\n");
#endif

  printf("[8] mutation control: a stale resource MUST be reported\n");
  {
    int before = stub_uaf();
    stub_uaf_probe_stale();
    printf("      injected stale-resource use: uaf %d -> %d\n", before, stub_uaf());
    CHECK(stub_uaf() > before, "stub reports use-after-destroy when it happens (has teeth)");
  }

  free(base); free(q);
  printf("\n  RESULT: %s (%ld failure(s))\n",
         g_fail == 0 ? "ALL CHECKS PASSED" : "FAILURES", g_fail);
  exit(g_fail == 0 ? 0 : 1);
}

int main(void) { run(); return 0; }
