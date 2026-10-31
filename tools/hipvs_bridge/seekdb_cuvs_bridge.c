/* seekdb <-> hipVS bridge: plain-C CAGRA kNN; all cuVS/ROCm deps live here. */
#include "seekdb_cuvs_bridge.h"
#include <cuvs/core/c_api.h>
#include <cuvs/neighbors/cagra.h>
#include <dlpack/dlpack.h>
#ifdef __HIP_PLATFORM_AMD__
#include <cuvs/cuda_runtime.h>
#else
#include <cuda_runtime.h>
#endif
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Upper bound of the top-k one handle may be asked for. The contest gates query
 * Top-10 and Top-100; keep scratch sizing at the larger value. */
#define CUVS_SCRATCH_TOPK 128

typedef struct {
  cuvsResources_t         res;   /* borrowed from the process-wide holder */
  cuvsCagraIndexParams_t  ip;
  cuvsCagraIndex_t        idx;
  cuvsCagraSearchParams_t sp;    /* built once, reused per search */
  unsigned*               nd;    /* search scratch: nq x CUVS_SCRATCH_TOPK */
  float*                  dd;    /* search scratch: nq x CUVS_SCRATCH_TOPK */
  size_t                  nd_cap;/* capacity in elements */
  long n, dim;
} sk_cuvs_index;

/* Shared-resource reuse is OFF by default.
 *
 * Reusing one cuVS resource removes the per-round context create/destroy cost,
 * but it also makes every CAGRA handle share one resource, and cuVS does not
 * document how a shared resource behaves across several live indexes (memory
 * pool ownership in particular). The 4-point hard gate depends on the build
 * path staying exactly as validated, so the default build keeps the
 * one-resource-per-handle lifetime. Define SEEKDB_CUVS_SHARED_RES=1 to opt in
 * and measure it. */
#ifdef SEEKDB_CUVS_SHARED_RES
#define SK_CUVS_SHARED_RES 1
#else
#define SK_CUVS_SHARED_RES 0
#endif

static pthread_mutex_t g_res_mu = PTHREAD_MUTEX_INITIALIZER;
static cuvsResources_t g_res = NULL;
static int             g_res_state = 0;  /* 0 = untried, 1 = ready, -1 = failed */

/* Returns the resource this call must use and sets *shared to whether the
 * caller must NOT destroy it. Shared mode serializes every use. */
static cuvsResources_t sk_cuvs_res_acquire(int* shared)
{
#if SK_CUVS_SHARED_RES
  *shared = 1;
  pthread_mutex_lock(&g_res_mu);
  if (g_res_state == 0) {
    g_res_state = (cuvsResourcesCreate(&g_res) == CUVS_SUCCESS) ? 1 : -1;
  }
  if (g_res_state != 1) { pthread_mutex_unlock(&g_res_mu); return NULL; }
  return g_res;
#else
  cuvsResources_t res = NULL;
  *shared = 0;
  if (cuvsResourcesCreate(&res) != CUVS_SUCCESS) { return NULL; }
  return res;
#endif
}

static void sk_cuvs_res_release(cuvsResources_t res, int shared)
{
#if SK_CUVS_SHARED_RES
  (void)res;
  if (shared) { pthread_mutex_unlock(&g_res_mu); }
#else
  (void)shared;
  if (res != NULL) { cuvsResourcesDestroy(res); }
#endif
}

/* Search params are pure configuration: create once, reuse. */
static cuvsCagraSearchParams_t sk_cuvs_search_params(long topk)
{
  cuvsCagraSearchParams_t sp = NULL;
  size_t required_itopk = (size_t)topk * 4;
  if (cuvsCagraSearchParamsCreate(&sp) != CUVS_SUCCESS) { return NULL; }
  if (required_itopk < 512) required_itopk = 512;
  if (sp->itopk_size < required_itopk) sp->itopk_size = required_itopk;
  sp->search_width = 4;
  return sp;
}

/* Borrow the shared resource and grow the handle's scratch buffers if needed.
 * Caller must already hold g_res_mu via sk_cuvs_res_acquire(). */
static int sk_cuvs_scratch_reserve(sk_cuvs_index* h, cuvsResources_t res, long nq)
{
  const size_t need = (size_t)nq * (size_t)CUVS_SCRATCH_TOPK;
  if (h->nd_cap >= need) { return 0; }
  if (h->nd) { cuvsRMMFree(res, h->nd, h->nd_cap * sizeof(unsigned)); h->nd = NULL; }
  if (h->dd) { cuvsRMMFree(res, h->dd, h->nd_cap * sizeof(float));    h->dd = NULL; }
  h->nd_cap = 0;
  if (cuvsRMMAlloc(res, (void**)&h->nd, need * sizeof(unsigned)) != CUVS_SUCCESS ||
      cuvsRMMAlloc(res, (void**)&h->dd, need * sizeof(float))    != CUVS_SUCCESS) {
    if (h->nd) { cuvsRMMFree(res, h->nd, need * sizeof(unsigned)); h->nd = NULL; }
    if (h->dd) { cuvsRMMFree(res, h->dd, need * sizeof(float));    h->dd = NULL; }
    return 1;
  }
  h->nd_cap = need;
  return 0;
}

static void sk_cuvs_fill_tensor(DLManagedTensor* t, void* data, long* shape,
                                int ndim, DLDataTypeCode code, int bits)
{
  memset(t, 0, sizeof(*t));
  t->dl_tensor.data = data;
  t->dl_tensor.device.device_type = kDLCUDA;
  t->dl_tensor.device.device_id = 0;
  t->dl_tensor.ndim = ndim;
  t->dl_tensor.dtype.code = code;
  t->dl_tensor.dtype.bits = (uint8_t)bits;
  t->dl_tensor.dtype.lanes = 1;
  t->dl_tensor.shape = shape;
  t->dl_tensor.strides = 0;
  t->dl_tensor.byte_offset = 0;
}

/* ---------------------------------------------------------------- build tune
 *
 * The contest clocks end-to-end index construction, so the knobs below are the
 * scoring surface. The shipped defaults (NN_DESCENT / graph_degree=128 /
 * intermediate_graph_degree=256 / niter=100) spend build time on
 * over-provisioning that a recall@10 gate cannot see:
 *
 *  - intermediate_graph_degree=256 for graph_degree=128 sits at the very top of
 *    cuVS's own recommended band. From cagra.hpp graph_build_heuristic: "a good
 *    practice is to have it 1.5x to 2x of the desired graph_degree and a
 *    multiple of 32" -> 192 for 128. In the NN-Descent path this value is the
 *    graph width (nn_descent.cuh builds GnndGraph with node_degree =
 *    intermediate_node_degree), so it scales the per-iteration candidate work.
 *    graph_degree, which is what survives into the searched graph, is unchanged.
 *    192 is a multiple of both 32 and 64, so align64::roundUp and the
 *    segment_size assertions in nn_descent.cuh are unaffected.
 *
 *  - nn_descent_niter=100 is only an upper bound: nn_descent.cuh breaks out once
 *    update_counter_ < termination_threshold * nrow * dim / counter_interval
 *    (after at least 5 iterations). Lowering it to the library default of 20 is
 *    therefore not expected to truncate a converged graph, and it removes the
 *    worst case where termination never triggers.
 *
 * Both are overridable from the environment so the values can be re-measured on
 * the target GPU without a rebuild. */
#ifndef SK_BUILD_NITER
#define SK_BUILD_NITER 20
#endif
#ifndef SK_BUILD_INTERMEDIATE_DEGREE
#define SK_BUILD_INTERMEDIATE_DEGREE 192
#endif
#ifndef SK_BUILD_GRAPH_DEGREE
#define SK_BUILD_GRAPH_DEGREE 128
#endif

static size_t sk_env_size(const char* name, size_t dflt, size_t lo, size_t hi)
{
  const char* s = getenv(name);
  long v;
  if (s == NULL || *s == '\0') { return dflt; }
  v = strtol(s, NULL, 10);
  if (v < (long)lo || v > (long)hi) { return dflt; }
  return (size_t)v;
}

void* seekdb_cuvs_build(const float* base, long n, long dim)
{
  sk_cuvs_index* h = (sk_cuvs_index*)calloc(1, sizeof(*h));
  cuvsResources_t res = NULL;
  int shared = 0;
  DLManagedTensor dset; long ds[2] = {n, dim};
  size_t gdeg, ideg, niter;
  if (!h) { return NULL; }
  if ((res = sk_cuvs_res_acquire(&shared)) == NULL) { free(h); return NULL; }
  h->res = res;

  /* Input stays host-resident: cuVS copies it to the device during build, which
   * is exactly the "data scan + CPU/GPU transfer" stage the contest clocks. */
  memset(&dset, 0, sizeof(dset));
  dset.dl_tensor.data = (void*)base;
  dset.dl_tensor.device.device_type = kDLCPU;
  dset.dl_tensor.device.device_id = 0;
  dset.dl_tensor.ndim = 2;
  dset.dl_tensor.dtype.code = kDLFloat; dset.dl_tensor.dtype.bits = 32; dset.dl_tensor.dtype.lanes = 1;
  dset.dl_tensor.shape = ds; dset.dl_tensor.strides = 0; dset.dl_tensor.byte_offset = 0;

  if (cuvsCagraIndexParamsCreate(&h->ip) != CUVS_SUCCESS) { goto fail_ip; }

  gdeg  = sk_env_size("SK_BUILD_GRAPH_DEGREE", SK_BUILD_GRAPH_DEGREE, 16, 4096);
  ideg  = sk_env_size("SK_BUILD_INTERMEDIATE_DEGREE", SK_BUILD_INTERMEDIATE_DEGREE, gdeg, 8192);
  niter = sk_env_size("SK_BUILD_NITER", SK_BUILD_NITER, 1, 512);
  /* cuVS clamps these two internally; doing it here keeps the log honest. */
  if (n > 1 && ideg >= (size_t)n) { ideg = (size_t)n - 1; }
  if (ideg < gdeg) { gdeg = ideg; }

  h->ip->build_algo                = NN_DESCENT;
  h->ip->graph_degree              = gdeg;
  h->ip->intermediate_graph_degree = ideg;
  h->ip->nn_descent_niter          = niter;
  fprintf(stderr,
          "[seekdb_cuvs] build n=%ld dim=%ld algo=NN_DESCENT gdeg=%zu ideg=%zu niter=%zu\n",
          n, dim, gdeg, ideg, niter);
  if (cuvsCagraIndexCreate(&h->idx) != CUVS_SUCCESS) { goto fail_idx; }
  if (cuvsCagraBuild(res, h->ip, &dset, h->idx) != CUVS_SUCCESS) { goto fail_build; }
  h->sp = sk_cuvs_search_params(CUVS_SCRATCH_TOPK);
  if (h->sp == NULL) { goto fail_build; }
  h->n = n; h->dim = dim;
  sk_cuvs_res_release(res, shared);
  return h;

fail_build:
  cuvsCagraIndexDestroy(h->idx);
fail_idx:
  cuvsCagraIndexParamsDestroy(h->ip);
fail_ip:
  sk_cuvs_res_release(res, shared);
  fprintf(stderr, "[seekdb_cuvs] build failed: %s\n", cuvsGetLastErrorText());
  free(h);
  return NULL;
}

int seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                       unsigned int* out_ids, float* out_dist)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  cuvsResources_t res = NULL;
  int shared = 0;
  float* qd = 0;
  int rc = 0;
  if (h == NULL) { return 1; }
  if (topk <= 0 || topk > CUVS_SCRATCH_TOPK || nq <= 0) { return 2; }
  if ((res = sk_cuvs_res_acquire(&shared)) == NULL) { return 1; }

  if (sk_cuvs_scratch_reserve(h, res, nq) != 0) { rc = 12; goto out; }
  if (cuvsRMMAlloc(res, (void**)&qd, sizeof(float)*nq*h->dim) != CUVS_SUCCESS) { rc = 10; goto out; }
  if (cudaMemcpy(qd, query, sizeof(float)*nq*h->dim, cudaMemcpyDefault) != cudaSuccess) { rc = 13; goto out; }

  {
    DLManagedTensor qt, nt, dt;
    long qs[2] = {nq, h->dim}, nsh[2] = {nq, topk}, dsh[2] = {nq, topk};
    cuvsCagraSearchParams_t sp = h->sp;
    cuvsCagraSearchParams_t tmp = NULL;
    cuvsFilter f;

    sk_cuvs_fill_tensor(&qt, qd,      qs,  2, kDLFloat,  32);
    sk_cuvs_fill_tensor(&nt, h->nd,   nsh, 2, kDLUInt,   32);
    sk_cuvs_fill_tensor(&dt, h->dd,   dsh, 2, kDLFloat,  32);

    if (sp == NULL || sp->itopk_size < (size_t)topk * 4) {
      /* topk beyond what the cached params were sized for. */
      tmp = sk_cuvs_search_params(topk);
      if (tmp == NULL) { rc = 14; goto out; }
      sp = tmp;
    }
    f.type = NO_FILTER; f.addr = (uintptr_t)0;
    if (cuvsCagraSearch(res, sp, h->idx, &qt, &nt, &dt, f) != CUVS_SUCCESS) { rc = 15; }
    if (tmp) { cuvsCagraSearchParamsDestroy(tmp); }
    if (rc != 0) { goto out; }
  }

  if (cudaMemcpy(out_ids, h->nd, sizeof(unsigned)*nq*topk, cudaMemcpyDefault) != cudaSuccess) {
    rc = 16;
  } else if (out_dist != NULL &&
             cudaMemcpy(out_dist, h->dd, sizeof(float)*nq*topk, cudaMemcpyDefault) != cudaSuccess) {
    rc = 17;
  }

out:
  if (qd) { cuvsRMMFree(res, qd, sizeof(float)*nq*h->dim); }
  sk_cuvs_res_release(res, shared);
  if (rc) { fprintf(stderr, "[seekdb_cuvs] search rc=%d: %s\n", rc, cuvsGetLastErrorText()); }
  return rc;
}

void seekdb_cuvs_free(void* handle)
{
  sk_cuvs_index* h = (sk_cuvs_index*)handle;
  cuvsResources_t res = NULL;
  int shared = 0;
  if (h == NULL) { return; }
  if ((res = sk_cuvs_res_acquire(&shared)) != NULL) {
    if (h->nd) { cuvsRMMFree(res, h->nd, h->nd_cap * sizeof(unsigned)); }
    if (h->dd) { cuvsRMMFree(res, h->dd, h->nd_cap * sizeof(float)); }
    sk_cuvs_res_release(res, shared);
  }
  cuvsCagraIndexDestroy(h->idx);
  cuvsCagraIndexParamsDestroy(h->ip);
  if (h->sp) { cuvsCagraSearchParamsDestroy(h->sp); }
  free(h);
}

int seekdb_cuvs_serialize(void* handle, const char* filename)
{
  sk_cuvs_index* index = (sk_cuvs_index*)handle;
  cuvsResources_t res = NULL;
  int shared = 0;
  int rc = 0;
  if (!index || !filename) return 1;
  if ((res = sk_cuvs_res_acquire(&shared)) == NULL) return 1;
  if (cuvsCagraSerialize(res, filename, index->idx, 1) != CUVS_SUCCESS) {
    fprintf(stderr, "[seekdb_cuvs] serialize failed: %s\n", cuvsGetLastErrorText());
    rc = 1;
  }
  sk_cuvs_res_release(res, shared);
  return rc;
}

void* seekdb_cuvs_deserialize(const char* filename, long n, long dim)
{
  sk_cuvs_index* index = NULL;
  cuvsResources_t res = NULL;
  int shared = 0;
  if (!filename || n <= 0 || dim <= 0) return NULL;
  index = (sk_cuvs_index*)calloc(1, sizeof(*index));
  if (!index) return NULL;
  if ((res = sk_cuvs_res_acquire(&shared)) == NULL) goto fail;
  index->res = res;
  if (cuvsCagraIndexParamsCreate(&index->ip) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraIndexCreate(&index->idx) != CUVS_SUCCESS) goto fail_ip;
  if (cuvsCagraDeserialize(res, filename, index->idx) != CUVS_SUCCESS) goto fail_idx;
  {
    int64_t actual_n = 0;
    int64_t actual_dim = 0;
    if (cuvsCagraIndexGetSize(index->idx, &actual_n) != CUVS_SUCCESS ||
        cuvsCagraIndexGetDims(index->idx, &actual_dim) != CUVS_SUCCESS ||
        actual_n != n || actual_dim != dim) {
      fprintf(stderr, "[seekdb_cuvs] snapshot shape mismatch: %ld x %ld, expected %ld x %ld\n",
              (long)actual_n, (long)actual_dim, n, dim);
      goto fail_idx;
    }
  }
  index->sp = sk_cuvs_search_params(CUVS_SCRATCH_TOPK);
  if (index->sp == NULL) goto fail_idx;
  index->n = n;
  index->dim = dim;
  sk_cuvs_res_release(res, shared);
  return index;
fail_idx:
  cuvsCagraIndexDestroy(index->idx);
fail_ip:
  cuvsCagraIndexParamsDestroy(index->ip);
  sk_cuvs_res_release(res, shared);
fail:
  fprintf(stderr, "[seekdb_cuvs] deserialize failed: %s\n", cuvsGetLastErrorText());
  free(index);
  return NULL;
}

int seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                          const float* query, long nq, long topk,
                          unsigned int* out_ids)
{
  void* h = seekdb_cuvs_build(base, n, dim);
  int rc;
  if (!h) return 10;
  rc = seekdb_cuvs_search(h, query, nq, topk, out_ids, NULL);
  seekdb_cuvs_free(h);
  return rc;
}
