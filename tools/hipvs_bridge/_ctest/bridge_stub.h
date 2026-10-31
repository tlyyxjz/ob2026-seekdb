/* Declarations for the stub cuVS/ROCm/DLPack surface used by the seekdb bridge.
 *
 * Types only plus function prototypes; every body lives in stub_runtime.c so
 * the bridge and the test driver can both include this without duplicate
 * symbols. Handles carry a magic word so using a destroyed one is detectable.
 */
#ifndef BRIDGE_STUB_H
#define BRIDGE_STUB_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>

/* ---------------------------------------------------------------- DLPack --- */
#define kDLCPU  1
#define kDLCUDA 2

typedef enum { kDLInt = 0, kDLUInt = 1, kDLFloat = 2 } DLDataTypeCode;
typedef struct { int code; int bits; int lanes; } DLDataType;
typedef struct { int device_type; int device_id; } DLDevice;
typedef struct { void* data; DLDevice device; int ndim; DLDataType dtype;
                 long* shape; long* strides; unsigned long long byte_offset; } DLTensor;
typedef struct { DLTensor dl_tensor; void* manager_ctx; void* deleter; } DLManagedTensor;

/* --------------------------------------------------------------- cuVS core -- */
typedef int cuvsError_t;
#define CUVS_SUCCESS 0
#define MAGIC_LIVE   0xC0FFEE01u
#define MAGIC_DEAD   0xDEADDEADu

struct cuvsResources_s { unsigned magic; int id; };
typedef struct cuvsResources_s* cuvsResources_t;

enum cuvsFilterType { NO_FILTER, BITSET, CUVS_BITMAP };
typedef struct { uintptr_t addr; enum cuvsFilterType type; } cuvsFilter;

cuvsError_t cuvsResourcesCreate(cuvsResources_t* out);
cuvsError_t cuvsResourcesDestroy(cuvsResources_t r);
const char* cuvsGetLastErrorText(void);
cuvsError_t cuvsRMMAlloc(cuvsResources_t r, void** p, size_t n);
cuvsError_t cuvsRMMFree(cuvsResources_t r, void* p, size_t n);

/* ------------------------------------------------------------- CUDA shims -- */
typedef int cudaError_t;
#define cudaSuccess 0
#define cudaMemcpyDefault 4
cudaError_t cudaMemcpy(void* d, const void* s, size_t n, int kind);

/* ------------------------------------------------------------------ cagra -- */
typedef enum { L2Expanded } cuvsDistanceType;
enum cuvsCagraGraphBuildAlgo { AUTO_SELECT, IVF_PQ, NN_DESCENT, ITERATIVE_CAGRA_SEARCH };

struct cuvsCagraIndexParams {
  cuvsDistanceType metric;
  size_t intermediate_graph_degree;
  size_t graph_degree;
  enum cuvsCagraGraphBuildAlgo build_algo;
  size_t nn_descent_niter;
  void* compression;
  void* graph_build_params;
};
typedef struct cuvsCagraIndexParams* cuvsCagraIndexParams_t;

struct cuvsCagraSearchParams {
  size_t max_queries; size_t itopk_size; size_t search_width; size_t team_size;
  double min_iterations; double max_iterations;
};
typedef struct cuvsCagraSearchParams* cuvsCagraSearchParams_t;

struct cuvsCagraIndex_s { unsigned magic; };
typedef struct cuvsCagraIndex_s* cuvsCagraIndex_t;

cuvsError_t cuvsCagraIndexParamsCreate(cuvsCagraIndexParams_t* p);
cuvsError_t cuvsCagraIndexParamsDestroy(cuvsCagraIndexParams_t p);
cuvsError_t cuvsCagraSearchParamsCreate(cuvsCagraSearchParams_t* p);
cuvsError_t cuvsCagraSearchParamsDestroy(cuvsCagraSearchParams_t p);
cuvsError_t cuvsCagraIndexCreate(cuvsCagraIndex_t* p);
cuvsError_t cuvsCagraIndexDestroy(cuvsCagraIndex_t p);
cuvsError_t cuvsCagraBuild(cuvsResources_t r, cuvsCagraIndexParams_t ip,
                           DLManagedTensor* ds, cuvsCagraIndex_t idx);
cuvsError_t cuvsCagraSearch(cuvsResources_t r, cuvsCagraSearchParams_t sp,
                            cuvsCagraIndex_t idx, DLManagedTensor* q,
                            DLManagedTensor* n, DLManagedTensor* d, cuvsFilter f);
cuvsError_t cuvsCagraSerialize(cuvsResources_t r, const char* fn, cuvsCagraIndex_t idx, int up);
cuvsError_t cuvsCagraDeserialize(cuvsResources_t r, const char* fn, cuvsCagraIndex_t idx);
cuvsError_t cuvsCagraIndexGetSize(cuvsCagraIndex_t idx, long long* n);
cuvsError_t cuvsCagraIndexGetDims(cuvsCagraIndex_t idx, long long* d);

/* What the fake build recorded, so the test can assert the shipped knobs. */
void dbg(const char* m);
void* seekdb_cuvs_build(const float* base, long n, long dim);
int   seekdb_cuvs_search(void* h, const float* q, long nq, long topk,
                         unsigned int* ids, float* dist);
int   seekdb_cuvs_serialize(void* h, const char* path);
void* seekdb_cuvs_deserialize(const char* path, long n, long dim);
void  seekdb_cuvs_free(void* h);
int   seekdb_cuvs_shared_res_mode(void);
int   seekdb_cuvs_cagra_knn(const float* base, long n, long dim,
                            const float* q, long nq, long topk, unsigned int* ids);
extern size_t g_last_ideg, g_last_gdeg, g_last_niter;
extern int    g_build_calls, g_res_created, g_res_destroyed, g_use_after_destroy;
extern long   g_shape_n, g_shape_d;

#endif
