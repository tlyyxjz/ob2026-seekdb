/* Counters the test driver needs to judge the bridge; bodies are in
 * stub_runtime.c. Declared here so the driver does not depend on build-time
 * knowledge of whether the stub's own main() is compiled in. */
#ifndef STUB_IFACE_H
#define STUB_IFACE_H

#include <stddef.h>

typedef struct {
  size_t graph_degree;
  size_t intermediate_graph_degree;
  size_t nn_descent_niter;
} stub_build_opts;

int         stub_res_created(void);
int         stub_res_destroyed(void);
int         stub_uaf(void);
int         stub_build_calls(void);
int         stub_live_handles(void);
/* Identity of the pool resource (creation order index), -1 when it does not
 * exist. Lets the driver check identity instead of comparing freed pointers. */
int         stub_res_pool_id(void);
/* Identity (creation index) of the resource the n-th most recent build used. */
int         stub_idx_resource_id(int back);
/* Identity of the resource that owns the block at `p` (-1 if not ours). */
int         stub_block_resource_id(void* p);
const stub_build_opts* stub_last_build_opts(void);

/* Deliberately use a destroyed resource exactly once, so the driver can prove
 * the stub really does report use-after-destroy (a suite that cannot fail
 * proves nothing). */
void        stub_uaf_probe_stale(void);

/* Create the bridge's shared pool now (no-op when the exclusive build is in
 * effect), so a subsequent exclusive build runs with a different resource live. */
void        stub_pool_make_live(void);

#endif
