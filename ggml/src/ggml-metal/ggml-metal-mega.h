#pragma once

// GGML_METAL_MEGA: one persistent kernel runs a graph's serial steps, with a grid barrier between them, instead of a
// dispatch a step. its source is generated from the kernel sources of the pipelines a graph uses: each kernel becomes a
// device function (attributes stripped, every device pointer coherent, function constants made compile-time per
// variant, a variant of a kind a namespace of its own), and the persistent kernel calls them by a case number.
// measured on an M5 Ultra (stepbench/pdl.m): a 2304x2304 int8 mat-vec step costs 8.35 us this way, 9.30 us with an
// encoder a step

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_METAL_MEGA_MAX_FC 48

struct ggml_metal_mega_pipe {
    const char * base;   // the kernel's host name
    int          kind;   // index of its source in kind_srcs
    int          n_fc;
    int          fc_idx [GGML_METAL_MEGA_MAX_FC];
    int          fc_type[GGML_METAL_MEGA_MAX_FC]; // 0 bool, 1 short, 2 int
    int64_t      fc_val [GGML_METAL_MEGA_MAX_FC];
};

// flags of a pipeline in the generated source
#define GGML_METAL_MEGA_FLAG_OK          1 // has a case
#define GGML_METAL_MEGA_FLAG_NO_BARRIER  2 // its threads may sit out (no threadgroup barrier inside)
#define GGML_METAL_MEGA_FLAG_CONST_SHIFT 8 // bit 8+i: binding i is a constant parameter (set_bytes)

// hdr: the shared headers (common.h, dequantize.h, quantize.h, ggml-common.h, ggml-metal-impl.h flattened);
// kind_srcs[k]: the source of kind k after those headers. out_case[i] = case of pipes[i] (or -1), out_flags[i]
// returns a malloc'ed source, or NULL
char * ggml_metal_mega_source(const char * hdr, const char * const * kind_srcs, int n_kinds,
                              const struct ggml_metal_mega_pipe * pipes, int n_pipes, int * out_case, int * out_flags);

#ifdef __cplusplus
}
#endif
