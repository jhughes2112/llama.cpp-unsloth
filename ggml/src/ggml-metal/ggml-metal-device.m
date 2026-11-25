#import "ggml-metal-device.h"
#import "ggml-metal-fusion.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"
#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-mega.h"

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>

#include <stdatomic.h>
#include <unistd.h>

#ifndef TARGET_OS_VISION
#define TARGET_OS_VISION 0
#endif

// create residency sets only on macOS >= 15.0
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000 || \
    TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_TV && __TV_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_VISION && __VISION_OS_VERSION_MAX_ALLOWED >= 200000
#define GGML_METAL_HAS_RESIDENCY_SETS 1
#endif

// overload of MTLGPUFamilyMetalX (not available in some environments)
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;
static const NSInteger MTLGPUFamilyMetal4_GGML = 5002;

// MTLLanguageVersion4_0 is not present in older SDKs
static const NSUInteger MTLLanguageVersion4_0_GGML = 4 << 16;

#if !GGML_METAL_EMBED_LIBRARY
// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end
#endif

struct ggml_metal_library;
static void ggml_metal_mega_set_library(struct ggml_metal_library * lib);
static void ggml_metal_mega_forget_library(struct ggml_metal_library * lib);
static bool ggml_metal_mega_included(const char * base);
struct ggml_metal_mega_rec;
static bool ggml_metal_mega_stale(const struct ggml_metal_mega_rec * r, int seg_d0, int d0, int d1);

//
// MTLFunctionConstantValues wrapper
//

struct ggml_metal_cv {
    MTLFunctionConstantValues * obj;

    // the values again, for GGML_METAL_MEGA (which compiles them into its source)
    int     n;
    int     idx [GGML_METAL_MEGA_MAX_FC];
    int     type[GGML_METAL_MEGA_MAX_FC];
    int64_t val [GGML_METAL_MEGA_MAX_FC];
};

static void ggml_metal_cv_keep(ggml_metal_cv_t cv, int64_t value, int type, int32_t idx) {
    for (int i = 0; i < cv->n; ++i) {
        if (cv->idx[i] == idx) {
            cv->val[i] = value;
            cv->type[i] = type;
            return;
        }
    }
    if (cv->n < GGML_METAL_MEGA_MAX_FC) {
        cv->idx [cv->n] = idx;
        cv->type[cv->n] = type;
        cv->val [cv->n] = value;
        cv->n++;
    }
}

ggml_metal_cv_t ggml_metal_cv_init(void) {
    ggml_metal_cv_t res = calloc(1, sizeof(struct ggml_metal_cv));

    res->obj = [[MTLFunctionConstantValues alloc] init];

    return res;
}

void ggml_metal_cv_free(ggml_metal_cv_t cv) {
    [cv->obj release];
    free(cv);
}

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeShort atIndex:idx];
    ggml_metal_cv_keep(cv, value, 1, idx);
}

void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeInt atIndex:idx];
    ggml_metal_cv_keep(cv, value, 2, idx);
}

void ggml_metal_cv_set_bool(ggml_metal_cv_t cv, bool value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeBool atIndex:idx];
    ggml_metal_cv_keep(cv, value ? 1 : 0, 0, idx);
}

//
// MTLComputePipelineState wrapper
//

struct ggml_metal_pipeline {
    id<MTLComputePipelineState> obj;

    // for GGML_METAL_MEGA
    char    base[128];  // host name of the kernel
    int     kind;       // the library (source) it is in, -1 if unknown
    int     n_fc;
    int     fc_idx [GGML_METAL_MEGA_MAX_FC];
    int     fc_type[GGML_METAL_MEGA_MAX_FC];
    int64_t fc_val [GGML_METAL_MEGA_MAX_FC];
    int     mega_gen;   // generation of the megakernel its case is in (0: none)
    int     mega_case;
    int     mega_flags;
    bool    mega_never; // its kind is not supported
};

ggml_metal_pipeline_t ggml_metal_pipeline_init(void) {
    ggml_metal_pipeline_t res = calloc(1, sizeof(struct ggml_metal_pipeline));

    *res = (struct ggml_metal_pipeline) {
        /*.obj  =*/ nil,
    };

    return res;
}

void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline) {
    [pipeline->obj release];

    free(pipeline);
}

int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_with_params pipeline) {
    return pipeline.pipeline->obj.maxTotalThreadsPerThreadgroup;
}

//
// MTLLibrary collection (one library per op-source, compiled separately)
//

// Single source of truth for the per-kind metal libraries. The order here
// defines the enum values and every per-kind table below, so adding a library
// is a one-line change here (plus adding its source to CMakeLists.txt).
//   X(suffix, name): name is both the kernels/<name>.metal basename and the
//   ggml_metallib_<name>_{start,end} embed-symbol stem.
#define GGML_METAL_LIBS \
    X(FA_AUX,          fa_aux)         \
    X(FA_F16,          fa_f16)         \
    X(FA_F32,          fa_f32)         \
    X(FA_Q4_0,         fa_q4_0)        \
    X(FA_Q4_1,         fa_q4_1)        \
    X(FA_Q5_0,         fa_q5_0)        \
    X(FA_Q5_1,         fa_q5_1)        \
    X(FA_Q8_0,         fa_q8_0)        \
    X(FA_TENSOR,       fa_tensor)      \
    X(FA_VEC_F16,      fa_vec_f16)     \
    X(FA_VEC_F32,      fa_vec_f32)     \
    X(FA_VEC_Q4_0,     fa_vec_q4_0)    \
    X(FA_VEC_Q4_1,     fa_vec_q4_1)    \
    X(FA_VEC_Q5_0,     fa_vec_q5_0)    \
    X(FA_VEC_Q5_1,     fa_vec_q5_1)    \
    X(FA_VEC_Q8_0,     fa_vec_q8_0)    \
    X(MUL_MV,          mul_mv)         \
    X(MUL_MV_MMA,      mul_mv_mma)     \
    X(MUL_MM,          mul_mm)         \
    X(QUANTIZE,        quantize)       \
    X(SOFTMAX,         softmax)        \
    X(NORM,            norm)           \
    X(UNARY,           unary)          \
    X(BINBCAST,        binbcast)       \
    X(REDUCE,          reduce)         \
    X(TRI,             tri)            \
    X(SSM,             ssm)            \
    X(WKV,             wkv)            \
    X(GATED_DELTA_NET, gated_delta_net)\
    X(SOLVE_TRI,       solve_tri)      \
    X(ROPE,            rope)           \
    X(CONV,            conv)           \
    X(UPSCALE,         upscale)        \
    X(ARGSORT,         argsort)        \
    X(POOL,            pool)           \
    X(MISC,            misc)

enum ggml_metal_lib_kind {
#define X(e, s) GGML_METAL_LIB_##e,
    GGML_METAL_LIBS
#undef X
    GGML_METAL_LIB_COUNT,
};

static const char * const k_lib_names[GGML_METAL_LIB_COUNT] = {
#define X(e, s) [GGML_METAL_LIB_##e] = #s,
    GGML_METAL_LIBS
#undef X
};

struct ggml_metal_library {
    // Per-kind compiled libraries. When single_library is true, the whole library
    // (e.g. a pre-compiled default.metallib or a from-source build) lives at
    // objs[0] and the remaining slots are nil.
    id<MTLLibrary> objs[GGML_METAL_LIB_COUNT];
    bool single_library; // true: combined library at objs[0]; false: per-kind libs in objs[*]

    // Routing table: kernel function name -> objs[] index, populated from each
    // compiled library's -[MTLLibrary functionNames]. The actual compiled
    // libraries are the single source of truth for which library owns a kernel,
    // so adding kernels later requires no manual routing maintenance.
    // nil in single_library mode (everything resolves to objs[0]).
    NSMutableDictionary<NSString *, NSNumber *> * fn_to_lib;

    // kernels from a second metallib, resolved ahead of the combined library
    NSSet<NSString *> * override_fns;

    ggml_metal_device_t dev;
    ggml_metal_pipelines_t pipelines; // cache of compiled pipelines

    NSLock * lock;

    // for GGML_METAL_MEGA: the embedded source of each kind, and the compile options of the libraries
    const char * src_start[GGML_METAL_LIB_COUNT];
    const char * src_end  [GGML_METAL_LIB_COUNT];
    NSDictionary * prep;
};

// Build the fn_to_lib routing table by querying each compiled library's public
// function names. Call once after all per-kind libraries have been compiled.
static void ggml_metal_library_build_index(ggml_metal_library_t lib) {
    @autoreleasepool {
        NSMutableDictionary<NSString *, NSNumber *> * index = [[NSMutableDictionary alloc] init];
        for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
            for (NSString * fname in [lib->objs[kind] functionNames]) {
                index[fname] = @(kind);
            }
        }
        lib->fn_to_lib = index;
    }
}

// note: defined below, after struct ggml_metal_device
static void ggml_metal_device_disable_tensor(ggml_metal_device_t dev);

// the tensor API headers are exposed to the shader compiler only at Metal language version 4.0
static void ggml_metal_compile_options_set_lang(MTLCompileOptions * options, bool has_tensor) {
    if (!has_tensor) {
        return;
    }

    options.languageVersion = (MTLLanguageVersion) MTLLanguageVersion4_0_GGML;
}

// Parse a `#include "name"` line. Returns the quoted name in *include_name on
// success. Whitespace-tolerant; ignores `#include <...>` (system headers).
static bool ggml_metal_library_parse_quoted_include(NSString * line, NSString ** include_name) {
    NSScanner * scanner = [NSScanner scannerWithString:line];
    scanner.charactersToBeSkipped = [NSCharacterSet whitespaceCharacterSet];

    if (![scanner scanString:@"#" intoString:NULL] ||
        ![scanner scanString:@"include" intoString:NULL] ||
        ![scanner scanString:@"\"" intoString:NULL]) {
        return false;
    }

    NSString * name = nil;
    if (![scanner scanUpToString:@"\"" intoString:&name]) {
        return false;
    }

    if (include_name) {
        *include_name = name;
    }
    return true;
}

// Recursively inline `#include "name"` directives. System includes (<...>),
// `#if/#else/#endif`, and other preprocessor lines are passed through to the
// Metal compiler unchanged. `#pragma once` is dropped since `seen` already
// guards against double-inclusion.
static bool ggml_metal_library_flatten_file(NSMutableString * dst, NSString * path,
                                            NSArray<NSString *> * search_paths,
                                            NSMutableSet<NSString *> * seen, NSError ** error) {
    NSString * key = [path stringByStandardizingPath];
    if ([seen containsObject:key]) {
        return true;
    }
    [seen addObject:key];

    NSString * src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:error];
    if (!src) {
        return false;
    }

    NSFileManager * fm = [NSFileManager defaultManager];
    for (NSString * line in [src componentsSeparatedByString:@"\n"]) {
        NSString * trimmed = [line stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceCharacterSet]];
        if ([trimmed isEqualToString:@"#pragma once"]) {
            continue;
        }

        NSString * include_name = nil;
        if (ggml_metal_library_parse_quoted_include(line, &include_name)) {
            NSString * resolved = nil;
            for (NSString * dir in search_paths) {
                NSString * candidate = [dir stringByAppendingPathComponent:include_name];
                if ([fm isReadableFileAtPath:candidate]) {
                    resolved = candidate;
                    break;
                }
            }
            if (!resolved) {
                if (error) {
                    NSString * msg = [NSString stringWithFormat:@"could not resolve include \"%@\" from '%@'", include_name, path];
                    *error = [NSError errorWithDomain:@"ggml-metal-source-flatten" code:1
                                             userInfo:@{NSLocalizedDescriptionKey: msg}];
                }
                return false;
            }
            if (!ggml_metal_library_flatten_file(dst, resolved, search_paths, seen, error)) {
                return false;
            }
            continue;
        }

        [dst appendString:line];
        [dst appendString:@"\n"];
    }

    return true;
}

static NSString * ggml_metal_library_flatten_source(NSString * path_source, NSError ** error) {
    // Search paths cover both runtime layout (build/bin/kernels + build/bin)
    // and source-tree layout (ggml/src/ggml-metal/kernels + ggml/src/ggml-metal + ggml/src).
    NSString * path_kernels = [path_source stringByDeletingLastPathComponent];
    NSString * path_base    = [path_kernels stringByDeletingLastPathComponent];
    NSArray<NSString *> * search_paths = @[
        path_kernels,
        path_base,
        [path_base stringByDeletingLastPathComponent],
    ];

    NSMutableString * src = [[NSMutableString alloc] init];
    NSMutableSet<NSString *> * seen = [NSMutableSet set];

    if (!ggml_metal_library_flatten_file(src, path_source, search_paths, seen, error)) {
        [src release];
        return nil;
    }
    return src;
}

// Compile all per-kind libraries in parallel. `source_for_kind` returns the MSL
// source for a kind (the helper takes ownership and releases it), or nil with
// *err set on failure. On success the objs[] slots are populated and the routing
// index is built; on any failure every error is logged and false is returned
// (the caller is responsible for freeing `res`).
static bool ggml_metal_library_compile_all(
        ggml_metal_library_t res,
        id<MTLDevice> device,
        NSDictionary * prep,
        NSString * (^source_for_kind)(int kind, NSError ** err),
        const char * origin) {
    const int64_t t_start = ggml_time_us();

    int64_t  * t_per_lib   = calloc(GGML_METAL_LIB_COUNT, sizeof(int64_t));
    NSError ** err_per_lib = calloc(GGML_METAL_LIB_COUNT, sizeof(NSError *));
    __block atomic_bool any_failure = false;

    dispatch_group_t group = dispatch_group_create();
    dispatch_queue_t queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);

    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        dispatch_group_async(group, queue, ^{

            const int64_t t0 = ggml_time_us();

            NSError * error = nil;

            NSString * src = source_for_kind(kind, &error);
            if (!src) {
                err_per_lib[kind] = [error retain];
                atomic_store(&any_failure, true);
                return;
            }

            id<MTLLibrary> lib = nil;

            @autoreleasepool {
                MTLCompileOptions * options = [MTLCompileOptions new];
                options.preprocessorMacros = prep;
                ggml_metal_compile_options_set_lang(options, ggml_metal_device_get_props(res->dev)->has_tensor);

                lib = [device newLibraryWithSource:src options:options error:&error];

                [options release];

                // retain the error before the autorelease pool drains it
                if (!lib) {
                    err_per_lib[kind] = [error retain];
                }
            }

            [src release];

            t_per_lib[kind] = ggml_time_us() - t0;

            if (!lib) {
                atomic_store(&any_failure, true);
                return;
            }

            res->objs[kind] = lib;
        });
    }
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    dispatch_release(group);

    const bool ok = !atomic_load(&any_failure);

    if (ok) {
        const int64_t t_total = ggml_time_us() - t_start;
        int64_t t_max = 0;
        for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
            GGML_LOG_DEBUG("%s: compiled '%s' library in %.3f sec\n",
                           __func__, k_lib_names[kind], t_per_lib[kind] / 1e6);
            if (t_per_lib[kind] > t_max) t_max = t_per_lib[kind];
        }
        GGML_LOG_INFO("%s: loaded %d libraries from %s in %.3f sec (max single = %.3f sec)\n",
                      __func__, GGML_METAL_LIB_COUNT, origin, t_total / 1e6, t_max / 1e6);

        ggml_metal_library_build_index(res);
    } else {
        for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
            if (err_per_lib[kind]) {
                GGML_LOG_ERROR("%s: failed to build '%s' library: %s\n", __func__,
                               k_lib_names[kind], [[err_per_lib[kind] description] UTF8String]);
                [err_per_lib[kind] release];
            }
        }
    }

    free(err_per_lib);
    free(t_per_lib);

    return ok;
}

// look for <name>.metallib as a bundle resource, then next to the running binary
static NSString * ggml_metal_find_metallib(NSBundle * bundle, NSString * name) {
    NSError * error = nil;

    NSString * path_lib = [bundle pathForResource:name ofType:@"metallib"];
    if (path_lib == nil) {
        // Try to find the resource in the directory where the current binary located.
        NSString * bin_cur = [[NSProcessInfo processInfo] arguments][0];
        NSString * bin_dir = [bin_cur stringByDeletingLastPathComponent];

        NSString * path_lib_default = [NSString pathWithComponents:@[bin_dir, [name stringByAppendingPathExtension:@"metallib"]]];
        if ([[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
            GGML_LOG_INFO("%s: found '%s'\n", __func__, [path_lib_default UTF8String]);

            NSDictionary * atts = [[NSFileManager defaultManager] attributesOfItemAtPath:path_lib_default error:&error];
            if (atts && atts[NSFileType] == NSFileTypeSymbolicLink) {
                // Optionally, if this is a symlink, try to resolve it.
                path_lib_default = [[NSFileManager defaultManager] destinationOfSymbolicLinkAtPath:path_lib_default error:&error];
                if (path_lib_default && [path_lib_default length] > 0 && ![[path_lib_default substringToIndex:1] isEqualToString:@"/"]) {
                    // It is a relative path, adding the binary directory as directory prefix.
                    path_lib_default = [NSString pathWithComponents:@[bin_dir, path_lib_default]];
                }
                if (!path_lib_default || ![[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                    // Link to the resource could not be resolved.
                    path_lib_default = nil;
                } else {
                    GGML_LOG_INFO("%s: symlink resolved '%s'\n", __func__, [path_lib_default UTF8String]);
                }
            }
        } else {
            // The resource couldn't be found in the binary's directory.
            path_lib_default = nil;
        }

        path_lib = path_lib_default;
    }

    return path_lib;
}

ggml_metal_library_t ggml_metal_library_init(ggml_metal_device_t dev) {
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));
    res->dev       = dev;
    res->pipelines = ggml_metal_pipelines_init();
    res->lock      = [NSLock new];

    // shared MTLCompileOptions preprocessor macros (matches the build-time defines)
    NSMutableDictionary * prep = [NSMutableDictionary dictionary];
    if (ggml_metal_device_get_props(dev)->has_bfloat) {
        [prep setObject:@"1" forKey:@"GGML_METAL_HAS_BF16"];
    }
    if (ggml_metal_device_get_props(dev)->has_tensor) {
        [prep setObject:@"1" forKey:@"GGML_METAL_HAS_TENSOR"];
    }
#if GGML_METAL_EMBED_LIBRARY
    [prep setObject:@"1" forKey:@"GGML_METAL_EMBED_LIBRARY"];
#endif

#if GGML_METAL_EMBED_LIBRARY
    GGML_LOG_INFO("%s: using embedded metal library\n", __func__);

    // start/end symbols emitted by CMake (see CMakeLists.txt), one pair per kind
#define X(e, s) extern const char ggml_metallib_##s##_start[]; extern const char ggml_metallib_##s##_end[];
    GGML_METAL_LIBS
#undef X

    static const char * const lib_start[GGML_METAL_LIB_COUNT] = {
#define X(e, s) [GGML_METAL_LIB_##e] = ggml_metallib_##s##_start,
    GGML_METAL_LIBS
#undef X
    };
    static const char * const lib_end[GGML_METAL_LIB_COUNT] = {
#define X(e, s) [GGML_METAL_LIB_##e] = ggml_metallib_##s##_end,
    GGML_METAL_LIBS
#undef X
    };

    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        res->src_start[kind] = lib_start[kind];
        res->src_end  [kind] = lib_end[kind];
    }
    res->prep = [prep retain];

    const bool ok = ggml_metal_library_compile_all(res, device, prep,
        ^NSString * (int kind, NSError ** err) {
            (void) err;
            return [[NSString alloc] initWithBytes:lib_start[kind]
                                            length:(lib_end[kind] - lib_start[kind])
                                          encoding:NSUTF8StringEncoding];
        }, "embedded data");

    if (!ok) {
        ggml_metal_library_free(res);
        return NULL;
    }

    return res;
#else
#ifdef SWIFT_PACKAGE
    NSBundle * bundle = SWIFTPM_MODULE_BUNDLE;
#else
    NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
#endif

    const int64_t t_start = ggml_time_us();

    NSError * error = nil;
    NSString * path_lib = ggml_metal_find_metallib(bundle, @"default");

    if (path_lib != nil) {
        // pre-compiled library found: a single combined default.metallib
        NSURL * libURL = [NSURL fileURLWithPath:path_lib];
        GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

        res->objs[0]        = [device newLibraryWithURL:libURL error:&error];
        res->single_library = true;
        if (!res->objs[0]) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            ggml_metal_library_free(res);
            return NULL;
        }

        // the tensor API kernels are built into a separate metallib
        if (ggml_metal_device_get_props(dev)->has_tensor) {
            NSString * path_mm = ggml_metal_find_metallib(bundle, @"ggml-tensor");

            id<MTLLibrary> lib_mm = nil;
            if (path_mm != nil) {
                lib_mm = [device newLibraryWithURL:[NSURL fileURLWithPath:path_mm] error:&error];
                if (!lib_mm && error) {
                    GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
                }
            }

            if (lib_mm) {
                GGML_LOG_INFO("%s: loaded '%s'\n", __func__, [path_mm UTF8String]);

                res->objs[GGML_METAL_LIB_MUL_MM] = [lib_mm retain];
                res->override_fns                = [[NSSet setWithArray:[lib_mm functionNames]] retain];
            } else {
                GGML_LOG_INFO("%s: ggml-tensor.metallib not found - disabling the tensor API\n", __func__);

                ggml_metal_device_disable_tensor(dev);
            }
        }

        GGML_LOG_INFO("%s: loaded in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
        return res;
    }

    // no pre-compiled metallib: fall back to compiling each kernel source separately
    GGML_LOG_INFO("%s: default.metallib not found, loading kernel sources\n", __func__);

    NSString * path_resource = [[NSProcessInfo processInfo].environment objectForKey:@"GGML_METAL_PATH_RESOURCES"];
    if (path_resource) {
        GGML_LOG_INFO("%s: GGML_METAL_PATH_RESOURCES = %s\n", __func__, [path_resource UTF8String]);
    }

    // resolve each kind's source path up front (file lookup/logging stays on the calling thread)
    NSString ** path_per_kind = calloc(GGML_METAL_LIB_COUNT, sizeof(NSString *));
    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        NSString * rel = [NSString stringWithFormat:@"kernels/%s.metal", k_lib_names[kind]];

        NSString * path_source = nil;
        if (path_resource) {
            path_source = [path_resource stringByAppendingPathComponent:rel];
        } else {
            NSString * stem = [NSString stringWithFormat:@"kernels/%s", k_lib_names[kind]];
            path_source = [bundle pathForResource:stem ofType:@"metal"];
        }

        if (path_source == nil || ![[NSFileManager defaultManager] isReadableFileAtPath:path_source]) {
            GGML_LOG_WARN("%s: could not locate %s in bundle, falling back to cwd\n", __func__, [rel UTF8String]);
            path_source = rel;
        }

        GGML_LOG_DEBUG("%s: loading '%s'\n", __func__, [path_source UTF8String]);

        path_per_kind[kind] = [path_source retain];
    }

    const bool ok = ggml_metal_library_compile_all(res, device, prep,
        ^NSString * (int kind, NSError ** err) {
            return ggml_metal_library_flatten_source(path_per_kind[kind], err);
        }, "source");

    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        [path_per_kind[kind] release];
    }
    free(path_per_kind);

    if (!ok) {
        ggml_metal_library_free(res);
        return NULL;
    }

    return res;
#endif
}

ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, const char * source, bool verbose) {
    if (source == NULL) {
        GGML_LOG_ERROR("%s: source is NULL\n", __func__);
        return NULL;
    }

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLLibrary> library = nil;
    NSError * error = nil;

    const int64_t t_start = ggml_time_us();

    NSString * src = [[NSString alloc] initWithBytes:source
                                              length:strlen(source)
                                            encoding:NSUTF8StringEncoding];
    if (!src) {
        GGML_LOG_ERROR("%s: failed to create NSString from source\n", __func__);
        return NULL;
    }

    @autoreleasepool {
        NSMutableDictionary * prep = [NSMutableDictionary dictionary];

        MTLCompileOptions * options = [MTLCompileOptions new];
        options.preprocessorMacros = prep;
        ggml_metal_compile_options_set_lang(options, ggml_metal_device_get_props(dev)->has_tensor);

        library = [device newLibraryWithSource:src options:options error:&error];
        if (error) {
            if (verbose) {
                GGML_LOG_ERROR("%s: error compiling source: %s\n", __func__, [[error description] UTF8String]);
            } else {
                GGML_LOG_ERROR("%s: error compiling source\n", __func__);
            }
            library = nil;
        }

        [options release];
    }

    [src release];

    if (!library) {
        if (verbose) {
            GGML_LOG_ERROR("%s: failed to create Metal library from source\n", __func__);
        }

        return NULL;
    }

    if (verbose) {
        GGML_LOG_INFO("%s: compiled in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));
    if (!res) {
        GGML_LOG_ERROR("%s: calloc failed\n", __func__);
        return NULL;
    }

    res->objs[0]        = library;
    res->single_library = true;
    res->dev            = dev;
    res->pipelines      = ggml_metal_pipelines_init();
    res->lock           = [NSLock new];

    return res;
}

// bumped when a library is freed: see ggml_metal_library_get_pipeline
static _Atomic uint64_t g_pipeline_cache_gen = 1;

void ggml_metal_library_free(ggml_metal_library_t lib) {
    atomic_fetch_add(&g_pipeline_cache_gen, 1);

    if (!lib) {
        return;
    }

    ggml_metal_mega_forget_library(lib);

    for (int kind = 0; kind < GGML_METAL_LIB_COUNT; ++kind) {
        if (lib->objs[kind]) {
            [lib->objs[kind] release];
        }
    }

    if (lib->fn_to_lib) {
        [lib->fn_to_lib release];
    }

    if (lib->override_fns) {
        [lib->override_fns release];
    }

    ggml_metal_pipelines_free(lib->pipelines);

    [lib->lock release];

    free(lib);
}

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib) {
    return lib->dev;
}

// every kernel that is encoded looks its pipeline up by name. the library lock made that the point where the encoder
// threads of a graph wait for one another, so each thread keeps the pipelines it has found. a pipeline stays valid
// while its library lives; the generation drops the caches when a library is freed
#define GGML_METAL_PIPELINE_CACHE_SIZE 512
#define GGML_METAL_PIPELINE_CACHE_NAME 120

struct ggml_metal_pipeline_cache_entry {
    uint64_t hash;
    uint64_t gen;
    ggml_metal_library_t lib;
    ggml_metal_pipeline_t pipeline;
    char name[GGML_METAL_PIPELINE_CACHE_NAME];
};

static __thread struct ggml_metal_pipeline_cache_entry t_pipeline_cache[GGML_METAL_PIPELINE_CACHE_SIZE];

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline(ggml_metal_library_t lib, const char * name) {
    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    uint64_t hash = 1469598103934665603ull;
    size_t len = 0;
    for (; name[len] != 0; ++len) {
        hash = (hash ^ (uint8_t) name[len])*1099511628211ull;
    }

    const uint64_t gen = atomic_load(&g_pipeline_cache_gen);

    struct ggml_metal_pipeline_cache_entry * entry = &t_pipeline_cache[hash % GGML_METAL_PIPELINE_CACHE_SIZE];

    if (entry->hash == hash && entry->gen == gen && entry->lib == lib && strcmp(entry->name, name) == 0) {
        res.pipeline = entry->pipeline;
        return res;
    }

    [lib->lock lock];

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);

    [lib->lock unlock];

    if (res.pipeline != nil && len < GGML_METAL_PIPELINE_CACHE_NAME) {
        entry->hash     = hash;
        entry->gen      = gen;
        entry->lib      = lib;
        entry->pipeline = res.pipeline;
        memcpy(entry->name, name, len + 1);
    }

    return res;
}

struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv) {
    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    [lib->lock lock];

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);
    if (res.pipeline) {
        [lib->lock unlock];

        return res;
    }

    @autoreleasepool {
        NSError * error = nil;

        NSString * base_func = [NSString stringWithUTF8String:base];

        GGML_LOG_DEBUG("%s: compiling pipeline: base = '%s', name = '%s'\n", __func__, base, name);

        // route to the library that actually defines this kernel; fn_to_lib is
        // built from -[MTLLibrary functionNames] so it's always in sync
        int lib_idx = 0;
        if (lib->override_fns && [lib->override_fns containsObject:base_func]) {
            lib_idx = GGML_METAL_LIB_MUL_MM;
        } else if (!lib->single_library) {
            NSNumber * idx = lib->fn_to_lib[base_func];
            if (!idx) {
                [lib->lock unlock];

                GGML_LOG_ERROR("%s: kernel not found in any metal library: base = '%s', name = '%s'\n", __func__, base, name);

                return res;
            }
            lib_idx = [idx intValue];
        }

        id<MTLLibrary> mtl_lib = lib->objs[lib_idx];

        id<MTLFunction> mtl_function;
        if (!cv) {
            mtl_function = [mtl_lib newFunctionWithName:base_func];
        } else {
            mtl_function = [mtl_lib newFunctionWithName:base_func constantValues:cv->obj error:&error];
        }
        if (!mtl_function) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to compile pipeline: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        id<MTLDevice> device = ggml_metal_device_get_obj(lib->dev);
        // usable from an indirect command buffer too: see ggml_metal_recording
        MTLComputePipelineDescriptor * desc = [[MTLComputePipelineDescriptor alloc] init];
        desc.computeFunction = mtl_function;
        desc.supportIndirectCommandBuffers = YES;
        desc.label = [NSString stringWithUTF8String:name];

        id<MTLComputePipelineState> obj = [device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&error];

        [desc release];

        [mtl_function release];

        if (!obj) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to create pipeline state: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, name,
                (void *) obj,
                (int)    obj.maxTotalThreadsPerThreadgroup,
                (int)    obj.threadExecutionWidth);

        if (obj.maxTotalThreadsPerThreadgroup == 0 || obj.threadExecutionWidth == 0) {
            [obj release];

            [lib->lock unlock];

            GGML_LOG_ERROR("%s: incompatible pipeline %s\n", __func__, name);

            return res;
        }

        res.pipeline = ggml_metal_pipeline_init();
        res.pipeline->obj = obj;

        snprintf(res.pipeline->base, sizeof(res.pipeline->base), "%s", base);
        res.pipeline->kind = lib->single_library ? -1 : lib_idx;
        if (lib->override_fns && [lib->override_fns containsObject:base_func]) {
            res.pipeline->kind = -1;
        }
        if (cv) {
            res.pipeline->n_fc = cv->n;
            for (int i = 0; i < cv->n; ++i) {
                res.pipeline->fc_idx [i] = cv->idx[i];
                res.pipeline->fc_type[i] = cv->type[i];
                res.pipeline->fc_val [i] = cv->val[i];
            }
        }
        ggml_metal_mega_set_library(lib);

        ggml_metal_pipelines_add(lib->pipelines, name, res.pipeline);
    }

    [lib->lock unlock];

    return res;
}

//
// MTLComputeCommandEncoder wrapper
//

// a graph that is computed again with the same nodes encodes the same kernels with the same buffers and arguments:
// only the contents of the buffers changed. the kernels are recorded once in an indirect command buffer and executed
// from it after that with one call, instead of being encoded again for every token
#define GGML_METAL_RECORDING_BINDS     31
#define GGML_METAL_RECORDING_RESOURCES 64

#define GGML_METAL_RECORDING_STEP_RES 48

struct ggml_metal_recording {
    id<MTLIndirectCommandBuffer> icb;
    id<MTLBuffer> args; // the arguments that were set_bytes
    size_t args_size;
    size_t args_offs;
    int n;
    int n_max;
    bool failed;
    int * starts; // the first command of each serial step
    int n_starts;
    id<MTLBuffer> __unsafe_unretained * step_res; // the buffers of each step, GGML_METAL_RECORDING_STEP_RES a step
    int * step_n_res;
    id<MTLBuffer> resources[GGML_METAL_RECORDING_RESOURCES];
    int n_resources;

    // an indirect command does not keep a large buffer offset (measured: an offset of 4 GiB reads at 0, and a
    // graph bound at 2 to 4 GiB computes garbage), and buffers can be larger than that. a binding beyond 2 GiB goes
    // through a window: a second buffer over the same memory that starts every GiB, so that the offset in it is
    // below 1 GiB. windows cost about 0.8 ms an execution (measured with 6 of them), so buffers are made small
    // enough not to need any: see ggml_metal_buffer_view_max
    struct {
        id<MTLBuffer> src;
        uint64_t      start;
        id<MTLBuffer> buf;
    } windows[GGML_METAL_RECORDING_RESOURCES];
    int n_windows;
};

struct ggml_metal_encoder {
    id<MTLComputeCommandEncoder> obj;
    id<MTLCommandBuffer> cmd; // for GGML_METAL_BARRIER=encoder
    bool concurrent;

    // recording: no Metal encoder, the state of one is kept here and written to a command at each dispatch
    ggml_metal_recording_t rec;
    id<MTLComputePipelineState> pso;
    struct {
        id<MTLBuffer> buf;
        size_t offs;
    } bind[GGML_METAL_RECORDING_BINDS];
    int    n_bind;
    size_t tg_size[2];
    bool   barrier;

    id<MTLComputePipelineState> obj_pso; // the pipeline last set (for GGML_METAL_KERNEL_LIST)
    int    tl_slot; // GGML_METAL_TIMELINE: the sample pair of this encoder, -1 if none

    struct ggml_metal_mega_rec * mega; // GGML_METAL_MEGA: dispatches are recorded, encoded at the end
    uint64_t kl_b[16]; // GGML_METAL_KERNEL_LIST: the bindings (address, 1 for bytes)
};

// GGML_METAL_TIMELINE=file[,skip[,count]]: GPU timestamps at the start and the end of every compute encoder (with an
// encoder a serial step, that is every step), for the encoders after the first `skip` ones, up to `count`. written to
// file at exit: slot, start ns, end ns, dispatches, label of the first op. CAUTION, measured on an M5 Ultra: sampled
// tokens run ~25% slower (10.6 instead of 8.6 ms, even with GGML_METAL_TIMELINE_EVERY=2) and show ~300 us holes every
// 80 encoders that are not there unsampled, so it gives the order and relative size of steps, not the idle time
static int g_tl_on = -1;
static id<MTLCounterSampleBuffer> g_tl_buf = nil;
static id<MTLDevice> g_tl_dev = nil;
static atomic_int g_tl_seen = 0;
static int g_tl_skip = 0, g_tl_cap = 0;
static char g_tl_path[512];
static char (*g_tl_lab)[64] = NULL;
static int * g_tl_ndisp = NULL;
static MTLTimestamp g_tl_cpu0, g_tl_gpu0;
static __thread char t_tl_label[64];

void ggml_metal_timeline_label(const char * s) {
    if (g_tl_on == 1) {
        snprintf(t_tl_label, sizeof(t_tl_label), "%s", s);
    }
}

static void ggml_metal_timeline_dump(void) {
    if (g_tl_buf == nil) {
        return;
    }
    int n = (atomic_load(&g_tl_seen) - g_tl_skip + (getenv("GGML_METAL_TIMELINE_EVERY") ? atoi(getenv("GGML_METAL_TIMELINE_EVERY")) : 1) - 1)/(getenv("GGML_METAL_TIMELINE_EVERY") ? atoi(getenv("GGML_METAL_TIMELINE_EVERY")) : 1);
    if (n > g_tl_cap) n = g_tl_cap;
    if (n <= 0) return;
    MTLTimestamp cpu1, gpu1;
    [g_tl_dev sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
    const double ns_per_tick = gpu1 > g_tl_gpu0 ? (double)(cpu1 - g_tl_cpu0)/(double)(gpu1 - g_tl_gpu0) : 1.0;
    NSData * d = [g_tl_buf resolveCounterRange:NSMakeRange(0, 2*n)];
    if (d == nil) {
        fprintf(stderr, "ggml_metal_timeline: resolve failed\n");
        return;
    }
    const MTLCounterResultTimestamp * ts = (const MTLCounterResultTimestamp *) [d bytes];
    FILE * f = fopen(g_tl_path, "w");
    if (!f) return;
    fprintf(f, "# ns_per_tick %.6f\n", ns_per_tick);
    for (int i = 0; i < n; ++i) {
        fprintf(f, "%d %llu %llu %d %s\n", i, (unsigned long long) ts[2*i].timestamp, (unsigned long long) ts[2*i + 1].timestamp,
                g_tl_ndisp[i], g_tl_lab[i][0] ? g_tl_lab[i] : "-");
    }
    fclose(f);
    fprintf(stderr, "ggml_metal_timeline: %d encoders written to %s\n", n, g_tl_path);
}

static bool ggml_metal_timeline_init(id<MTLDevice> dev) {
    if (g_tl_on >= 0) {
        return g_tl_on == 1;
    }
    g_tl_on = 0;
    const char * v = getenv("GGML_METAL_TIMELINE");
    if (!v) return false;
    char tmp[512];
    snprintf(tmp, sizeof(tmp), "%s", v);
    char * c1 = strchr(tmp, ',');
    int cap = 16384;
    if (c1) {
        *c1 = 0;
        g_tl_skip = atoi(c1 + 1);
        char * c2 = strchr(c1 + 1, ',');
        if (c2) cap = atoi(c2 + 1);
    }
    snprintf(g_tl_path, sizeof(g_tl_path), "%s", tmp);
    if (![dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        fprintf(stderr, "ggml_metal_timeline: no stage-boundary sampling on this device\n");
        return false;
    }
    id<MTLCounterSet> set = nil;
    for (id<MTLCounterSet> cs in [dev counterSets]) {
        if ([[cs name] isEqualToString:MTLCommonCounterSetTimestamp]) set = cs;
    }
    if (set == nil) {
        fprintf(stderr, "ggml_metal_timeline: no timestamp counter set\n");
        return false;
    }
    for (; cap >= 256; cap /= 2) {
        MTLCounterSampleBufferDescriptor * desc = [[MTLCounterSampleBufferDescriptor alloc] init];
        desc.counterSet  = set;
        desc.storageMode = MTLStorageModeShared;
        desc.sampleCount = 2*cap;
        NSError * err = nil;
        g_tl_buf = [dev newCounterSampleBufferWithDescriptor:desc error:&err];
        [desc release];
        if (g_tl_buf != nil) break;
    }
    if (g_tl_buf == nil) {
        fprintf(stderr, "ggml_metal_timeline: no sample buffer\n");
        return false;
    }
    g_tl_cap   = cap;
    g_tl_dev   = [dev retain];
    g_tl_lab   = calloc(cap, sizeof(*g_tl_lab));
    g_tl_ndisp = calloc(cap, sizeof(int));
    [dev sampleTimestamps:&g_tl_cpu0 gpuTimestamp:&g_tl_gpu0];
    atexit(ggml_metal_timeline_dump);
    fprintf(stderr, "ggml_metal_timeline: %d encoders after the first %d, to %s\n", cap, g_tl_skip, g_tl_path);
    g_tl_on = 1;
    return true;
}

// a compute encoder; with GGML_METAL_TIMELINE it samples the GPU clock at its start and end
static id<MTLComputeCommandEncoder> ggml_metal_new_compute_encoder(id<MTLCommandBuffer> cmd, bool concurrent, int * slot) {
    *slot = -1;
    if (ggml_metal_timeline_init([cmd device])) {
        static int every = -1;
        if (every < 0) {
            every = getenv("GGML_METAL_TIMELINE_EVERY") ? atoi(getenv("GGML_METAL_TIMELINE_EVERY")) : 1;
            every = every < 1 ? 1 : every;
        }
        const int seen = atomic_fetch_add(&g_tl_seen, 1);
        const int k = (seen - g_tl_skip)/every;
        if (seen >= g_tl_skip && (seen - g_tl_skip) % every == 0 && k < g_tl_cap) {
            MTLComputePassDescriptor * d = [MTLComputePassDescriptor computePassDescriptor];
            d.dispatchType = concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;
            d.sampleBufferAttachments[0].sampleBuffer              = g_tl_buf;
            d.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2*k;
            d.sampleBufferAttachments[0].endOfEncoderSampleIndex   = 2*k + 1;
            *slot = k;
            g_tl_lab[k][0] = 0;
            g_tl_ndisp[k]  = 0;
            return [cmd computeCommandEncoderWithDescriptor:d];
        }
    }
    return concurrent ? [cmd computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent] : [cmd computeCommandEncoder];
}

// the next encoder made on this thread records to this
static __thread ggml_metal_recording_t t_recording_next = NULL;

ggml_metal_recording_t ggml_metal_recording_init(ggml_metal_device_t dev, int n_max) {
    ggml_metal_recording_t res = calloc(1, sizeof(struct ggml_metal_recording));

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    MTLIndirectCommandBufferDescriptor * desc = [[MTLIndirectCommandBufferDescriptor alloc] init];
    desc.commandTypes = MTLIndirectCommandTypeConcurrentDispatch;
    desc.inheritBuffers = NO;
    desc.inheritPipelineState = NO;
    desc.maxKernelBufferBindCount = GGML_METAL_RECORDING_BINDS;
    desc.maxKernelThreadgroupMemoryBindCount = 2;

    res->icb = [device newIndirectCommandBufferWithDescriptor:desc maxCommandCount:n_max options:0];

    [desc release];

    res->n_max     = n_max;
    res->starts    = calloc(n_max + 1, sizeof(int));
    res->step_res   = (id<MTLBuffer> __unsafe_unretained *) calloc((size_t) (n_max + 1)*GGML_METAL_RECORDING_STEP_RES, sizeof(void *));
    res->step_n_res = calloc(n_max + 1, sizeof(int));
    res->args_size = (size_t) n_max*256;
    res->args      = [device newBufferWithLength:res->args_size options:MTLResourceStorageModeShared];
    res->failed    = res->icb == nil || res->args == nil;

    return res;
}

void ggml_metal_recording_free(ggml_metal_recording_t rec) {
    if (!rec) {
        return;
    }
    [rec->icb release];
    [rec->args release];
    free(rec->starts);
    free(rec->step_res);
    free(rec->step_n_res);
    for (int i = 0; i < rec->n_windows; ++i) {
        [rec->windows[i].buf release];
    }
    free(rec);
}

bool ggml_metal_recording_ok(ggml_metal_recording_t rec) {
    return rec && !rec->failed && rec->n > 0;
}

int ggml_metal_recording_n(ggml_metal_recording_t rec) {
    return rec ? rec->n : 0;
}

void ggml_metal_recording_begin(ggml_metal_recording_t rec) {
    t_recording_next = rec;
}

static void ggml_metal_recording_use(ggml_metal_recording_t rec, id<MTLBuffer> buf) {
    for (int i = 0; i < rec->n_resources; ++i) {
        if (rec->resources[i] == buf) {
            return;
        }
    }
    if (rec->n_resources == GGML_METAL_RECORDING_RESOURCES) {
        rec->failed = true;
        return;
    }
    rec->resources[rec->n_resources++] = buf;
}

#define GGML_METAL_RECORDING_WINDOW_STEP ((uint64_t) 1 << 30)

// the buffer and offset to bind in an indirect command for an offset in buf; false if there is no way
static bool ggml_metal_recording_window(ggml_metal_recording_t rec, id<MTLBuffer> buf, size_t offs, id<MTLBuffer> * buf_out, size_t * offs_out) {
    if (offs < ((uint64_t) 1 << 31)) {
        *buf_out  = buf;
        *offs_out = offs;
        return true;
    }

    const uint64_t start = offs/GGML_METAL_RECORDING_WINDOW_STEP*GGML_METAL_RECORDING_WINDOW_STEP;

    for (int i = 0; i < rec->n_windows; ++i) {
        if (rec->windows[i].src == buf && rec->windows[i].start == start) {
            *buf_out  = rec->windows[i].buf;
            *offs_out = offs - start;
            return true;
        }
    }

    // only memory the host sees can be given a second buffer
    char * contents = (char *) [buf contents];
    if (contents == NULL || rec->n_windows == GGML_METAL_RECORDING_RESOURCES) {
        return false;
    }

    const uint64_t page = (uint64_t) sysconf(_SC_PAGESIZE);
    uint64_t len = [buf length] - start;
    if (len > 2*GGML_METAL_RECORDING_WINDOW_STEP - page) {
        len = 2*GGML_METAL_RECORDING_WINDOW_STEP - page;
    }
    len = (len + page - 1)/page*page;

    id<MTLBuffer> win = [[buf device] newBufferWithBytesNoCopy:contents + start length:len options:MTLResourceStorageModeShared deallocator:nil];
    if (win == nil) {
        return false;
    }

    rec->windows[rec->n_windows].src   = buf;
    rec->windows[rec->n_windows].start = start;
    rec->windows[rec->n_windows].buf   = win;
    rec->n_windows++;
    if (getenv("GGML_METAL_PROFILE")) {
        fprintf(stderr, "ggml_metal_recording: window %d: buffer of %.2f GiB at %.2f GiB" "\n", rec->n_windows, [buf length]/1073741824.0, start/1073741824.0);
    }

    *buf_out  = win;
    *offs_out = offs - start;
    return true;
}

// execute the recorded kernels in a command buffer
void ggml_metal_recording_replay(ggml_metal_recording_t rec, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    // GGML_METAL_REPLAY_STEPS=encoder: an encoder a serial step (experiment)
    static int steps = -1;
    if (steps < 0) {
        const char * v = getenv("GGML_METAL_REPLAY_STEPS");
        steps = v && strcmp(v, "encoder") == 0 ? 1 : 0;
    }
    if (steps == 1) {
        for (int i = 0; i < rec->n_starts; ++i) {
            const int beg = rec->starts[i];
            const int end = i + 1 < rec->n_starts ? rec->starts[i + 1] : rec->n;
            id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            [enc useResources:rec->step_res + (size_t) i*GGML_METAL_RECORDING_STEP_RES count:rec->step_n_res[i] usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            [enc executeCommandsInBuffer:rec->icb withRange:NSMakeRange(beg, end - beg)];
            [enc endEncoding];
        }
        return;
    }

    id<MTLComputeCommandEncoder> enc = [cmd_buf computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];

    // every execution has to declare the buffers: with the buffers only kept resident (a residency set on the
    // command buffer) the kernels computed garbage
    [enc useResources:rec->resources count:rec->n_resources usage:MTLResourceUsageRead | MTLResourceUsageWrite];

    [enc executeCommandsInBuffer:rec->icb withRange:NSMakeRange(0, rec->n)];
    [enc endEncoding];
}

//
// GGML_METAL_MEGA (see ggml-metal-mega.h): the encoder records the dispatches and the step boundaries of a graph; at
// the end, every run of steps whose kernels the persistent kernel can run becomes one dispatch of it, and the other
// steps are encoded as usual (an encoder a step)
//

#define GGML_METAL_MEGA_RING 8
static int g_mega_T = 256; // threads a threadgroup of the persistent kernel (GGML_METAL_MEGA_TPT)
#define GGML_METAL_MEGA_THREADS g_mega_T

struct ggml_metal_mega_bind {
    id<MTLBuffer> buf;  // nil: bytes (or unbound)
    size_t offs;
    int    bytes_off;   // >= 0: set_bytes data at this offset of rec->bytes
    int    bytes_size;
};

struct ggml_metal_mega_disp {
    ggml_metal_pipeline_t pl;
    struct ggml_metal_mega_bind b[GGML_METAL_RECORDING_BINDS];
    int    nb;
    size_t tg[2];
    int    grid[3];
    int    tptg[3];
    int    step;
    int    r0, r1;  // its memory ranges in rec->rng (GGML_METAL_MEGA_NOCOH: the stale-line check)
};

struct ggml_metal_mega_rec {
    struct ggml_metal_mega_disp * d;
    int n, cap;
    uint8_t * bytes;
    size_t n_bytes, cap_bytes;
    int step;           // current step
    bool step_used;     // the current step has a dispatch
    bool failed;        // something it cannot record: encode as usual
    struct ggml_metal_mega_disp cur;
    struct { uint64_t lo, hi; int w; } * rng; // memory ranges of the dispatches (w: written)
    int n_rng, cap_rng;
};

static struct {
    int on;             // -1: not decided yet
    int R;              // threadgroups of the persistent kernel
    int min_steps;      // shortest run of steps worth a persistent dispatch
    int gen;            // generation of the compiled megakernel
    id<MTLComputePipelineState> pso;
    id<MTLComputePipelineState> pso_one; // kernel_mega_one (GGML_METAL_MEGA_ONESHOT)
    NSMutableArray * old_psos; // kept: command buffers do not retain them
    ggml_metal_library_t lib;
    ggml_metal_pipeline_t * pipes;
    int n_pipes, cap_pipes;
    struct {
        id<MTLBuffer> buf;
        id<MTLCommandBuffer> cmd;
        size_t n_ctr;
        size_t ctr[1024]; // offsets of the barrier counters written last time
    } ring[GGML_METAL_MEGA_RING];
    int ring_i;
    long long n_steps_mega, n_steps_plain, n_disp; // statistics
} g_mega = { .on = -1, .gen = 1 };

static NSLock * g_mega_lock = nil;

// the library whose pipelines the megakernel is built from: the latest that compiled one. when it is freed, so are its
// pipelines: ggml_metal_mega_forget_library drops them
static void ggml_metal_mega_set_library(struct ggml_metal_library * lib) {
    if (g_mega.lib != lib) {
        g_mega.lib     = lib;
        g_mega.n_pipes = 0;
    }
}

static void ggml_metal_mega_forget_library(struct ggml_metal_library * lib) {
    if (g_mega.lib == lib) {
        g_mega.lib     = NULL;
        g_mega.n_pipes = 0;
    }
}

static __thread bool t_mega_next = false;
static int g_mega_pt = 0; // GGML_METAL_MEGA_PASSTHRU: record, but encode as usual too (no replay)

static bool ggml_metal_mega_enabled(void) {
    if (g_mega.on < 0) {
        g_mega.on = getenv("GGML_METAL_MEGA") != NULL && atoi(getenv("GGML_METAL_MEGA")) != 0 && !getenv("GGML_METAL_MEGA_IGNORE");
        g_mega.R  = getenv("GGML_METAL_MEGA_TGS") ? atoi(getenv("GGML_METAL_MEGA_TGS")) : 64;
        g_mega.min_steps = getenv("GGML_METAL_MEGA_MIN_STEPS") ? atoi(getenv("GGML_METAL_MEGA_MIN_STEPS")) : 2;
        g_mega_T = getenv("GGML_METAL_MEGA_TPT") ? atoi(getenv("GGML_METAL_MEGA_TPT")) : 256;
        g_mega_lock = [NSLock new];
        g_mega.old_psos = [NSMutableArray new];
        if (g_mega.on) {
            fprintf(stderr, "ggml_metal_mega: on, %d threadgroups of %d threads\n", g_mega.R, GGML_METAL_MEGA_THREADS);
        }
    }
    return g_mega.on == 1;
}

void ggml_metal_mega_next_encoder(void) {
    if (ggml_metal_mega_enabled() && !getenv("GGML_METAL_MEGA_NOREC")) {
        t_mega_next = true;
    }
}

static const int k_mega_kinds[] = { GGML_METAL_LIB_MISC, GGML_METAL_LIB_NORM, GGML_METAL_LIB_UNARY, GGML_METAL_LIB_BINBCAST,
                                    GGML_METAL_LIB_ARGSORT, GGML_METAL_LIB_QUANTIZE, GGML_METAL_LIB_MUL_MV,
                                    GGML_METAL_LIB_FA_VEC_F16 };

static bool ggml_metal_mega_kind_ok(int kind) {
    for (size_t i = 0; i < sizeof(k_mega_kinds)/sizeof(k_mega_kinds[0]); ++i) {
        if (k_mega_kinds[i] == kind) return true;
    }
    return false;
}

#define GGML_METAL_MEGA_SPLIT "// @@GGML_METAL_MEGA_SPLIT"

// the part of an embedded kind source after the shared headers (or before them: head)
static char * ggml_metal_mega_kind_text(ggml_metal_library_t lib, int kind, bool head) {
    const char * s = lib->src_start[kind];
    const char * e = lib->src_end[kind];
    if (!s || !e || e <= s) return NULL;
    const size_t n = e - s;
    const char * m = NULL;
    const size_t lm = strlen(GGML_METAL_MEGA_SPLIT);
    for (size_t i = 0; i + lm <= n; ++i) {
        if (s[i] == '/' && memcmp(s + i, GGML_METAL_MEGA_SPLIT, lm) == 0) { m = s + i; break; }
    }
    if (!m) return NULL;
    const char * a = head ? s : m + lm;
    const char * b = head ? m : e;
    char * r = malloc(b - a + 1);
    memcpy(r, a, b - a);
    r[b - a] = 0;
    return r;
}

// can R threadgroups of the persistent kernel all be on the GPU at once? (else a grid barrier waits for one that never
// starts) two empty steps with a barrier between them and a short spin limit
static bool ggml_metal_mega_probe(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pso, int R) {
    id<MTLBuffer> b = [[queue device] newBufferWithLength:4096 options:MTLResourceStorageModeShared];
    memset([b contents], 0, 4096);
    ggml_metal_mega_step * st = (ggml_metal_mega_step *) ((char *) [b contents] + 256);
    st[0].e0 = st[0].e1 = st[0].n_items = 0;
    st[1].e0 = st[1].e1 = st[1].n_items = 0;
    ggml_metal_mega_params prm = { 2, { 1u << 14, 0, 0 } };
    id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:b offset:1024 atIndex:0];
    [enc setBuffer:b offset:512  atIndex:1];
    [enc setBuffer:b offset:256  atIndex:2];
    [enc setBuffer:b offset:0    atIndex:3];
    [enc setBytes:&prm length:sizeof(prm) atIndex:4];
    // as much threadgroup memory as a dispatch may take (see ggml_metal_mega_layout): fewer threadgroups fit with more
    [enc setThreadgroupMemoryLength:32768 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(R, 1, 1) threadsPerThreadgroup:MTLSizeMake(GGML_METAL_MEGA_THREADS, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    const bool ok = ((const uint32_t *) [b contents])[1] == 0 && [cmd status] == MTLCommandBufferStatusCompleted;
    [b release];
    return ok;
}

// a pipeline of the megakernel library that is told how many threads a threadgroup it is dispatched with: by default the
// compiler budgets registers for 1024 and spills (GGML_METAL_MEGA_MAXTPT=0 keeps the default, for comparison)
static id<MTLComputePipelineState> ggml_metal_mega_pipeline(id<MTLDevice> device, id<MTLFunction> fn, NSError ** error) {
    if (!fn) return nil;
    static int lim = -2;
    if (lim == -2) lim = getenv("GGML_METAL_MEGA_MAXTPT") ? atoi(getenv("GGML_METAL_MEGA_MAXTPT")) : -1;
    MTLComputePipelineDescriptor * desc = [[MTLComputePipelineDescriptor alloc] init];
    desc.computeFunction = fn;
    if (lim != 0) desc.maxTotalThreadsPerThreadgroup = lim > 0 ? lim : GGML_METAL_MEGA_THREADS;
    id<MTLComputePipelineState> p = [device newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:error];
    [desc release];
    return p;
}

// make sure the megakernel has a case for every pipeline of the recording that it can have one for
static void ggml_metal_mega_update(struct ggml_metal_mega_rec * r) {
    [g_mega_lock lock];
    bool missing = false;
    for (int i = 0; i < r->n && !missing; ++i) {
        ggml_metal_pipeline_t pl = r->d[i].pl;
        if (pl->mega_never || pl->mega_gen == g_mega.gen) continue;
        if (pl->kind < 0 || !ggml_metal_mega_kind_ok(pl->kind) || !g_mega.lib || (!ggml_metal_mega_included(pl->base) && !getenv("GGML_METAL_MEGA_LIB_ALL"))) { pl->mega_never = true; continue; }
        missing = true;
    }
    if (getenv("GGML_METAL_MEGA_VERBOSE") && r->n > 0) {
        static int nv = 0;
        if (nv++ < 2) fprintf(stderr, "ggml_metal_mega: update: missing %d on %d lib %p first %s kind %d never %d gen %d/%d\n", missing, g_mega.on, (void *) g_mega.lib, r->d[0].pl->base, r->d[0].pl->kind, r->d[0].pl->mega_never, r->d[0].pl->mega_gen, g_mega.gen);
    }
    if (!missing || g_mega.on != 1 || getenv("GGML_METAL_MEGA_NOBUILD")) {
        [g_mega_lock unlock];
        return;
    }
    for (int i = 0; i < r->n; ++i) {
        ggml_metal_pipeline_t pl = r->d[i].pl;
        if (pl->mega_never) continue;
        if (!ggml_metal_mega_included(pl->base) && !getenv("GGML_METAL_MEGA_LIB_ALL")) { pl->mega_never = true; continue; }
        bool have = false;
        for (int j = 0; j < g_mega.n_pipes; ++j) if (g_mega.pipes[j] == pl) { have = true; break; }
        if (have) continue;
        if (g_mega.n_pipes == g_mega.cap_pipes) {
            g_mega.cap_pipes = g_mega.cap_pipes ? 2*g_mega.cap_pipes : 64;
            g_mega.pipes = realloc(g_mega.pipes, g_mega.cap_pipes*sizeof(*g_mega.pipes));
        }
        g_mega.pipes[g_mega.n_pipes++] = pl;
    }

    const int64_t t0 = ggml_time_us();
    ggml_metal_library_t lib = g_mega.lib;
    char * hdr = ggml_metal_mega_kind_text(lib, GGML_METAL_LIB_QUANTIZE, true);
    const char * ksrc[GGML_METAL_LIB_COUNT] = { 0 };
    for (int k = 0; k < GGML_METAL_LIB_COUNT; ++k) {
        ksrc[k] = ggml_metal_mega_kind_ok(k) ? ggml_metal_mega_kind_text(lib, k, false) : NULL;
    }
    const int np = g_mega.n_pipes;
    struct ggml_metal_mega_pipe * mp = calloc(np, sizeof(*mp));
    int * cases = calloc(np, sizeof(int));
    int * flags = calloc(np, sizeof(int));
    for (int i = 0; i < np; ++i) {
        ggml_metal_pipeline_t pl = g_mega.pipes[i];
        mp[i].base = pl->base;
        mp[i].kind = pl->kind;
        mp[i].n_fc = pl->n_fc;
        for (int j = 0; j < pl->n_fc; ++j) {
            mp[i].fc_idx[j] = pl->fc_idx[j]; mp[i].fc_type[j] = pl->fc_type[j]; mp[i].fc_val[j] = pl->fc_val[j];
        }
    }
    char * src = hdr ? ggml_metal_mega_source(hdr, ksrc, GGML_METAL_LIB_COUNT, mp, np, cases, flags) : NULL;
    if (!src) {
        fprintf(stderr, "ggml_metal_mega: no source (shared headers %s), off\n", hdr ? "found" : "not found");
    }
    id<MTLComputePipelineState> pso = nil;
    if (src) {
        if (getenv("GGML_METAL_MEGA_DUMP")) {
            FILE * f = fopen(getenv("GGML_METAL_MEGA_DUMP"), "w");
            if (f) { fputs(src, f); fclose(f); }
        }
        @autoreleasepool {
            id<MTLDevice> device = ggml_metal_device_get_obj(lib->dev);
            MTLCompileOptions * options = [MTLCompileOptions new];
            options.preprocessorMacros = lib->prep;
            options.languageVersion = ggml_metal_device_get_props(lib->dev)->has_tensor ?
                (MTLLanguageVersion) MTLLanguageVersion4_0_GGML : MTLLanguageVersion3_2;
            NSError * error = nil;
            id<MTLLibrary> mlib = [device newLibraryWithSource:[NSString stringWithUTF8String:src] options:options error:&error];
            [options release];
            if (!mlib) {
                const char * msg = [[error description] UTF8String];
                fprintf(stderr, "ggml_metal_mega: the megakernel does not compile, off: %.4000s\n", msg ? msg : "?");
            } else {
                id<MTLFunction> fn = [mlib newFunctionWithName:@"kernel_mega"];
                pso = ggml_metal_mega_pipeline(device, fn, &error);
                [fn release];
                id<MTLFunction> fn1 = [mlib newFunctionWithName:@"kernel_mega_one"];
                if (g_mega.pso_one) [g_mega.old_psos addObject:g_mega.pso_one];
                g_mega.pso_one = ggml_metal_mega_pipeline(device, fn1, &error);
                [fn1 release];
                if (pso) {
                    fprintf(stderr, "ggml_metal_mega: kernel_mega max %d threads a threadgroup, kernel_mega_one %d\n",
                            (int) pso.maxTotalThreadsPerThreadgroup, g_mega.pso_one ? (int) g_mega.pso_one.maxTotalThreadsPerThreadgroup : 0);
                }
                [mlib release];
                if (pso && pso.maxTotalThreadsPerThreadgroup < GGML_METAL_MEGA_THREADS) {
                    fprintf(stderr, "ggml_metal_mega: the megakernel runs only %d threads a threadgroup, off\n", (int) pso.maxTotalThreadsPerThreadgroup);
                    [pso release];
                    pso = nil;
                }
            }
        }
    }
    if (pso) {
        // the most threadgroups that are all resident, from the asked number down
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(lib->dev);
        const int want = g_mega.R;
        const int tries[] = { want, want*3/4, want/2, want/4 };
        int got = 0;
        for (int k = 0; k < 4 && !got; ++k) {
            if (tries[k] >= 8 && ggml_metal_mega_probe(queue, pso, tries[k])) got = tries[k];
        }
        if (!got) {
            fprintf(stderr, "ggml_metal_mega: not even %d threadgroups are resident at once, off\n", want/4);
            [pso release];
            pso = nil;
        } else if (got != want) {
            fprintf(stderr, "ggml_metal_mega: %d threadgroups are not resident at once, %d are\n", want, got);
            g_mega.R = got;
        }
    }
    if (!pso) {
        g_mega.on = 0;
    } else {
        if (g_mega.pso) [g_mega.old_psos addObject:g_mega.pso];
        g_mega.pso = pso;
        g_mega.gen++;
        int n_ok = 0;
        for (int i = 0; i < np; ++i) {
            ggml_metal_pipeline_t pl = g_mega.pipes[i];
            pl->mega_gen   = g_mega.gen;
            pl->mega_case  = cases[i];
            pl->mega_flags = flags[i];
            n_ok += cases[i] >= 0;
        }
        fprintf(stderr, "ggml_metal_mega: megakernel %d with %d of %d kernels built in %.2f s\n", g_mega.gen, n_ok, np,
                (ggml_time_us() - t0)/1e6);
    }
    free(src); free(hdr); free(mp); free(cases); free(flags);
    for (int k = 0; k < GGML_METAL_LIB_COUNT; ++k) free((void *) ksrc[k]);
    [g_mega_lock unlock];
}

// layout of a dispatch in the persistent kernel; false if it cannot run there
static const char * g_mega_why = "";

// GGML_METAL_MEGA_INCLUDE=prefix,prefix: only these kernels go in (to find a wrong or slow one); also kept out of the generated source
static bool ggml_metal_mega_included(const char * base) {
    static const char * in = (const char *) -1;
    if (in == (const char *) -1) in = getenv("GGML_METAL_MEGA_INCLUDE");
    if (!in) return true;
    const char * p = in;
    while (*p) {
        const char * q = strchr(p, 0x2c);
        const size_t n = q ? (size_t) (q - p) : strlen(p);
        if (n > 0 && strncmp(base, p, n) == 0) return true;
        if (!q) break;
        p = q + 1;
    }
    return false;
}

static bool ggml_metal_mega_layout(const struct ggml_metal_mega_disp * d, ggml_metal_mega_entry * e) {
    const ggml_metal_pipeline_t pl = d->pl;
    if (pl->mega_gen != g_mega.gen || pl->mega_case < 0 || !(pl->mega_flags & GGML_METAL_MEGA_FLAG_OK)) { g_mega_why = "r1"; return false; }
    {
        // GGML_METAL_MEGA_EXCLUDE=prefix,prefix: kernels kept out (encoded as usual)
        // GGML_METAL_MEGA_INCLUDE=prefix,prefix: only these kernels go in (to find a wrong one)
        static const char * in = (const char *) -1;
        if (in == (const char *) -1) in = getenv("GGML_METAL_MEGA_INCLUDE");
        if (in) {
            bool hit = false;
            const char * p = in;
            while (*p) {
                const char * q = strchr(p, ',');
                const size_t n = q ? (size_t) (q - p) : strlen(p);
                if (n > 0 && strncmp(pl->base, p, n) == 0) { hit = true; break; }
                if (!q) break;
                p = q + 1;
            }
            if (!hit) { g_mega_why = "include"; return false; }
        }
        static const char * ex = (const char *) -1;
        if (ex == (const char *) -1) ex = getenv("GGML_METAL_MEGA_EXCLUDE");
        if (ex) {
            const char * p = ex;
            while (*p) {
                const char * q = strchr(p, ',');
                const size_t n = q ? (size_t) (q - p) : strlen(p);
                if (n > 0 && strncmp(pl->base, p, n) == 0) { g_mega_why = "r2"; return false; }
                if (!q) break;
                p = q + 1;
            }
        }
    }
    const int t = d->tptg[0]*d->tptg[1]*d->tptg[2];
    const int T = GGML_METAL_MEGA_THREADS;
    const bool nob = (pl->mega_flags & GGML_METAL_MEGA_FLAG_NO_BARRIER) != 0;
    if (d->nb > GGML_METAL_MEGA_NB) { g_mega_why = "r3"; return false; }
    // the constant parameters must have been set with set_bytes
    const int cmask = pl->mega_flags >> GGML_METAL_MEGA_FLAG_CONST_SHIFT;
    for (int i = 0; i < d->nb; ++i) {
        if ((cmask >> i) & 1) {
            if (d->b[i].bytes_off < 0) { g_mega_why = "r4"; return false; }
        }
    }
    memset(e, 0, sizeof(*e));
    e->pipe = pl->mega_case;
    e->reps = 1;
    if (t == T) {
        // as dispatched, any shape
        e->lanes = 1; e->width = T;
        e->tptg[0] = d->tptg[0]; e->tptg[1] = d->tptg[1]; e->tptg[2] = d->tptg[2];
    } else if (t > T && nob) {
        // wider than the persistent threadgroup and no barrier inside: each thread runs several of its threads, the
        // kernel sees its own width (kernel_get_rows has a thread an element: run narrower it copied a quarter)
        e->lanes = 1; e->width = T;
        e->reps = (t + T - 1)/T;
        e->tptg[0] = d->tptg[0]; e->tptg[1] = d->tptg[1]; e->tptg[2] = d->tptg[2];
    } else if (t > T) {
        // with barriers it can only run narrower: only for kernels that stride over their threads
        static const char * gen[] = { "kernel_rms_norm", "kernel_depth_mix", "kernel_ctrl_down", "kernel_norm_router" };
        bool ok = false;
        for (size_t k = 0; k < sizeof(gen)/sizeof(gen[0]); ++k) if (strncmp(pl->base, gen[k], strlen(gen[k])) == 0) ok = true;
        if (!ok || d->tptg[1] != 1 || d->tptg[2] != 1) { g_mega_why = "r5"; return false; }
        e->lanes = 1; e->width = T;
        e->tptg[0] = T; e->tptg[1] = 1; e->tptg[2] = 1;
    } else {
        e->lanes = T/t; e->width = t;
        if (T % t != 0 && !nob) { g_mega_why = "r6"; return false; } // threads left over would sit out of a barrier
        e->tptg[0] = d->tptg[0]; e->tptg[1] = d->tptg[1]; e->tptg[2] = d->tptg[2];
    }
    e->n_vtg   = d->grid[0]*d->grid[1]*d->grid[2];
    if (e->n_vtg <= 0) { g_mega_why = "r7"; return false; }
    e->n_items = (e->n_vtg + e->lanes - 1)/e->lanes;
    e->idle_ok = nob ? 1 : 0;
    e->grid[0] = d->grid[0]; e->grid[1] = d->grid[1]; e->grid[2] = d->grid[2];
    const int s0 = (int) ((d->tg[0] + 15) & ~(size_t) 15);
    const int s1 = (int) ((d->tg[1] + 15) & ~(size_t) 15);
    e->smem0     = s0;
    e->smem_lane = s0 + s1;
    if (e->lanes*e->smem_lane > 32768) { g_mega_why = "r8"; return false; }
    return true;
}

static void ggml_metal_mega_encode_plain(id<MTLCommandBuffer> cmd, bool concurrent, struct ggml_metal_mega_rec * r, int d0, int d1) {
    // each encoder released right after it ends, as the normal encoding does
    @autoreleasepool {
    int slot = -1;
    id<MTLComputeCommandEncoder> enc = nil;
    if (getenv("GGML_METAL_MEGA_DESC")) {
        MTLComputePassDescriptor * pd = [MTLComputePassDescriptor computePassDescriptor];
        pd.dispatchType = concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial;
        enc = [cmd computeCommandEncoderWithDescriptor:pd];
    } else {
        enc = ggml_metal_new_compute_encoder(cmd, concurrent, &slot);
    }
    for (int i = d0; i < d1; ++i) {
        const struct ggml_metal_mega_disp * d = &r->d[i];
        [enc setComputePipelineState:d->pl->obj];
        for (int j = 0; j < d->nb; ++j) {
            if (d->b[j].bytes_off >= 0) {
                [enc setBytes:r->bytes + d->b[j].bytes_off length:d->b[j].bytes_size atIndex:j];
            } else if (d->b[j].buf) {
                [enc setBuffer:d->b[j].buf offset:d->b[j].offs atIndex:j];
            }
        }
        for (int j = 0; j < 2; ++j) {
            if (d->tg[j]) [enc setThreadgroupMemoryLength:d->tg[j] atIndex:j];
        }
        [enc dispatchThreadgroups:MTLSizeMake(d->grid[0], d->grid[1], d->grid[2])
            threadsPerThreadgroup:MTLSizeMake(d->tptg[0], d->tptg[1], d->tptg[2])];
    }
    [enc endEncoding];
    }
}

static void ggml_metal_mega_flush(ggml_metal_encoder_t encoder) {
    struct ggml_metal_mega_rec * r = encoder->mega;
    id<MTLCommandBuffer> cmd = encoder->cmd;
    if (r->n == 0) return;

    const int64_t t_flush0 = ggml_time_us();
    ggml_metal_mega_update(r);

    const int nd = r->n;
    const int ns = r->d[nd - 1].step + 1;
    ggml_metal_mega_entry * ent = malloc(nd*sizeof(*ent));
    bool * dok = malloc(nd);
    int  * sfirst = malloc((ns + 1)*sizeof(int));
    bool * sok = malloc(ns);
    for (int s = 0; s <= ns; ++s) sfirst[s] = nd;
    for (int s = 0; s < ns; ++s) sok[s] = true;
    for (int i = nd - 1; i >= 0; --i) sfirst[r->d[i].step] = i;
    for (int s = ns - 1; s >= 0; --s) if (sfirst[s] == nd) sfirst[s] = sfirst[s + 1];
    for (int i = 0; i < nd; ++i) {
        dok[i] = g_mega.on == 1 && !r->failed && ggml_metal_mega_layout(&r->d[i], &ent[i]);
        if (!dok[i]) sok[r->d[i].step] = false;
    }

    const int64_t t_flush1 = ggml_time_us();
    bool any_ok = false;
    for (int k = 0; k < ns && !any_ok; ++k) any_ok = sok[k];
    if (!any_ok && getenv("GGML_METAL_MEGA_NORING")) {
        for (int k = 0; k < ns; ++k) ggml_metal_mega_encode_plain(cmd, encoder->concurrent, r, sfirst[k], sfirst[k + 1]);
        free(ent); free(dok); free(sfirst); free(sok);
        return;
    }
    // a ring slot for this command buffer
    const int ri = g_mega.ring_i++ % GGML_METAL_MEGA_RING;
    if (g_mega.ring[ri].cmd) {
        // a pre-encoded command buffer that was dropped is never committed: waiting for it would block forever
        const MTLCommandBufferStatus st = [g_mega.ring[ri].cmd status];
        if (st >= MTLCommandBufferStatusCommitted && st < MTLCommandBufferStatusCompleted) {
            [g_mega.ring[ri].cmd waitUntilCompleted];
        }
        const uint32_t * base = (const uint32_t *) [g_mega.ring[ri].buf contents];
        for (size_t k = 0; k < g_mega.ring[ri].n_ctr; ++k) {
            const uint32_t nto = base[g_mega.ring[ri].ctr[k]/4 + 1];
            if (nto) {
                fprintf(stderr, "ggml_metal_mega: %u grid barrier waits timed out (results of that graph are garbage), off\n", nto);
                g_mega.on = 0;
            }
        }
        [g_mega.ring[ri].cmd release];
        g_mega.ring[ri].cmd = nil;
    }
    g_mega.ring[ri].n_ctr = 0;
    // its size: every dispatch an entry, every step one, all the bytes, a counter a run, alignment
    size_t need = 256 + nd*sizeof(ggml_metal_mega_entry) + ns*sizeof(ggml_metal_mega_step) + r->n_bytes + 1024*ns + 1024 + 16*nd;
    if (!g_mega.ring[ri].buf || [g_mega.ring[ri].buf length] < need) {
        if (g_mega.ring[ri].buf) [g_mega.ring[ri].buf release];
        id<MTLDevice> device = [cmd device];
        g_mega.ring[ri].buf = [device newBufferWithLength:MAX(need*3/2, (size_t) 4 << 20) options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> rb = g_mega.ring[ri].buf;
    uint8_t * rbase = (uint8_t *) [rb contents];
    size_t cursor = 0;
    #define MEGA_ALLOC(sz) ({ cursor = (cursor + 255) & ~(size_t) 255; size_t _o = cursor; cursor += (sz); _o; })

    const int64_t t_flush2 = ggml_time_us();

    // GGML_METAL_MEGA_NOCOH: steps where a persistent dispatch must end so that no plain load can see a stale line.
    // the same graph has the same ranges every token: worked out once a signature
    static int nocoh = -1;
    if (nocoh < 0) nocoh = (getenv("GGML_METAL_MEGA_NOCOH") != NULL || getenv("GGML_METAL_MEGA_COHSTORE") != NULL) && getenv("GGML_METAL_MEGA_NOSTALECHECK") == NULL;
    static uint64_t cut_sig = 0;
    static bool * cut = NULL;
    static int cut_n = 0;
    if (nocoh) {
        uint64_t sig = 1469598103934665603ull ^ (uint64_t) nd ^ ((uint64_t) ns << 32);
        for (int i = 0; i < r->n_rng; ++i) {
            sig = (sig ^ r->rng[i].lo)*1099511628211ull;
            sig = (sig ^ (r->rng[i].hi + (uint64_t) r->rng[i].w))*1099511628211ull;
        }
        for (int k = 0; k < ns; ++k) sig = (sig ^ (uint64_t) sok[k])*1099511628211ull;
        if (sig != cut_sig || cut_n != ns) {
            const int64_t tc = ggml_time_us();
            free(cut);
            cut = calloc(ns + 1, sizeof(bool));
            cut_n = ns;
            cut_sig = sig;
            int seg = 0, n_cut = 0;
            for (int k = 0; k < ns; ++k) {
                if (!sok[k]) { seg = k + 1; continue; }
                if (k > seg && ggml_metal_mega_stale(r, sfirst[seg], sfirst[k], sfirst[k + 1])) {
                    cut[k] = true;
                    seg = k;
                    n_cut++;
                }
            }
            int n_w = 0;
            for (int i = 0; i < r->n_rng; ++i) n_w += r->rng[i].w;
            fprintf(stderr, "ggml_metal_mega: plain loads: %d dispatch ends for stale lines (%d ranges, %d written; worked out in %.1f ms)\n",
                    n_cut, r->n_rng, n_w, (ggml_time_us() - tc)/1e3);
        }
    }

    int s = 0;
    while (s < ns) {
        int t = s;
        static int cut_every = -1;
        if (cut_every < 0) cut_every = getenv("GGML_METAL_MEGA_CUT_EVERY") ? atoi(getenv("GGML_METAL_MEGA_CUT_EVERY")) : 0;
        while (t < ns && sok[t] && !(nocoh && t > s && cut[t]) && !(cut_every > 0 && t - s >= cut_every)) ++t;
        if (t - s >= g_mega.min_steps && g_mega.ring[ri].n_ctr < 1024) {
            // steps [s, t) in one persistent dispatch
            const int e0 = sfirst[s], e1 = sfirst[t];
            const size_t o_ctr = MEGA_ALLOC(256); // counter, timeouts, and (at +128) the flag of GGML_METAL_MEGA_FLAGBAR
            const size_t o_stp = MEGA_ALLOC((t - s)*sizeof(ggml_metal_mega_step));
            const size_t o_ent = MEGA_ALLOC((e1 - e0)*sizeof(ggml_metal_mega_entry));
            size_t argsz = 0;
            for (int i = e0; i < e1; ++i) for (int j = 0; j < r->d[i].nb; ++j) {
                if (r->d[i].b[j].bytes_off >= 0) argsz += (r->d[i].b[j].bytes_size + 15) & ~15;
            }
            const size_t o_arg = MEGA_ALLOC(MAX(argsz, (size_t) 16));
            memset(rbase + o_ctr, 0, 256);
            g_mega.ring[ri].ctr[g_mega.ring[ri].n_ctr++] = o_ctr;
            ggml_metal_mega_step * stp = (ggml_metal_mega_step *) (rbase + o_stp);
            ggml_metal_mega_entry * en = (ggml_metal_mega_entry *) (rbase + o_ent);
            size_t ao = 0;
            id<MTLBuffer> res[64];
            int n_res = 0;
            int smem = 16;
            for (int k = s; k < t; ++k) {
                stp[k - s].e0 = sfirst[k] - e0;
                stp[k - s].e1 = sfirst[k + 1] - e0;
                stp[k - s].n_items = 0;
                stp[k - s].pad = 0;
                for (int i = sfirst[k]; i < sfirst[k + 1]; ++i) stp[k - s].n_items += ent[i].n_items;
            }
            for (int i = e0; i < e1; ++i) {
                ggml_metal_mega_entry * e = &en[i - e0];
                *e = ent[i];
                { static int os = -1; if (os < 0) os = getenv("GGML_METAL_MEGA_ONE_STOP") ? atoi(getenv("GGML_METAL_MEGA_ONE_STOP")) : 0; e->pad0 = os; }
                const struct ggml_metal_mega_disp * d = &r->d[i];
                for (int j = 0; j < d->nb && j < GGML_METAL_MEGA_NB; ++j) {
                    if (d->b[j].bytes_off >= 0) {
                        memcpy(rbase + o_arg + ao, r->bytes + d->b[j].bytes_off, d->b[j].bytes_size);
                        e->a[j] = (uint32_t) ao;
                        ao += (d->b[j].bytes_size + 15) & ~15;
                    } else if (d->b[j].buf) {
                        e->p[j] = [d->b[j].buf gpuAddress] + d->b[j].offs;
                        bool have = false;
                        for (int q = 0; q < n_res; ++q) if (res[q] == d->b[j].buf) { have = true; break; }
                        if (!have && n_res < 63) res[n_res++] = d->b[j].buf;
                    }
                }
                smem = MAX(smem, e->lanes*e->smem_lane);
            }
            res[n_res++] = rb;
            ggml_metal_mega_params prm = { (uint32_t) (t - s), { 0, 0, 0 } };
            {
                // profiling, results are garbage: GGML_METAL_MEGA_STOP=1 (return at once), 2 (steps and barriers
                // only), 3 (no kernel calls); GGML_METAL_MEGA_UPTO=n: work only in the first n steps of the graph
                static int stop = -1, upto = -1;
                if (stop < 0) {
                    stop = getenv("GGML_METAL_MEGA_STOP") ? atoi(getenv("GGML_METAL_MEGA_STOP")) : 0;
                    upto = getenv("GGML_METAL_MEGA_UPTO") ? atoi(getenv("GGML_METAL_MEGA_UPTO")) : -1;
                }
                prm.pad[1] = (uint32_t) stop | (getenv("GGML_METAL_MEGA_FLAGBAR") ? 0x100u : 0u) | (getenv("GGML_METAL_MEGA_NOFENCE") ? 0x200u : 0u);
                if (upto >= 0) {
                    const int n_exec = MAX(0, MIN(upto - s, t - s));
                    prm.pad[2] = (uint32_t) n_exec + 1;
                }
            }

            static int oneshot = -1;
            if (oneshot < 0) oneshot = getenv("GGML_METAL_MEGA_ONESHOT") != NULL;
            if (oneshot && g_mega.pso_one) {
                // the generated code, launched as ordinary dispatches: an encoder a step, a threadgroup an item
                for (int k = s; k < t; ++k) {
                    int slot = -1;
                    id<MTLComputeCommandEncoder> enc = ggml_metal_new_compute_encoder(cmd, encoder->concurrent, &slot);
                    [enc setComputePipelineState:g_mega.pso_one];
                    [enc setBuffer:rb offset:o_arg atIndex:0];
                    if (!getenv("GGML_METAL_MEGA_NOUSE")) [enc useResources:res count:n_res usage:MTLResourceUsageRead | MTLResourceUsageWrite];
                    for (int i = sfirst[k]; i < sfirst[k + 1]; ++i) {
                        const ggml_metal_mega_entry * e = &en[i - e0];
                        [enc setBuffer:rb offset:o_ent + (i - e0)*sizeof(ggml_metal_mega_entry) atIndex:1];
                        [enc setThreadgroupMemoryLength:MAX(16, (e->lanes*e->smem_lane + 15) & ~15) atIndex:0];
                        [enc dispatchThreadgroups:MTLSizeMake(e->n_items, 1, 1) threadsPerThreadgroup:MTLSizeMake(GGML_METAL_MEGA_THREADS, 1, 1)];
                    }
                    [enc endEncoding];
                }
                g_mega.n_steps_mega += t - s;
                s = t;
                continue;
            }
            int slot = -1;
            id<MTLComputeCommandEncoder> enc = ggml_metal_new_compute_encoder(cmd, encoder->concurrent, &slot);
            [enc setComputePipelineState:g_mega.pso];
            [enc setBuffer:rb offset:o_arg atIndex:0];
            [enc setBuffer:rb offset:o_ent atIndex:1];
            [enc setBuffer:rb offset:o_stp atIndex:2];
            [enc setBuffer:rb offset:o_ctr atIndex:3];
            [enc setBytes:&prm length:sizeof(prm) atIndex:4];
            [enc setThreadgroupMemoryLength:(smem + 15) & ~15 atIndex:0];
            if (!getenv("GGML_METAL_MEGA_NOUSE")) [enc useResources:res count:n_res usage:MTLResourceUsageRead | MTLResourceUsageWrite];
            [enc dispatchThreadgroups:MTLSizeMake(g_mega.R, 1, 1) threadsPerThreadgroup:MTLSizeMake(GGML_METAL_MEGA_THREADS, 1, 1)];
            [enc endEncoding];
            g_mega.n_steps_mega += t - s;
            g_mega.n_disp++;
            s = t;
        } else {
            const int t1 = t > s ? t : s + 1;
            for (int k = s; k < t1; ++k) {
                ggml_metal_mega_encode_plain(cmd, getenv("GGML_METAL_MEGA_SERIAL") ? false : true, r, sfirst[k], sfirst[k + 1]);
                static int pc = 0; if (pc++ == 0 && getenv("GGML_METAL_MEGA_VERBOSE")) fprintf(stderr, "ggml_metal_mega: encoder concurrent = %d\n", (int) encoder->concurrent);
                g_mega.n_steps_plain++;
            }
            s = t1;
        }
    }
    #undef MEGA_ALLOC
    const int64_t t_flush3 = ggml_time_us();
    g_mega.ring[ri].cmd = [cmd retain];

    static int verbose = -1;
    if (verbose < 0) verbose = getenv("GGML_METAL_MEGA_VERBOSE") != NULL;
    if (verbose) {
        static int nf = 0;
        if (nf++ < 4) {
            int nok = 0;
            for (int k = 0; k < ns; ++k) nok += sok[k];
            fprintf(stderr, "ggml_metal_mega: graph of %d dispatches, %d steps, %d of them in the megakernel\n", nd, ns, nok);
            for (int i = 0; i < nd && nf == 1; ++i) {
                if (!dok[i]) {
                    ggml_metal_mega_entry tmp;
                    ggml_metal_mega_layout(&r->d[i], &tmp);
                    fprintf(stderr, "  not in it (%s, nb %d, tg %zu %zu): %s (grid %d %d %d, threads %d %d %d, case %d, flags 0x%x)\n",
                            g_mega_why, r->d[i].nb, r->d[i].tg[0], r->d[i].tg[1], r->d[i].pl->base,
                            r->d[i].grid[0], r->d[i].grid[1], r->d[i].grid[2], r->d[i].tptg[0], r->d[i].tptg[1], r->d[i].tptg[2],
                            r->d[i].pl->mega_case, r->d[i].pl->mega_flags);
                }
            }
        }
    }
    free(ent); free(dok); free(sfirst); free(sok);
    {
        static int64_t t_sum = 0, t_a = 0, t_b = 0, t_c = 0; static int n_sum = 0;
        t_sum += ggml_time_us() - t_flush0;
        t_a += t_flush1 - t_flush0; t_b += t_flush2 - t_flush1; t_c += t_flush3 - t_flush2;
        if (++n_sum % 16 == 0 && verbose) fprintf(stderr, "ggml_metal_mega: flush %.0f us on average (layout %.0f, ring %.0f, encode %.0f)\n",
            (double) t_sum/n_sum, (double) t_a/n_sum, (double) t_b/n_sum, (double) t_c/n_sum);
    }
}

static struct ggml_metal_mega_rec * ggml_metal_mega_rec_init(void) {
    struct ggml_metal_mega_rec * r = calloc(1, sizeof(*r));
    r->cap = 2048;
    r->d = malloc(r->cap*sizeof(*r->d));
    r->cap_bytes = 1 << 20;
    r->bytes = malloc(r->cap_bytes);
    for (int j = 0; j < GGML_METAL_RECORDING_BINDS; ++j) r->cur.b[j].bytes_off = -1;
    return r;
}

static void ggml_metal_mega_rec_free(struct ggml_metal_mega_rec * r) {
    free(r->d);
    free(r->bytes);
    free(r->rng);
    free(r);
}

int ggml_metal_encoder_mega_n_disp(ggml_metal_encoder_t encoder) {
    return encoder->mega ? encoder->mega->n : -1;
}

void ggml_metal_encoder_mega_ranges(ggml_metal_encoder_t encoder, int first_disp, const uint64_t * rr, int nr, const uint64_t * ww, int nw) {
    struct ggml_metal_mega_rec * r = encoder->mega;
    if (!r || first_disp >= r->n) return;
    if (r->n_rng + nr + nw > r->cap_rng) {
        r->cap_rng = MAX(2*r->cap_rng, r->n_rng + nr + nw + 4096);
        r->rng = realloc(r->rng, r->cap_rng*sizeof(*r->rng));
    }
    const int r0 = r->n_rng;
    for (int i = 0; i < nr; ++i) { r->rng[r->n_rng].lo = rr[2*i]; r->rng[r->n_rng].hi = rr[2*i + 1]; r->rng[r->n_rng].w = 0; r->n_rng++; }
    for (int i = 0; i < nw; ++i) { r->rng[r->n_rng].lo = ww[2*i]; r->rng[r->n_rng].hi = ww[2*i + 1]; r->rng[r->n_rng].w = 1; r->n_rng++; }
    for (int i = first_disp; i < r->n; ++i) {
        r->d[i].r0 = r0;
        r->d[i].r1 = r->n_rng;
    }
}

// GGML_METAL_MEGA_NOCOH (plain cached loads): could a step read a stale line, given the steps before it in the same
// persistent dispatch? that needs a range it reads, written by an earlier step j of the dispatch, and touched
// (read or written) by a step before j: a core may still hold the line from that access. ranges are 128-byte lines
static bool ggml_metal_mega_stale(const struct ggml_metal_mega_rec * r, int seg_d0, int d0, int d1) {
    #define LN(x) ((x) & ~(uint64_t) 255)
    #define HN(x) (((x) + 255) & ~(uint64_t) 255)
    for (int i = d0; i < d1; ++i) {
        const struct ggml_metal_mega_disp * d = &r->d[i];
        for (int a = d->r0; a < d->r1; ++a) {
            if (r->rng[a].w) continue;
            const uint64_t lo = LN(r->rng[a].lo), hi = HN(r->rng[a].hi);
            // writes in the dispatch before this step that overlap
            for (int j = seg_d0; j < d0; ++j) {
                const struct ggml_metal_mega_disp * dj = &r->d[j];
                for (int b = dj->r0; b < dj->r1; ++b) {
                    if (!r->rng[b].w) continue;
                    const uint64_t wlo = MAX(lo, LN(r->rng[b].lo)), whi = MIN(hi, HN(r->rng[b].hi));
                    if (wlo >= whi) continue;
                    // anything touching [wlo, whi) in dj s step or before (an in-place op reads and writes in one step)
                    for (int k = seg_d0; k < d0 && r->d[k].step <= dj->step; ++k) {
                        const struct ggml_metal_mega_disp * dk = &r->d[k];
                        for (int c = dk->r0; c < dk->r1; ++c) {
                            // in the writer's own step only reads count (the write itself is not an earlier touch)
                            if (dk->step == dj->step && r->rng[c].w) continue;
                            if (MAX(wlo, LN(r->rng[c].lo)) < MIN(whi, HN(r->rng[c].hi))) return true;
                        }
                    }
                }
            }
        }
    }
    #undef LN
    #undef HN
    return false;
}

// GGML_METAL_KERNEL_LIST=file,skip,count: every dispatch after the first skip ones: kernel, grid, threads, threadgroup
// memory (as recorded by GGML_METAL_MEGA too, to compare its replay with the normal encoding)
static atomic_int g_kl_step = 0;
static uint64_t ggml_metal_kl_hash(const void * p, size_t n) { uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; ++i) h = (h ^ ((const uint8_t *) p)[i])*1099511628211ull; return h | 1; }

static void ggml_metal_kernel_list(id<MTLComputePipelineState> pso, int step, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2,
                                   size_t sm0, size_t sm1, const uint64_t * kb) {
    static FILE * kl = NULL; static int kl_on = -1, kl_skip = 0, kl_cnt = 0; static atomic_int kl_n = 0;
    if (kl_on < 0) {
        kl_on = 0;
        const char * v = getenv("GGML_METAL_KERNEL_LIST");
        if (v) { char tmp[512]; snprintf(tmp, sizeof(tmp), "%s", v); char * c = strchr(tmp, 0x2c); kl_cnt = 4000;
            if (c) { *c = 0; kl_skip = atoi(c + 1); char * c2 = strchr(c + 1, 0x2c); if (c2) kl_cnt = atoi(c2 + 1); }
            kl = fopen(tmp, "w"); kl_on = kl != NULL; }
    }
    if (kl_on == 1) {
        const int k = atomic_fetch_add(&kl_n, 1) - kl_skip;
        if (k >= 0 && k < kl_cnt) {
            fprintf(kl, "%d %d %s %d %d %d %d %d %d %zu %zu", k, step, [[pso label] UTF8String] ?: "?",
                    tg0, tg1, tg2, tptg0, tptg1, tptg2, sm0, sm1);
            for (int j = 0; j < 16; ++j) fprintf(kl, " %llx", (unsigned long long) kb[j]);
            fprintf(kl, "\n");
            if (k == kl_cnt - 1) fflush(kl);
        }
    }
}

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent) {
    ggml_metal_encoder_t res = calloc(1, sizeof(struct ggml_metal_encoder));

    if (t_mega_next && !t_recording_next) {
        t_mega_next = false;
        res->mega       = ggml_metal_mega_rec_init();
        res->cmd        = (id<MTLCommandBuffer>) cmd_buf_raw;
        res->concurrent = concurrent;
        res->tl_slot    = -1;
        g_mega_pt = getenv("GGML_METAL_MEGA_PASSTHRU") != NULL;
        if (g_mega_pt) { res->obj = ggml_metal_new_compute_encoder(res->cmd, concurrent, &res->tl_slot); [res->obj retain]; }
        return res;
    }
    t_mega_next = false;

    if (t_recording_next) {
        res->rec = t_recording_next;
        t_recording_next = NULL;
        return res;
    }

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    res->obj = ggml_metal_new_compute_encoder(cmd_buf, concurrent, &res->tl_slot);

    [res->obj retain];

    res->cmd        = cmd_buf;
    res->concurrent = concurrent;

    return res;
}

void ggml_metal_encoder_free(ggml_metal_encoder_t encoder) {
    if (encoder->mega) {
        if (g_mega_pt) [encoder->obj release];
        ggml_metal_mega_rec_free(encoder->mega);
        free(encoder);
        return;
    }
    if (!encoder->rec) {
        [encoder->obj release];
    }
    free(encoder);
}

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name) {
    if (encoder->rec || encoder->mega) {
        return;
    }
    @autoreleasepool {
        [encoder->obj pushDebugGroup:[NSString stringWithCString:name encoding:NSUTF8StringEncoding]];
    }
}

void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder) {
    if (encoder->rec || encoder->mega) {
        return;
    }
    [encoder->obj popDebugGroup];
}

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, struct ggml_metal_pipeline_with_params pipeline) {
    if (!pipeline.pipeline) {
        GGML_ABORT("%s: nil Metal pipeline (missing kernel; see compile_pipeline log above)\n", __func__);
    }

    if (encoder->mega) {
        struct ggml_metal_mega_rec * r = encoder->mega;
        // threadgroup memory lengths stay set in a Metal encoder (a step) until set again: keep them
        const size_t tg0 = r->step_used ? r->cur.tg[0] : 0, tg1 = r->step_used ? r->cur.tg[1] : 0;
        memset(&r->cur, 0, sizeof(r->cur));
        r->cur.tg[0] = tg0;
        r->cur.tg[1] = tg1;
        for (int j = 0; j < GGML_METAL_RECORDING_BINDS; ++j) r->cur.b[j].bytes_off = -1;
        r->cur.pl = pipeline.pipeline;
        if (!g_mega_pt) return;
    }

    if (encoder->rec) {
        // every op sets its pipeline, then all that its kernel reads: what an earlier kernel had bound is dropped, so
        // that a command carries only its own buffers. GGML_METAL_REPLAY_KEEP_BINDS: keep them, as a Metal encoder does
        static int keep = -1;
        if (keep < 0) {
            keep = getenv("GGML_METAL_REPLAY_KEEP_BINDS") != NULL;
        }
        if (!keep) {
            for (int i = 0; i < encoder->n_bind; ++i) {
                encoder->bind[i].buf = nil;
            }
            encoder->n_bind = 0;
            encoder->tg_size[0] = encoder->tg_size[1] = 0;
        }
        encoder->pso = pipeline.pipeline->obj;
        return;
    }
    [encoder->obj setComputePipelineState:pipeline.pipeline->obj];
    encoder->obj_pso = pipeline.pipeline->obj;
}

void ggml_metal_encoder_set_bytes(ggml_metal_encoder_t encoder, void * data, size_t size, int idx) {
    if (encoder->mega) {
        struct ggml_metal_mega_rec * r = encoder->mega;
        if (idx >= GGML_METAL_RECORDING_BINDS) { r->failed = true; return; }
        const size_t offs = (r->n_bytes + 15) & ~(size_t) 15;
        if (offs + size > r->cap_bytes) {
            while (offs + size > r->cap_bytes) r->cap_bytes *= 2;
            r->bytes = realloc(r->bytes, r->cap_bytes);
        }
        memcpy(r->bytes + offs, data, size);
        r->n_bytes = offs + size;
        r->cur.b[idx].buf = nil;
        r->cur.b[idx].bytes_off  = (int) offs;
        r->cur.b[idx].bytes_size = (int) size;
        if (idx >= r->cur.nb) r->cur.nb = idx + 1;
        if (!g_mega_pt) return;
    }
    if (encoder->rec) {
        // the bytes go to the arguments buffer of the recording and are bound like any buffer
        ggml_metal_recording_t rec = encoder->rec;
        const size_t offs = (rec->args_offs + 15) & ~((size_t) 15);
        if (idx >= GGML_METAL_RECORDING_BINDS || offs + size > rec->args_size) {
            rec->failed = true;
            return;
        }
        memcpy((char *) [rec->args contents] + offs, data, size);
        rec->args_offs = offs + size;
        encoder->bind[idx].buf  = rec->args;
        encoder->bind[idx].offs = offs;
        if (idx >= encoder->n_bind) {
            encoder->n_bind = idx + 1;
        }
        return;
    }
    if (idx < 16) encoder->kl_b[idx] = ggml_metal_kl_hash(data, size);
    [encoder->obj setBytes:data length:size atIndex:idx];
}

void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx) {
    if (encoder->mega) {
        struct ggml_metal_mega_rec * r = encoder->mega;
        if (idx >= GGML_METAL_RECORDING_BINDS) { r->failed = true; return; }
        r->cur.b[idx].buf  = buffer.metal;
        r->cur.b[idx].offs = buffer.offs;
        r->cur.b[idx].bytes_off = -1;
        if (idx >= r->cur.nb) r->cur.nb = idx + 1;
        if (!g_mega_pt) return;
    }
    if (encoder->rec) {
        if (idx >= GGML_METAL_RECORDING_BINDS) {
            encoder->rec->failed = true;
            return;
        }
        encoder->bind[idx].buf  = buffer.metal;
        encoder->bind[idx].offs = buffer.offs;
        if (idx >= encoder->n_bind) {
            encoder->n_bind = idx + 1;
        }
        return;
    }
    if (idx < 16) encoder->kl_b[idx] = [buffer.metal gpuAddress] + buffer.offs;
    [encoder->obj setBuffer:buffer.metal offset:buffer.offs atIndex:idx];
}

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx) {
    // ref: https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/setthreadgroupmemorylength(_:index:)
    GGML_ASSERT(size % 16 == 0);

    if (encoder->mega) {
        if (idx >= 2) { encoder->mega->failed = true; return; }
        encoder->mega->cur.tg[idx] = size;
        if (!g_mega_pt) return;
    }
    if (encoder->rec) {
        if (idx >= 2) {
            encoder->rec->failed = true;
            return;
        }
        encoder->tg_size[idx] = size;
        return;
    }
    if (idx < 2) encoder->tg_size[idx] = size;
    [encoder->obj setThreadgroupMemoryLength:size atIndex:idx];
}

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2) {
    if (encoder->mega) {
        struct ggml_metal_mega_rec * r = encoder->mega;
        if (r->cur.pl == NULL) { r->failed = true; return; }
        if (r->n == r->cap) {
            r->cap *= 2;
            r->d = realloc(r->d, r->cap*sizeof(*r->d));
        }
        struct ggml_metal_mega_disp * d = &r->d[r->n++];
        *d = r->cur;
        d->grid[0] = tg0; d->grid[1] = tg1; d->grid[2] = tg2;
        d->tptg[0] = tptg0; d->tptg[1] = tptg1; d->tptg[2] = tptg2;
        d->step = r->step;
        r->step_used = true;
        { uint64_t kb[16] = { 0 }; for (int j = 0; j < 16 && j < d->nb; ++j) kb[j] = d->b[j].bytes_off >= 0 ? ggml_metal_kl_hash(r->bytes + d->b[j].bytes_off, d->b[j].bytes_size) : (d->b[j].buf ? [d->b[j].buf gpuAddress] + d->b[j].offs : 0);
          ggml_metal_kernel_list(d->pl->obj, d->step, tg0, tg1, tg2, tptg0, tptg1, tptg2, d->tg[0], d->tg[1], kb); }
        if (!g_mega_pt) return;
    }
    if (encoder->rec) {
        ggml_metal_recording_t rec = encoder->rec;
        if (rec->failed || rec->n == rec->n_max || encoder->pso == nil) {
            rec->failed = true;
            return;
        }

        if (encoder->barrier || rec->n == 0) {
            rec->step_n_res[rec->n_starts] = 0;
            rec->starts[rec->n_starts++] = rec->n;
        }

        // the state of the encoder as it is now: what was bound for earlier kernels stays bound, as in Metal
        id<MTLIndirectComputeCommand> cmd = [rec->icb indirectComputeCommandAtIndex:rec->n++];

        [cmd setComputePipelineState:encoder->pso];
        for (int i = 0; i < encoder->n_bind; ++i) {
            if (encoder->bind[i].buf != nil) {
                id<MTLBuffer> buf = nil;
                size_t offs = 0;
                if (!ggml_metal_recording_window(rec, encoder->bind[i].buf, encoder->bind[i].offs, &buf, &offs)) {
                    rec->failed = true;
                    return;
                }
                [cmd setKernelBuffer:buf offset:offs atIndex:i];
                ggml_metal_recording_use(rec, buf);
                {
                    const int is = rec->n_starts - 1;
                    id<MTLBuffer> __unsafe_unretained * sr = rec->step_res + (size_t) is*GGML_METAL_RECORDING_STEP_RES;
                    int k = 0;
                    while (k < rec->step_n_res[is] && sr[k] != buf) {
                        k++;
                    }
                    if (k == rec->step_n_res[is]) {
                        if (k == GGML_METAL_RECORDING_STEP_RES) {
                            rec->failed = true;
                            return;
                        }
                        sr[k] = buf;
                        rec->step_n_res[is]++;
                    }
                }
            }
        }
        for (int i = 0; i < 2; ++i) {
            if (encoder->tg_size[i] > 0) {
                [cmd setThreadgroupMemoryLength:encoder->tg_size[i] atIndex:i];
            }
        }
        [cmd concurrentDispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];

        // the barrier of a command holds it until the commands before it are done
        if (encoder->barrier) {
            [cmd setBarrier];
        } else {
            [cmd clearBarrier];
        }
        encoder->barrier = false;
        return;
    }
    ggml_metal_kernel_list(encoder->obj_pso, atomic_load(&g_kl_step), tg0, tg1, tg2, tptg0, tptg1, tptg2, encoder->tg_size[0], encoder->tg_size[1], encoder->kl_b);
    if (encoder->tl_slot >= 0) {
        if (g_tl_ndisp[encoder->tl_slot]++ == 0) {
            memcpy(g_tl_lab[encoder->tl_slot], t_tl_label, sizeof(t_tl_label));
        }
    }
    [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];
}

static int g_bar_every = 0;

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder) {
    if (encoder->mega) {
        if (encoder->mega->step_used) {
            encoder->mega->step++;
            encoder->mega->step_used = false;
        }
        if (!g_mega_pt) return;
    }
    if (encoder->rec) {
        encoder->barrier = true;
        return;
    }
    // a serial step is a new encoder rather than a barrier in this one: measured on an M5 Ultra, a chain of small
    // kernels takes 2.2 us a step this way and 3.7 with a barrier, and Inkling 2 Mini has 1,140 steps a token.
    // GGML_METAL_BARRIER=scope goes back to the barrier
    static int mode = -1;
    if (mode < 0) {
        const char * v = getenv("GGML_METAL_BARRIER");
        mode = v && strcmp(v, "scope") == 0 ? 0 : 1;
        // GGML_METAL_BARRIER=none[,n]: profiling only, results are garbage: no step boundaries at all (or only every
        // n-th one), so the kernels of a token overlap; the GPU time left is the work without the serial waits
        if (v && strncmp(v, "none", 4) == 0) {
            mode = 2;
            g_bar_every = v[4] == 0x2c ? atoi(v + 5) : 0;
        }
    }
    if (mode == 2) {
        static atomic_int n_bar = 0;
        const int k = atomic_fetch_add(&n_bar, 1);
        if (g_bar_every <= 0 || k % g_bar_every != 0) {
            return;
        }
    }
    if (mode >= 1) {
        [encoder->obj endEncoding];
        [encoder->obj release];
        encoder->obj = ggml_metal_new_compute_encoder(encoder->cmd, encoder->concurrent, &encoder->tl_slot);
        encoder->tg_size[0] = encoder->tg_size[1] = 0;
        memset(encoder->kl_b, 0, sizeof(encoder->kl_b));
        atomic_fetch_add(&g_kl_step, 1);
        [encoder->obj retain];
        return;
    }
    [encoder->obj memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder) {
    if (encoder->mega) {
        @autoreleasepool {
            if (g_mega_pt) { [encoder->obj endEncoding]; return; }
            ggml_metal_mega_flush(encoder);
        }
        return;
    }
    if (encoder->rec) {
        return;
    }
    [encoder->obj endEncoding];
}

struct ggml_metal_device {
    id<MTLDevice> mtl_device;

    // a single global queue shared by all Metal backends
    // technically not needed for devices with unified memory, but enables discrete GPUs support
    // ref: https://github.com/ggml-org/llama.cpp/pull/15906
    id<MTLCommandQueue> mtl_queue;

    ggml_metal_rsets_t rsets;

    ggml_metal_library_t library;

    struct ggml_metal_device_props props;

    // shared fusion debugging context
    struct ggml_metal_fusion_info * finfo;

    // virtual address for GPU memory allocations
    atomic_uintptr_t addr_virt;
};

//
// MTLResidenceSet wrapper
//

struct ggml_metal_rsets {
    NSLock * lock;

    NSMutableArray * data;

    // number of seconds since the last graph computation
    // keep the residency sets wired for that amount of time to avoid being collected by the OS
    int keep_alive_s;
    int loops_per_s;
    int time_per_loop_ms;

    // background heartbeat thread to keep the residency sets alive
    atomic_bool d_stop;
    atomic_int  d_loop;

    dispatch_group_t d_group;
};

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
static void ggml_metal_dummy_work(ggml_metal_device_t dev) {
    if (dev->mtl_queue == nil) {
        return;
    }

    @autoreleasepool {
        // perform a minimal dummy operation on the GPU
        id<MTLBuffer> buf = [dev->mtl_device newBufferWithLength:1 options:MTLResourceStorageModePrivate];
        id<MTLCommandBuffer> cmd_buf = [dev->mtl_queue commandBuffer];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:buf range:NSMakeRange(0, 1) value:0];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [buf release];
    }
}
#endif

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev) {
    ggml_metal_rsets_t res = calloc(1, sizeof(struct ggml_metal_rsets));

    res->lock = [[NSLock alloc] init];
    res->data = [[NSMutableArray alloc] init];

    // by default keep the memory wired for 3 minutes
    res->keep_alive_s = 3*60;

    const char * GGML_METAL_RESIDENCY_KEEP_ALIVE_S = getenv("GGML_METAL_RESIDENCY_KEEP_ALIVE_S");
    if (GGML_METAL_RESIDENCY_KEEP_ALIVE_S) {
        res->keep_alive_s = atoi(GGML_METAL_RESIDENCY_KEEP_ALIVE_S);
    }

    if (res->keep_alive_s <= 0) {
        res->keep_alive_s = 3*60;
    }

    res->time_per_loop_ms = 5;
    res->loops_per_s = 1000/res->time_per_loop_ms;

    GGML_LOG_INFO("%s: creating a residency set collection (keep_alive = %d s)\n", __func__, res->keep_alive_s);

    atomic_store_explicit(&res->d_stop, false, memory_order_relaxed);
    atomic_store_explicit(&res->d_loop, res->loops_per_s*res->keep_alive_s, memory_order_relaxed);

    res->d_group = dispatch_group_create();

    // start a background thread that periodically requests residency for all the currently active sets in the collection
    // the requests stop after a certain amount of time (keep_alive_s) of inactivity
    dispatch_queue_t d_queue = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_group_async(res->d_group, d_queue, ^{
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
              while (!atomic_load_explicit(&res->d_stop, memory_order_relaxed)) {
                  if (atomic_load_explicit(&res->d_loop, memory_order_relaxed) > 0) {
                      [res->lock lock];

                      for (int i = 0; i < (int) res->data.count; ++i) {
                          [res->data[i] requestResidency];
                      }

                      atomic_fetch_sub_explicit(&res->d_loop, 1, memory_order_relaxed);

                      [res->lock unlock];
                  }

                  usleep(res->time_per_loop_ms * 1000);
              }
        }
#endif
    });

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        // workaround for residency set memory not being released if no GPU operation occurs
        // https://developer.apple.com/forums/thread/839089
        // https://github.com/ggml-org/llama.cpp/issues/25937
        ggml_metal_dummy_work(dev);
    }
#endif

    return res;
}

void ggml_metal_rsets_free(ggml_metal_rsets_t rsets) {
    if (rsets == NULL) {
        return;
    }

    // note: if you hit this assert, most likely you haven't deallocated all Metal resources before exiting
    GGML_ASSERT([rsets->data count] == 0);

    atomic_store_explicit(&rsets->d_stop, true, memory_order_relaxed);

    dispatch_group_wait(rsets->d_group, DISPATCH_TIME_FOREVER);
    dispatch_release(rsets->d_group);

    [rsets->data release];
    [rsets->lock release];

    free(rsets);
}

static const struct {
    const char *              name;
    const char *              token;
    enum ggml_metal_device_id id;
} k_metal_devices[] = {
#define DEV(name, id) { name, #id, id }
    DEV("M1",       GGML_METAL_DEVICE_M1),
    DEV("M1 Pro",   GGML_METAL_DEVICE_M1_PRO),
    DEV("M1 Max",   GGML_METAL_DEVICE_M1_MAX),
    DEV("M1 Ultra", GGML_METAL_DEVICE_M1_ULTRA),
    DEV("M2",       GGML_METAL_DEVICE_M2),
    DEV("M2 Pro",   GGML_METAL_DEVICE_M2_PRO),
    DEV("M2 Max",   GGML_METAL_DEVICE_M2_MAX),
    DEV("M2 Ultra", GGML_METAL_DEVICE_M2_ULTRA),
    DEV("M3",       GGML_METAL_DEVICE_M3),
    DEV("M3 Pro",   GGML_METAL_DEVICE_M3_PRO),
    DEV("M3 Max",   GGML_METAL_DEVICE_M3_MAX),
    DEV("M3 Ultra", GGML_METAL_DEVICE_M3_ULTRA),
    DEV("M4",       GGML_METAL_DEVICE_M4),
    DEV("M4 Pro",   GGML_METAL_DEVICE_M4_PRO),
    DEV("M4 Max",   GGML_METAL_DEVICE_M4_MAX),
    DEV("M5",       GGML_METAL_DEVICE_M5),
    DEV("M5 Pro",   GGML_METAL_DEVICE_M5_PRO),
    DEV("M5 Max",   GGML_METAL_DEVICE_M5_MAX),
    DEV("M5 Ultra", GGML_METAL_DEVICE_M5_ULTRA),
    DEV("A18 Pro",  GGML_METAL_DEVICE_A18_PRO),
#undef DEV
};

static enum ggml_metal_device_id ggml_metal_device_id_parse(const char * name) {
    if (!name) {
        return GGML_METAL_DEVICE_GENERIC;
    }

    static const char prefix[] = "Apple ";
    if (strncmp(name, prefix, sizeof(prefix) - 1) != 0) {
        return GGML_METAL_DEVICE_GENERIC;
    }
    const char * suffix = name + sizeof(prefix) - 1;

    for (size_t i = 0; i < sizeof(k_metal_devices)/sizeof(k_metal_devices[0]); ++i) {
        if (strcmp(suffix, k_metal_devices[i].name) == 0) {
            return k_metal_devices[i].id;
        }
    }
    return GGML_METAL_DEVICE_GENERIC;
}

const char * ggml_metal_device_id_token(enum ggml_metal_device_id id) {
    for (size_t i = 0; i < sizeof(k_metal_devices)/sizeof(k_metal_devices[0]); ++i) {
        if (k_metal_devices[i].id == id) {
            return k_metal_devices[i].token;
        }
    }
    return "GGML_METAL_DEVICE_GENERIC";
}

ggml_metal_device_t ggml_metal_device_init(int device, int n_devices) {
    ggml_metal_device_t dev = calloc(1, sizeof(struct ggml_metal_device));

    assert(dev != NULL);

    @autoreleasepool {
        if (dev->mtl_device == nil) {
            dev->mtl_device = MTLCreateSystemDefaultDevice();

            if (dev->mtl_device) {
                dev->mtl_queue = [dev->mtl_device newCommandQueue];
                if (dev->mtl_queue == nil) {
                    GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
                }

                dev->addr_virt = 0x000000400ULL;

                dev->props.device = device;

                // the Metal backend uses the system default device as the single physical device;
                // additional (virtual) devices are emulated on top of it via GGML_METAL_DEVICES
                dev->props.device_phys = 0;
                dev->props.device_virt = device;

                dev->props.has_simdgroup_reduction  = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
                dev->props.has_simdgroup_reduction |= [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];

                dev->props.has_simdgroup_mm = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
                dev->props.has_unified_memory = dev->mtl_device.hasUnifiedMemory;

                dev->props.has_bfloat  = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
                dev->props.has_bfloat |= [dev->mtl_device supportsFamily:MTLGPUFamilyApple6];
                if (getenv("GGML_METAL_BF16_DISABLE") != NULL) {
                    dev->props.has_bfloat = false;
                }

                dev->props.has_tensor = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal4_GGML];
                if (getenv("GGML_METAL_TENSOR_DISABLE") != NULL) {
                    dev->props.has_tensor = false;
                }

                // note: disable the tensor API by default for old chips because with the current implementation it is not useful
                // - M2 Ultra:   ~5% slower
                // - M4, M4 Max: no significant difference
                //
                // TODO: try to update the tensor API kernels to at least match the simdgroup performance
                if (getenv("GGML_METAL_TENSOR_ENABLE") == NULL &&
                    ![[dev->mtl_device name] containsString:@"M5"] &&
                    ![[dev->mtl_device name] containsString:@"M6"] &&
                    ![[dev->mtl_device name] containsString:@"A19"] &&
                    ![[dev->mtl_device name] containsString:@"A20"]) {
                    GGML_LOG_INFO("%s: tensor API disabled for pre-M5 and pre-A19 devices\n", __func__);
                    dev->props.has_tensor = false;
                }

                // double-check that the tensor API compiles
                if (dev->props.has_tensor) {
                    const char * src_tensor_f16 = "\n"
                        "#include <metal_stdlib> \n"
                        "#include <metal_tensor> \n"
                        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                        " \n"
                        "using namespace metal; \n"
                        "using namespace mpp::tensor_ops; \n"
                        " \n"
                        "kernel void dummy_kernel( \n"
                        "    tensor<device  half, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                        "    tensor<device  half, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                        "    device float * C [[buffer(2)]], \n"
                        "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                        "{ \n"
                        "    auto tA = A.slice(0, (int)tgid.y); \n"
                        "    auto tB = B.slice((int)tgid.x, 0); \n"
                        " \n"
                        "    matmul2d< \n"
                        "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                        "        execution_simdgroups<4>> mm; \n"
                        " \n"
                        "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                        " \n"
                        "    auto sA = tA.slice(0, 0); \n"
                        "    auto sB = tB.slice(0, 0); \n"
                        "    mm.run(sB, sA, cT); \n"
                        " \n"
                        "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16)); \n"
                        " \n"
                        "    cT.store(tC); \n"
                        "}";

                    GGML_LOG_INFO("%s: testing tensor API for f16 support\n", __func__);
                    ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_f16, false);
                    if (lib == NULL) {
                        GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                        dev->props.has_tensor = false;
                    } else {
                        struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                        if (!ppl.pipeline) {
                            GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                            dev->props.has_tensor = false;
                        }

                        ggml_metal_library_free(lib);
                    }
                }

                // try to compile a dummy kernel to determine if the tensor API is supported for bfloat
                if (dev->props.has_tensor && dev->props.has_bfloat) {
                    const char * src_tensor_bf16 = "\n"
                        "#include <metal_stdlib> \n"
                        "#include <metal_tensor> \n"
                        "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                        " \n"
                        "using namespace metal; \n"
                        "using namespace mpp::tensor_ops; \n"
                        " \n"
                        "kernel void dummy_kernel( \n"
                        "    tensor<device bfloat, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                        "    tensor<device bfloat, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                        "    device float * C [[buffer(2)]], \n"
                        "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                        "{ \n"
                        "    auto tA = A.slice(0, (int)tgid.y); \n"
                        "    auto tB = B.slice((int)tgid.x, 0); \n"
                        " \n"
                        "    matmul2d< \n"
                        "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                        "        execution_simdgroups<4>> mm; \n"
                        " \n"
                        "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                        " \n"
                        "    auto sA = tA.slice(0, 0); \n"
                        "    auto sB = tB.slice(0, 0); \n"
                        "    mm.run(sB, sA, cT); \n"
                        " \n"
                        "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(16, 16)); \n"
                        " \n"
                        "    cT.store(tC); \n"
                        "}";

                    GGML_LOG_INFO("%s: testing tensor API for bfloat support\n", __func__);
                    ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_bf16, false);
                    if (lib == NULL) {
                        GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                        dev->props.has_bfloat = false;
                    } else {
                        struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                        if (!ppl.pipeline) {
                            GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                            dev->props.has_bfloat = false;
                        }

                        ggml_metal_library_free(lib);
                    }
                }

                dev->props.use_residency_sets = true;
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
                dev->props.use_residency_sets = getenv("GGML_METAL_NO_RESIDENCY") == nil;
#endif

                dev->props.use_shared_buffers = dev->props.has_unified_memory;
#if TARGET_OS_OSX && TARGET_CPU_X86_64
                // In case of eGPU, shared memory may be preferable.
                dev->props.use_shared_buffers |= [dev->mtl_device location] == MTLDeviceLocationExternal;
#endif
                if (getenv("GGML_METAL_SHARED_BUFFERS_DISABLE") != NULL) {
                    dev->props.use_shared_buffers = false;
                }
                if (getenv("GGML_METAL_SHARED_BUFFERS_ENABLE") != NULL) {
                    dev->props.use_shared_buffers = true;
                }

                dev->props.device_id = ggml_metal_device_id_parse([[dev->mtl_device name] UTF8String]);

                dev->props.op_offload_min_batch_size  = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

                dev->props.max_buffer_size            = dev->mtl_device.maxBufferLength;
                dev->props.max_theadgroup_memory_size = dev->mtl_device.maxThreadgroupMemoryLength;
                if (@available(macOS 10.12, iOS 16.0, *)) {
                    dev->props.max_working_set_size   = dev->mtl_device.recommendedMaxWorkingSetSize;
                } else {
                    dev->props.max_working_set_size   = dev->mtl_device.maxBufferLength;
                }

                {
                    const char * val = getenv("GGML_METAL_FUSION_DEBUG");
                    dev->finfo = ggml_metal_fusion_info_init(
                            getenv("GGML_METAL_FUSION_DISABLE") == nil,
                            val ? atoi(val) : 0);
                }

                snprintf(dev->props.name, sizeof(dev->props.name), "%s%d", "MTL", device);
                const char * gpu_name = [[dev->mtl_device name] UTF8String];
                if (n_devices > 1) {
                    snprintf(dev->props.desc, sizeof(dev->props.desc), "%s (dev p%d/v%d)",
                             gpu_name, dev->props.device_phys, dev->props.device_virt);
                } else {
                    snprintf(dev->props.desc, sizeof(dev->props.desc), "%s", gpu_name);
                }

                dev->library = ggml_metal_library_init(dev);
                if (!dev->library) {
                    GGML_LOG_ERROR("%s: error: failed to create library\n", __func__);
                }

                if (dev->props.use_residency_sets) {
                    dev->rsets = ggml_metal_rsets_init(dev);
                } else {
                    dev->rsets = nil;
                }

                // print MTL GPU family:
                GGML_LOG_INFO("%s: GPU name:   %s (%s)\n", __func__, dev->props.name, dev->props.desc);

                // determine max supported GPU family
                // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
                // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
                {
                    for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
                        if ([dev->mtl_device supportsFamily:i]) {
                            dev->props.gpu_family = i - (int) MTLGPUFamilyApple1 + 1;
                            GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __func__, dev->props.gpu_family, i);
                            break;
                        }
                    }

#if TARGET_CPU_X86_64
                    for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
                        if ([dev->mtl_device supportsFamily:i]) {
                            GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __func__, i - (int) MTLGPUFamilyCommon1 + 1, i);
                            break;
                        }
                    }
#endif

                    for (int i = MTLGPUFamilyMetal3_GGML + 5; i >= MTLGPUFamilyMetal3_GGML; --i) {
                        if ([dev->mtl_device supportsFamily:i]) {
                            GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __func__, i - (int) MTLGPUFamilyMetal3_GGML + 3, i);
                            break;
                        }
                    }
                }

                GGML_LOG_INFO("%s: simdgroup reduction   = %s\n", __func__, dev->props.has_simdgroup_reduction ? "true" : "false");
                GGML_LOG_INFO("%s: simdgroup matrix mul. = %s\n", __func__, dev->props.has_simdgroup_mm        ? "true" : "false");
                GGML_LOG_INFO("%s: has unified memory    = %s\n", __func__, dev->props.has_unified_memory      ? "true" : "false");
                GGML_LOG_INFO("%s: has bfloat            = %s\n", __func__, dev->props.has_bfloat              ? "true" : "false");
                GGML_LOG_INFO("%s: has tensor            = %s\n", __func__, dev->props.has_tensor              ? "true" : "false");
                GGML_LOG_INFO("%s: use residency sets    = %s\n", __func__, dev->props.use_residency_sets      ? "true" : "false");
                GGML_LOG_INFO("%s: use shared buffers    = %s\n", __func__, dev->props.use_shared_buffers      ? "true" : "false");

#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
                if (@available(macOS 10.12, iOS 16.0, *)) {
                    GGML_LOG_INFO("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, dev->props.max_working_set_size / 1e6);
                }
#endif
            }
        }
    }

    return dev;
}

void ggml_metal_device_free(ggml_metal_device_t dev) {
    assert(dev != NULL);

    @autoreleasepool {
        ggml_metal_fusion_info_free(dev->finfo);

        ggml_metal_rsets_free(dev->rsets);

        ggml_metal_library_free(dev->library);
        dev->library = NULL;

        if (dev->mtl_queue) {
            [dev->mtl_queue release];
            dev->mtl_queue = nil;
        }

        if (dev->mtl_device) {
            [dev->mtl_device release];
            dev->mtl_device = nil;
        }
    }

    free(dev);
}

void * ggml_metal_device_get_obj(ggml_metal_device_t dev) {
    return dev->mtl_device;
}

void * ggml_metal_device_get_queue(ggml_metal_device_t dev) {
    return dev->mtl_queue;
}

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev) {
    return dev->library;
}

void ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data addObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_rm(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data removeObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev) {
    if (dev->rsets == NULL) {
        return;
    }

    atomic_store_explicit(&dev->rsets->d_loop, dev->rsets->loops_per_s*dev->rsets->keep_alive_s, memory_order_relaxed);
}

struct ggml_metal_event {
    void * obj; // id<MTLSharedEvent>

    atomic_int value;
};

void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeSignalEvent:event value:atomic_fetch_add_explicit(&ev->value, 1, memory_order_relaxed) + 1];
}

void ggml_metal_event_encode_wait(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLSharedEvent> event = (id<MTLSharedEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeWaitForEvent:event value:atomic_load_explicit(&ev->value, memory_order_relaxed)];
}

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev) {
    id<MTLSharedEvent> event = [dev->mtl_device newSharedEvent];

    ggml_metal_event_t ev = calloc(1, sizeof(struct ggml_metal_event));

    ev->obj = (__bridge void *)event;
    ev->value = 0;

    return ev;
}

void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLSharedEvent> event = ev->obj;
        [event release];

        free(ev);

        GGML_UNUSED(dev);
    }
}

void ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    id<MTLSharedEvent> event = ev->obj;
    const bool res = [event waitUntilSignaledValue:atomic_load_explicit(&ev->value, memory_order_relaxed) timeoutMS:60000];
    if (!res) {
        GGML_ABORT("%s: failed to wait for event\n", __func__);
    }

    GGML_UNUSED(dev);
}

void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total) {
    if (@available(macOS 10.12, iOS 16.0, *)) {
        *total     = dev->mtl_device.recommendedMaxWorkingSetSize;
        size_t cur = dev->mtl_device.currentAllocatedSize;
        // it's possible to allocate more than `recommendedMaxWorkingSetSize`
        *free      = *total > cur ? *total - cur : 0;
    } else {
        *free = 0;
        *total = 0;
    }
}

static bool ggml_metal_supports_mul_mat_src1_prec(const struct ggml_tensor * op, bool has_tensor) {
    const enum ggml_prec prec = ggml_prec_get_src(op, 1);

    if (prec == GGML_PREC_F32) {
        return false;
    }
    if (prec == GGML_PREC_BF16) {
        return has_tensor && op->op == GGML_OP_MUL_MAT && op->src[0]->type != GGML_TYPE_F32;
    }
    return true;
}

static bool ggml_metal_supports_mul_mat_op(
        bool has_simdgroup_reduction,
        const struct ggml_tensor * op,
        bool src0_f16_has_mv,
        bool mm_path) {
    if (!has_simdgroup_reduction ||
        op->src[0]->type == GGML_TYPE_F8_E4M3 ||
        op->src[0]->type == GGML_TYPE_TQ1_0) {
        return false;
    }

    if (op->src[1]->type != GGML_TYPE_F16) {
        return true;
    }

    if (op->src[0]->type == GGML_TYPE_BF16) {
        return false;
    }

    if (src0_f16_has_mv && op->src[0]->type == GGML_TYPE_F16) {
        return true;
    }

    return mm_path;
}

bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    const bool has_simdgroup_mm        = dev->props.has_simdgroup_mm;
    const bool has_simdgroup_reduction = dev->props.has_simdgroup_reduction;
    const bool has_bfloat              = dev->props.has_bfloat;

    if (!has_bfloat) {
        if (op->type == GGML_TYPE_BF16) {
            return false;
        }

        for (size_t i = 0, n = 3; i < n; ++i) {
            if (op->src[i] != NULL && op->src[i]->type == GGML_TYPE_BF16) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_SCALE:
            return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16);
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                case GGML_UNARY_OP_XIELU:
                    return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
                default:
                    return false;
            }
        case GGML_OP_SILU_BACK:
            return (op->src[0]->type == GGML_TYPE_F32) &&
                (op->src[1]->type == GGML_TYPE_F32) &&
                (op->type == GGML_TYPE_F32) &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) &&
                ggml_is_contiguous(op) &&
                ggml_are_same_shape(op->src[0], op->src[1]);
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    return ggml_is_contiguous_1(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
               default:
                    return false;
            }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            return true;
        case GGML_OP_CONCAT:
            {
                const enum ggml_type src0_type = op->src[0]->type;
                const enum ggml_type src1_type = op->src[1]->type;
                if (src0_type != src1_type || src0_type != op->type) {
                    return false;
                }
                switch (src0_type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_I8:
                    case GGML_TYPE_I16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_I64:
                        return true;
                    case GGML_TYPE_BF16:
                        return has_bfloat;
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            }
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ADD_ID:
            return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) && (op->src[0]->type == op->src[1]->type);
        case GGML_OP_ACC:
            return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_REPEAT:
        case GGML_OP_CONV_TRANSPOSE_1D:
            return true;
        case GGML_OP_CONV_TRANSPOSE_2D:
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
                (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_COL2IM_1D:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16) &&
                op->type == op->src[0]->type &&
                ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op);
        case GGML_OP_CONV_3D:
            return ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_SUM:
            return has_simdgroup_reduction && ggml_is_contiguous(op->src[0]);
        case GGML_OP_TRI:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_SUM_ROWS:
        case GGML_OP_CUMSUM:
        case GGML_OP_MEAN:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_L2_NORM:
            return has_simdgroup_reduction && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32 &&
                ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_COUNT_EQUAL:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_I32 &&
                op->src[1]->type == GGML_TYPE_I32 &&
                op->type == GGML_TYPE_I64;
        case GGML_OP_ARGMAX:
            return has_simdgroup_reduction;
        case GGML_OP_NORM:
            return has_simdgroup_reduction && (ggml_is_contiguous_rows(op->src[0]));
        case GGML_OP_RMS_NORM:
            return has_simdgroup_reduction && op->src[0]->type == GGML_TYPE_F32 && (ggml_is_contiguous_rows(op->src[0]));
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return ggml_rope_get_truncate(op) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
        case GGML_OP_IM2COL:
            return ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
        case GGML_OP_CONV_2D_DW:
            return op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
        case GGML_OP_UPSCALE:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_1D:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_2D:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_PAD:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_TIMESTEP_EMBEDDING:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_LEAKY_RELU:
            return op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16;
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_ARANGE:
            return true;
        case GGML_OP_ROLL:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_FLASH_ATTN_EXT:
            // K/V row lists (ggml_flash_attn_ext_rows): one list of cache rows per q slice
            if (op->src[5] != NULL) {
                const struct ggml_tensor * kv_rows = op->src[5];

                if (kv_rows->type != GGML_TYPE_I32 ||
                    !ggml_is_contiguous(kv_rows) ||
                    kv_rows->ne[2] != 1 || kv_rows->ne[3] != 1 ||
                    kv_rows->ne[1] != op->src[0]->ne[3] ||
                    !op->src[3] || op->src[1]->ne[3] != 1 || op->src[2]->ne[3] != 1) {
                    return false;
                }

                // the non-vec kernels attend over a contiguous F16 copy of the referenced rows:
                // - F16 K/V is gathered directly
                // - quantized K/V is first dequantized with kernel_flash_attn_ext_kv_*_f16
                // - other types keep attending to the whole cache (dense mask)
                switch (op->src[1]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        break;
                    default:
                        return false;
                }

                // the row gather copies F16 values in groups of 4
                if (op->src[2]->ne[0] % 4 != 0) {
                    return false;
                }
            }
            // for new head sizes, add checks here
            if (op->src[0]->ne[0] != 32 &&
                op->src[0]->ne[0] != 40 &&
                op->src[0]->ne[0] != 48 &&
                op->src[0]->ne[0] != 64 &&
                op->src[0]->ne[0] != 72 &&
                op->src[0]->ne[0] != 80 &&
                op->src[0]->ne[0] != 96 &&
                op->src[0]->ne[0] != 112 &&
                op->src[0]->ne[0] != 128 &&
                op->src[0]->ne[0] != 192 &&
                op->src[0]->ne[0] != 256 &&
                op->src[0]->ne[0] != 320 &&
                op->src[0]->ne[0] != 512 &&
                op->src[0]->ne[0] != 576) {
                return false;
            }
            // the kernels exist for K == V and for these K > V pairs only
            if (op->src[1]->ne[0] != op->src[2]->ne[0] &&
                !(op->src[1]->ne[0] ==  96 && op->src[2]->ne[0] ==  64) &&
                !(op->src[1]->ne[0] == 128 && op->src[2]->ne[0] ==  96) &&
                !(op->src[1]->ne[0] == 192 && op->src[2]->ne[0] == 128) &&
                !(op->src[1]->ne[0] == 320 && op->src[2]->ne[0] == 256) &&
                !(op->src[1]->ne[0] == 576 && op->src[2]->ne[0] == 512)) {
                return false;
            }
            if (op->src[1]->type != op->src[2]->type) {
                return false;
            }
            switch (op->src[1]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                    break;
                case GGML_TYPE_BF16:
                    if (!has_bfloat) {
                        return false;
                    }
                    break;
                default:
                    return false;
            }
            return has_simdgroup_mm; // TODO: over-restricted for vec-kernels
        case GGML_OP_FLASH_ATTN_EXT_BANDED:
            {
                // no vec kernel: the half8x8 or tensor kernel runs for any number of queries
                const struct ggml_tensor * mask  = op->src[3];
                const struct ggml_tensor * sinks = op->src[4];
                const struct ggml_tensor * rel   = op->src[6];

                if (op->src[0]->ne[0] != 64 && op->src[0]->ne[0] != 128) {
                    return false;
                }
                if (op->src[2]->ne[0] != op->src[1]->ne[0]) {
                    return false;
                }
                if (op->src[1]->type != op->src[2]->type) {
                    return false;
                }
                switch (op->src[1]->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                        break;
                    case GGML_TYPE_BF16:
                        if (!has_bfloat) {
                            return false;
                        }
                        break;
                    default:
                        return false;
                }
                if (rel->type != GGML_TYPE_F32 && rel->type != GGML_TYPE_F16) {
                    return false;
                }
                if (rel->nb[0] != ggml_type_size(rel->type)) {
                    return false;
                }
                if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[1] < op->src[0]->ne[1])) {
                    return false;
                }
                if (sinks && (sinks->type != GGML_TYPE_F32 || sinks->nb[0] != sizeof(float) ||
                        (sinks->ne[1] != 1 && sinks->ne[1] != op->src[0]->ne[1]) ||
                        (sinks->ne[2] != 1 && sinks->ne[2] != op->src[0]->ne[3]))) {
                    return false;
                }
                return has_simdgroup_mm;
            }
        case GGML_OP_LIGHTNING_INDEXER:
            if (op->src[0]->ne[0] != OP_LIGHTNING_INDEXER_DK) {
                return false;
            }
            if (!has_simdgroup_mm ||
                op->src[0]->type != GGML_TYPE_F32 ||
                op->src[2]->type != GGML_TYPE_F32 ||
                op->src[3]->type != GGML_TYPE_F16 ||
                op->type         != GGML_TYPE_F32 ||
                !ggml_is_contiguous_rows(op->src[0]) ||
                !ggml_is_contiguous_rows(op->src[1]) ||
                !ggml_is_contiguous_rows(op->src[2]) ||
                !ggml_is_contiguous_rows(op->src[3])) {
                return false;
            }
            switch (op->src[1]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    return true;
                case GGML_TYPE_BF16:
                    return has_bfloat;
                default:
                    return false;
            }
        case GGML_OP_DSV4_HC_COMB:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                op->type         == GGML_TYPE_F32 &&
                op->src[0]->ne[0] == 24 &&
                op->src[1]->ne[0] >= 3 &&
                op->src[2]->ne[0] == 24 &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]) &&
                ggml_is_contiguous_rows(op->src[2]);
        case GGML_OP_DSV4_HC_PRE:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type         == GGML_TYPE_F32 &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]);
        case GGML_OP_DSV4_HC_POST:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_F32 &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 &&
                (op->src[3] == NULL || op->src[3]->type == GGML_TYPE_F32) &&
                op->type         == GGML_TYPE_F32 &&
                op->src[1]->ne[1] == 4 &&
                op->src[2]->ne[0] == 4 &&
                (op->src[3] == NULL || (op->src[3]->ne[0] == 4 && op->src[3]->ne[1] == 4)) &&
                ggml_is_contiguous_rows(op->src[0]) &&
                ggml_is_contiguous_rows(op->src[1]) &&
                ggml_is_contiguous_rows(op->src[2]) &&
                (op->src[3] == NULL || ggml_is_contiguous_rows(op->src[3]));
        case GGML_OP_TOP_K_MASK:
            return (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32) &&
                op->src[0]->type == op->type &&
                op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) &&
                op->nb[0] == ggml_type_size(op->type) &&
                op->src[1]->type == GGML_TYPE_I32 &&
                (op->src[2] == NULL || op->src[2]->type == GGML_TYPE_F32) &&
                (op->src[3] == NULL || op->src[3]->type == GGML_TYPE_I32) &&
                (op->src[4] == NULL || op->src[4]->type == GGML_TYPE_I32);
        case GGML_OP_SSM_SCAN:
            return has_simdgroup_reduction;
        case GGML_OP_SSM_CONV:
            return has_simdgroup_reduction;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return has_simdgroup_reduction && op->src[2]->ne[0] % 32 == 0;
        case GGML_OP_SOLVE_TRI:
            return has_simdgroup_reduction && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_MUL_MAT:
            if (!ggml_metal_supports_mul_mat_src1_prec(op, dev->props.has_tensor)) {
                return false;
            }
            // the FWHT kernels read an F16 source directly; every other F16 src1 path
            // still goes through ggml_metal_supports_mul_mat_op
            if (op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F16 &&
                ggml_metal_op_mul_mat_use_fwht(op, dev->props.max_theadgroup_memory_size)) {
                return has_simdgroup_reduction;
            }
            return ggml_metal_supports_mul_mat_op(
                    has_simdgroup_reduction, op, true,
                    ggml_metal_op_mul_mat_use_mm(op, has_simdgroup_mm));
        case GGML_OP_MUL_MAT_ID:
            if (!ggml_metal_supports_mul_mat_src1_prec(op, dev->props.has_tensor)) {
                return false;
            }
            return ggml_metal_supports_mul_mat_op(
                    has_simdgroup_reduction, op, false,
                    ggml_metal_op_mul_mat_id_use_mm(op, has_simdgroup_mm));
        case GGML_OP_SET:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        switch (op->type) {
                           case GGML_TYPE_F32:
                           case GGML_TYPE_F16:
                           case GGML_TYPE_BF16:
                           case GGML_TYPE_Q8_0:
                           case GGML_TYPE_Q1_0:
                           case GGML_TYPE_Q2_0:
                           case GGML_TYPE_Q4_0:
                           case GGML_TYPE_Q4_1:
                           case GGML_TYPE_Q5_0:
                           case GGML_TYPE_Q5_1:
                           case GGML_TYPE_IQ4_NL:
                           case GGML_TYPE_TQ2_0:
                           case GGML_TYPE_I32:
                                return true;
                           default:
                                return false;
                        }
                    case GGML_TYPE_F16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_BF16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_BF16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_TQ2_0:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_I32:
                        return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32;
                    default:
                        return false;
                };
            }
        case GGML_OP_GET_ROWS:
            return op->src[0]->type != GGML_TYPE_TQ1_0 &&
                   op->src[0]->type != GGML_TYPE_F8_E4M3;
        case GGML_OP_SET_ROWS:
            {
                if (op->src[0]->type == GGML_TYPE_F16) {
                    return op->type == GGML_TYPE_F16;
                }

                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }

                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_TQ2_0:
                        return true;
                    default:
                        return false;
                };
            }
        case GGML_OP_DIAG:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return has_simdgroup_reduction;
        default:
            return false;
    }
}

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev) {
    return &dev->props;
}

static void ggml_metal_device_disable_tensor(ggml_metal_device_t dev) {
    dev->props.has_tensor = false;
}

struct ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev) {
    return dev->finfo;
}

//
// device buffers
//

// max memory buffers that can be mapped to the device
#define GGML_METAL_MAX_BUFFERS 64

struct ggml_metal_buffer_wrapper {
    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_metal_buffer {
    void * all_data;
    size_t all_size;

    // if false, the Metal buffer data is allocated in private GPU memory and is not shared with the host
    bool is_shared;
    bool owned;

    // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
    int n_buffers;
    struct ggml_metal_buffer_wrapper buffers[GGML_METAL_MAX_BUFFERS];

    bool use_residency_sets;

    // optional MTLResidencySet
    // note: cannot use explicitly "id<MTLResidencySet>" here because it is not available on certain OSes
    id rset;

    // pointers to global device
    ggml_metal_device_t dev;
};

static void ggml_metal_log_allocated_size(id<MTLDevice> device, size_t size_aligned) {
#ifndef GGML_METAL_NDEBUG
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        GGML_LOG_DEBUG("%s: allocated buffer, size = %8.2f MiB, (%8.2f / %8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0,
                device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (device.currentAllocatedSize > device.recommendedMaxWorkingSetSize) {
            GGML_LOG_WARN("%s: warning: current allocated size is greater than the recommended max working set size\n", __func__);
        }
    } else {
        GGML_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, (%8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0);
    }
#endif
#endif
    GGML_UNUSED(device);
    GGML_UNUSED(size_aligned);
}

// rset init
static bool ggml_metal_buffer_rset_init(ggml_metal_buffer_t buf) {
    buf->rset = nil;

    if (!buf->use_residency_sets) {
        return true;
    }

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ggml_metal";
        desc.initialCapacity = buf->n_buffers;

        NSError * error;
        buf->rset = [buf->dev->mtl_device newResidencySetWithDescriptor:desc error:&error];
        if (error) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            [desc release];
            return false;
        }

        [desc release];

        for (int i = 0; i < buf->n_buffers; i++) {
            [buf->rset addAllocation:buf->buffers[i].metal];
        }

        [buf->rset commit];
        [buf->rset requestResidency];

        return true;
    }
#endif

    return true;
}

// rset free
static void ggml_metal_buffer_rset_free(ggml_metal_buffer_t buf) {
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        if (buf->rset) {
            [buf->rset endResidency];
            [buf->rset removeAllAllocations];
            [buf->rset commit];
            [buf->rset release];
        }
    }
#else
    GGML_UNUSED(buf);
#endif
}

// the most bytes of host memory to put in one Metal buffer. the device takes far more, but an indirect command
// keeps a buffer offset right only below 2 GiB (see ggml_metal_recording), so a graph can only be executed from a
// recording without detours if every tensor is at less than that in its buffer. memory beyond it gets several
// buffers, each a view of 2 GiB, as memory beyond the limit of the device always did.
// GGML_METAL_MAX_BUFFER_GIB overrides the 2 (0: the limit of the device only)
static size_t ggml_metal_buffer_view_max(const struct ggml_metal_device_props * props_dev) {
    static int64_t gib = -1;
    if (gib < 0) {
        gib = getenv("GGML_METAL_MAX_BUFFER_GIB") ? atoi(getenv("GGML_METAL_MAX_BUFFER_GIB")) : 2;
    }
    const size_t cap = gib > 0 ? (size_t) gib << 30 : props_dev->max_buffer_size;
    return cap < props_dev->max_buffer_size ? cap : props_dev->max_buffer_size;
}

static void * ggml_metal_host_malloc(size_t n) {
    void * data = NULL;

    kern_return_t err = vm_allocate((vm_map_t) mach_task_self(), (void *) &data, n, VM_FLAGS_ANYWHERE);
    if (err != KERN_SUCCESS) {
        GGML_LOG_ERROR("%s: error: vm_allocate failed\n", __func__);
        return NULL;
    }

    return data;
}

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    shared = shared && props_dev->use_shared_buffers;

    // allocate shared buffer if the device supports it and it is required by the buffer type
    if (shared) {
        res->all_data = ggml_metal_host_malloc(size_aligned);
        res->is_shared = true;
    } else {
        // use virtual address
        res->all_data = (void *) atomic_fetch_add_explicit(&dev->addr_virt, size_aligned, memory_order_relaxed);
        res->is_shared = false;
    }
    res->all_size = size_aligned;

    res->owned = true;

    res->n_buffers = 1;

    if (res->all_data != NULL) {
        res->buffers[0].size  = size;
        res->buffers[0].metal = nil;

        if (size_aligned > 0) {
            const size_t size_view = ggml_metal_buffer_view_max(props_dev);

            if (props_dev->use_shared_buffers && shared && size_aligned > size_view &&
                (size_aligned + size_view/2 - 1)/(size_view/2) <= GGML_METAL_MAX_BUFFERS) {
                // views of size_view, one every half of that: a tensor of up to half a view is whole in one of them
                res->n_buffers = 0;
                for (size_t i = 0; i < size_aligned; i += size_view/2) {
                    const size_t size_i = i + size_view <= size_aligned ? size_view : size_aligned - i;

                    res->buffers[res->n_buffers].data  = (uint8_t *) res->all_data + i;
                    res->buffers[res->n_buffers].size  = size_i;
                    res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:(uint8_t *) res->all_data + i
                                                                              length:size_i
                                                                             options:MTLResourceStorageModeShared
                                                                         deallocator:nil];
                    if (res->buffers[res->n_buffers].metal == nil) {
                        res->buffers[0].metal = nil;
                        break;
                    }
                    ++res->n_buffers;

                    if (i + size_view >= size_aligned) {
                        break;
                    }
                }
            } else if (props_dev->use_shared_buffers && shared) {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:res->all_data
                                                                  length:size_aligned
                                                                 options:MTLResourceStorageModeShared
                                                             deallocator:nil];
            } else {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithLength:size_aligned options:MTLResourceStorageModePrivate];
            }
        }

        if (res->n_buffers == 1) {
            res->buffers[0].data = res->all_data;
        }
    }

    if (size_aligned > 0 && (res->all_data == NULL || res->buffers[0].metal == nil)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
        free(res);
        return NULL;
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    //ggml_metal_log_allocated_size(device, size_aligned);

    return res;
}

ggml_metal_buffer_t ggml_metal_buffer_map(ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    res->all_data = ptr;
    res->all_size = size;

    res->is_shared = true;
    res->owned = false;

    res->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) ptr % size_page;
        ptr  = (void *) ((char *) ptr - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    // the largest view: the limit of the device, or less when the largest tensor leaves room for the overlap
    size_t size_max = props_dev->max_buffer_size;
    if (max_tensor_size + 2*size_page < ggml_metal_buffer_view_max(props_dev)/2) {
        size_max = ggml_metal_buffer_view_max(props_dev);
    }

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= size_max) {
        res->buffers[res->n_buffers].data  = ptr;
        res->buffers[res->n_buffers].size  = size;
        res->buffers[res->n_buffers].metal = nil;

        if (size_aligned > 0) {
            res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (res->buffers[res->n_buffers].metal == nil) {
                GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
                free(res);
                return NULL;
            }
        }

        ggml_metal_log_allocated_size(res->dev->mtl_device, size_aligned);

        ++res->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_tensor_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = size_max - size_ovlp;
        const size_t size_view = size_max;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            res->buffers[res->n_buffers].data  = (void *) ((uint8_t *) ptr + i);
            res->buffers[res->n_buffers].size  = size_step_aligned;
            res->buffers[res->n_buffers].metal = nil;

            if (size_step_aligned > 0) {
                res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:(void *) ((uint8_t *) ptr + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (res->buffers[res->n_buffers].metal == nil) {
                    GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_step_aligned / 1024.0 / 1024.0);
                    free(res);
                    return NULL;
                }
            }

            ggml_metal_log_allocated_size(res->dev->mtl_device, size_step_aligned);

            if (i + size_step < size) {
                GGML_LOG_INFO("\n");
            }

            ++res->n_buffers;
        }
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    return res;
}

void ggml_metal_buffer_free(ggml_metal_buffer_t buf) {
    @autoreleasepool {
        ggml_metal_device_rsets_rm(buf->dev, buf->rset);

        for (int i = 0; i < buf->n_buffers; i++) {
            [buf->buffers[i].metal release];
        }

        ggml_metal_buffer_rset_free(buf);
    }

    if (buf->is_shared && buf->owned) {
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);
    }

    free(buf);
}

void * ggml_metal_buffer_get_base(ggml_metal_buffer_t buf) {
    return buf->all_data;
}

bool ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf) {
    return buf->is_shared;
}

void ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (buf->is_shared) {
        memset((char *) tensor->data + offset, value, size);
        return;
    }

    @autoreleasepool {
        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:bid_dst.metal
                          range:NSMakeRange(bid_dst.offs, size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_set_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (buf->is_shared) {
        memcpy((char *) tensor->data + offset, data, size);
        return;
    }

    @autoreleasepool {
        // src
        void * data_ptr = (void *)(uintptr_t) data; // "const cast" the src data
        id<MTLBuffer> buf_src = [buf->dev->mtl_device newBufferWithBytesNoCopy:data_ptr
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        GGML_ASSERT(buf_src);

        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        // note: for experimentation purposes, here we use a semaphore to wait for the copy to complete
        //       this is alternative to waitUntilCompleted, which should be faster, but don't seem to make much difference
        dispatch_semaphore_t completion_semaphore = dispatch_semaphore_create(0);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:buf_src
                       sourceOffset:0
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                             // TODO: can check for errors here
            GGML_UNUSED(cb);

            dispatch_semaphore_signal(completion_semaphore);
        }];

        [cmd_buf commit];

        dispatch_semaphore_wait(completion_semaphore, DISPATCH_TIME_FOREVER);
        dispatch_release(completion_semaphore);

        //[cmd_buf waitUntilCompleted];

        [buf_src release];
    }
}

void ggml_metal_buffer_get_tensor(ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (buf->is_shared) {
        memcpy(data, (const char *) tensor->data + offset, size);
        return;
    }

    @autoreleasepool {
        // src
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf, tensor);
        bid_src.offs += offset;

        // dst
        id<MTLBuffer> buf_dst = [buf->dev->mtl_device newBufferWithBytesNoCopy:data
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        GGML_ASSERT(buf_dst);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:buf_dst
                  destinationOffset:0
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        [buf_dst release];
    }
}

static bool ggml_metal_buffer_data_span(const void * const * data, const size_t * sizes, size_t n_tensors, uintptr_t * min_data, uintptr_t * max_data) {
    bool found = false;

    for (size_t i = 0; i < n_tensors; i++) {
        if (sizes[i] == 0) {
            continue;
        }

        const uintptr_t start = (uintptr_t) data[i];
        const uintptr_t end   = start + sizes[i];
        if (!found) {
            *min_data = start;
            *max_data = end;
            found = true;
        } else {
            if (start < *min_data) {
                *min_data = start;
            }
            if (end > *max_data) {
                *max_data = end;
            }
        }
    }

    return found;
}

void ggml_metal_buffer_set_tensor_n(ggml_metal_buffer_t buf, struct ggml_tensor * const * tensors, const void * const * data,
        const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    if (buf->is_shared) {
        for (size_t i = 0; i < n_tensors; i++) {
            if (sizes[i] == 0) {
                continue;
            }
            memcpy((char *) tensors[i]->data + offsets[i], data[i], sizes[i]);
        }
        return;
    }

    uintptr_t min_data = 0, max_data = 0;
    if (!ggml_metal_buffer_data_span(data, sizes, n_tensors, &min_data, &max_data)) {
        return;
    }

    @autoreleasepool {
        void * data_ptr = (void *) min_data;
        id<MTLBuffer> buf_src = [buf->dev->mtl_device newBufferWithBytesNoCopy:data_ptr
                                                               length:max_data - min_data
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        GGML_ASSERT(buf_src);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            for (size_t i = 0; i < n_tensors; i++) {
                if (sizes[i] == 0) {
                    continue;
                }

                ggml_backend_buffer_t tensor_buf_ggml = tensors[i]->view_src ? tensors[i]->view_src->buffer : tensors[i]->buffer;
                ggml_metal_buffer_t tensor_buf = (ggml_metal_buffer_t) tensor_buf_ggml->context;
                struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(tensor_buf, tensors[i]);
                bid_dst.offs += offsets[i];

                [encoder copyFromBuffer:buf_src
                           sourceOffset:(uintptr_t) data[i] - min_data
                               toBuffer:bid_dst.metal
                      destinationOffset:bid_dst.offs
                                   size:sizes[i]];
            }

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        [buf_src release];
    }
}

void ggml_metal_buffer_get_tensor_n(ggml_metal_buffer_t buf, const struct ggml_tensor * const * tensors, void * const * data,
        const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    if (buf->is_shared) {
        for (size_t i = 0; i < n_tensors; i++) {
            if (sizes[i] == 0) {
                continue;
            }
            memcpy(data[i], (const char *) tensors[i]->data + offsets[i], sizes[i]);
        }
        return;
    }

    uintptr_t min_data = 0, max_data = 0;
    if (!ggml_metal_buffer_data_span((const void * const *) data, sizes, n_tensors, &min_data, &max_data)) {
        return;
    }

    @autoreleasepool {
        id<MTLBuffer> buf_dst = [buf->dev->mtl_device newBufferWithBytesNoCopy:(void *) min_data
                                                               length:max_data - min_data
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        GGML_ASSERT(buf_dst);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            for (size_t i = 0; i < n_tensors; i++) {
                if (sizes[i] == 0) {
                    continue;
                }

                ggml_backend_buffer_t tensor_buf_ggml = tensors[i]->view_src ? tensors[i]->view_src->buffer : tensors[i]->buffer;
                ggml_metal_buffer_t tensor_buf = (ggml_metal_buffer_t) tensor_buf_ggml->context;
                struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(tensor_buf, tensors[i]);
                bid_src.offs += offsets[i];

                [encoder copyFromBuffer:bid_src.metal
                           sourceOffset:bid_src.offs
                               toBuffer:buf_dst
                      destinationOffset:(uintptr_t) data[i] - min_data
                                   size:sizes[i]];
            }

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        [buf_dst release];
    }
}

bool ggml_metal_buffer_cpy_tensor(ggml_metal_buffer_t buf_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_metal_buffer_t buf_src = (ggml_metal_buffer_t)src->buffer->context;

    const size_t size = ggml_nbytes(src);

    // if both buffers are shared, we can use memcpy directly
    if (buf_dst->is_shared && buf_src->is_shared) {
        memcpy(dst->data, src->data, size);
        return true;
    }

    // for private buffers, we need to use Metal blit commands
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf_src, src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf_dst, dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        id<MTLCommandBuffer> cmd_buf = [buf_dst->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }

    return true;
}

bool ggml_metal_buffer_cpy_tensor_n(ggml_metal_buffer_t buf_dst, const struct ggml_tensor * const * srcs, struct ggml_tensor * const * dsts, size_t n_tensors) {
    if (n_tensors == 0) {
        return true;
    }

    bool all_shared = true;
    for (size_t i = 0; i < n_tensors; ++i) {
        ggml_backend_buffer_t src_buf = srcs[i]->view_src ? srcs[i]->view_src->buffer : srcs[i]->buffer;
        ggml_metal_buffer_t buf_src = (ggml_metal_buffer_t) src_buf->context;
        if (!buf_dst->is_shared || !buf_src->is_shared) {
            all_shared = false;
            break;
        }
    }

    if (all_shared) {
        for (size_t i = 0; i < n_tensors; ++i) {
            memcpy(dsts[i]->data, srcs[i]->data, ggml_nbytes(srcs[i]));
        }
        return true;
    }

    // for private buffers, we need to use Metal blit commands
    @autoreleasepool {
        struct ggml_metal_buffer_id * bid_srcs = malloc(n_tensors * sizeof(struct ggml_metal_buffer_id));
        struct ggml_metal_buffer_id * bid_dsts = malloc(n_tensors * sizeof(struct ggml_metal_buffer_id));

        for (size_t i = 0; i < n_tensors; ++i) {
            ggml_backend_buffer_t src_buf = srcs[i]->view_src ? srcs[i]->view_src->buffer : srcs[i]->buffer;
            ggml_backend_buffer_t dst_buf = dsts[i]->view_src ? dsts[i]->view_src->buffer : dsts[i]->buffer;
            ggml_metal_buffer_t buf_src = (ggml_metal_buffer_t) src_buf->context;
            ggml_metal_buffer_t buf_dst_i = (ggml_metal_buffer_t) dst_buf->context;
            bid_srcs[i] = ggml_metal_buffer_get_id(buf_src, srcs[i]);
            bid_dsts[i] = ggml_metal_buffer_get_id(buf_dst_i, dsts[i]);

            if (bid_srcs[i].metal == nil || bid_dsts[i].metal == nil) {
                free(bid_srcs);
                free(bid_dsts);
                return false;
            }
        }

        id<MTLCommandBuffer> cmd_buf = [buf_dst->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            for (size_t i = 0; i < n_tensors; ++i) {
                [encoder copyFromBuffer:bid_srcs[i].metal
                           sourceOffset:bid_srcs[i].offs
                               toBuffer:bid_dsts[i].metal
                      destinationOffset:bid_dsts[i].offs
                                   size:ggml_nbytes(srcs[i])];
            }

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        free(bid_srcs);
        free(bid_dsts);
    }

    return true;
}

void ggml_metal_buffer_clear(ggml_metal_buffer_t buf, uint8_t value) {
    if (buf->is_shared) {
        memset(buf->all_data, value, buf->all_size);
        return;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:buf->buffers[0].metal
                          range:NSMakeRange(0, buf->buffers[0].size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t) {
    struct ggml_metal_buffer_id res = { nil, 0 };

    const int64_t tsize = ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < buf->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf->buffers[i].data;

        //GGML_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, buf->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, buf->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf->buffers[i].size) {
            res.metal = buf->buffers[i].metal;
            res.offs  = (size_t) ioffs;

            //GGML_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);

            return res;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return res;
}
