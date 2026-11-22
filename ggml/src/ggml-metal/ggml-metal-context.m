#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"
#import "ggml-metal-fusion.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#include <stdatomic.h>
#include <os/lock.h>

// GGML_METAL_PROFILE=1: sum the GPU execution time of every command buffer, printed when a context is freed.
// against the wall time of a decode it tells GPU work from waiting (encoding, scheduling, idle gaps)
static int             g_profile = -1;
static _Atomic int64_t g_profile_gpu_ns   = 0;
static _Atomic int64_t g_profile_sched_ns = 0; // kernel start -> GPU start
static _Atomic int64_t g_profile_n_cb     = 0;
static _Atomic int64_t g_profile_n_graphs = 0;

// GPU time by graph size, to tell the graphs of a round apart (the model, its drafter, their small helpers)
#define GGML_METAL_PROFILE_BUCKETS 24
static _Atomic int64_t g_profile_bucket_nodes [GGML_METAL_PROFILE_BUCKETS];
static _Atomic int64_t g_profile_bucket_graphs[GGML_METAL_PROFILE_BUCKETS];
static _Atomic int64_t g_profile_bucket_gpu_ns[GGML_METAL_PROFILE_BUCKETS];
// profile: from the end of a graph on the GPU to the CPU running again, and from there to the commit of the next
static double g_profile_t_wake      = 0.0;
static double g_profile_wake_s      = 0.0;
static double g_profile_turnaround_s = 0.0;
static int    g_profile_n_wake      = 0;

static _Atomic int64_t g_profile_gap_graph_ns = 0;
static _Atomic int64_t g_profile_gap_cb_ns    = 0;
static _Atomic int64_t g_profile_n_recorded   = 0;
static _Atomic int64_t g_profile_n_replayed   = 0;

static int ggml_metal_profile_bucket(int n_nodes) {
    for (int i = 0; i < GGML_METAL_PROFILE_BUCKETS; ++i) {
        int64_t cur = atomic_load(&g_profile_bucket_nodes[i]);
        if (cur == n_nodes) {
            return i;
        }
        if (cur == 0) {
            int64_t expected = 0;
            if (atomic_compare_exchange_strong(&g_profile_bucket_nodes[i], &expected, (int64_t) n_nodes) || expected == n_nodes) {
                return i;
            }
        }
    }
    return GGML_METAL_PROFILE_BUCKETS - 1;
}

static void ggml_metal_profile_cmd_buf_graph(id<MTLCommandBuffer> cmd_buf, int n_nodes, bool first) {
    if (g_profile < 0) {
        g_profile = getenv("GGML_METAL_PROFILE") != NULL ? 1 : 0;
    }
    if (!g_profile) {
        return;
    }
    const int b = ggml_metal_profile_bucket(n_nodes);
    if (first) {
        atomic_fetch_add(&g_profile_bucket_graphs[b], 1);
    }
    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        atomic_fetch_add(&g_profile_bucket_gpu_ns[b], (int64_t) (([cb GPUEndTime] - [cb GPUStartTime])*1e9));

        // the time the GPU sat idle before this command buffer: between two graphs (the CPU work of a decode) or
        // inside one (the encoder of the next command buffer was not done)
        static double end_prev = 0.0;
        static os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
        os_unfair_lock_lock(&lock);
        const double gap = end_prev > 0.0 ? [cb GPUStartTime] - end_prev : 0.0;
        if ([cb GPUEndTime] > end_prev) {
            end_prev = [cb GPUEndTime];
        }
        os_unfair_lock_unlock(&lock);
        if (gap > 0.0 && gap < 1.0) {
            atomic_fetch_add(first ? &g_profile_gap_graph_ns : &g_profile_gap_cb_ns, (int64_t) (gap*1e9));
        }
    }];
}

static void ggml_metal_profile_cmd_buf(id<MTLCommandBuffer> cmd_buf) {
    if (g_profile < 0) {
        g_profile = getenv("GGML_METAL_PROFILE") != NULL ? 1 : 0;
    }
    if (!g_profile) {
        return;
    }
    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        atomic_fetch_add(&g_profile_gpu_ns,   (int64_t) (([cb GPUEndTime] - [cb GPUStartTime])*1e9));
        atomic_fetch_add(&g_profile_sched_ns, (int64_t) (([cb GPUStartTime] - [cb kernelStartTime])*1e9));
        atomic_fetch_add(&g_profile_n_cb, 1);
    }];
}

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 32

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy;  // for async copies
    ggml_metal_event_t ev_sync; // destination completion signal

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;

    struct ggml_metal_fusion_info * finfo;

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers

    // first node of each extra command buffer, then the end of the graph
    int cb_start[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // buffers to release after async Metal operations complete
    // if Metal released them, it would do so on a Metal-internal thread without an autorelease pool, which could cause leaks
    NSMutableArray * buf_refs;

    // the kernels of the last graph, recorded to be executed again without encoding (see ggml_metal_recording).
    // a graph is the same as long as it has not gone through ggml_metal_graph_optimize again, which every graph that is
    // built or allocated does: graph_gen counts those
    bool     use_replay;
    uint64_t graph_gen;
    ggml_metal_recording_t rec;
    void *   rec_nodes; // identity of the recorded graph: its node array, node count and generation
    int      rec_n_nodes;
    uint64_t rec_gen;
    uint64_t rec_hash;

    // GGML_METAL_PREENCODE: the command buffer of the next execution of the same graph, encoded while the GPU runs
    // this one
    id<MTLCommandBuffer> pre_buf;
    void *   pre_nodes;
    int      pre_n_nodes;
    uint64_t pre_gen;
    uint64_t pre_hash;
    void *   seen_nodes; // of the last graph that was encoded: one that comes back is worth recording
    int      seen_n_nodes;
    uint64_t seen_gen;
    uint64_t seen_hash;
    bool     seen_failed;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;
};

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

    @autoreleasepool {
#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
        // Show all the Metal device instances in the system
        NSArray * devices = MTLCopyAllDevices();
        for (id<MTLDevice> device in devices) {
            GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
        }
        [devices release]; // since it was created by a *Copy* C method
#endif

        // init context
        ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

        id<MTLDevice> device = ggml_metal_device_get_obj(dev);

        GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

        // TODO: would it be better to have one queue for the backend and one queue for the device?
        //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
        //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
        if (queue == nil) {
            GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            free(res);
            return NULL;
        }

        res->dev = dev;
        res->lib = ggml_metal_device_get_library(dev);
        if (res->lib == NULL) {
            GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
            GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

            res->lib = ggml_metal_library_init(dev);
            if (res->lib == NULL) {
                GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

                free(res);

                return NULL;
            }
        }

        res->ev_cpy  = ggml_metal_device_event_init(dev);
        res->ev_sync = ggml_metal_device_event_init(dev);

        const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

        snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

        res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

        res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

        {
            const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
            res->debug_graph = val ? atoi(val) : 0;
        }

        res->use_graph_optimize = true;

        if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
            res->use_graph_optimize = false;
        }

        res->finfo = ggml_metal_device_get_fusion_info(dev);
        if (ggml_metal_fusion_info_stats(res->finfo)) {
            ggml_metal_fusion_info_labels_init(res->finfo);
            res->n_cb = 0;
        }

        GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, ggml_metal_fusion_info_enabled(res->finfo) ? "true" : "false");
        GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
        GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

        res->capture_compute = 0;
        res->capture_started = false;
        res->capture_scope = nil;

        {
            const char * val = getenv("GGML_METAL_CAPTURE_COMPUTE");
            if (val) {
                res->capture_compute = atoi(val);
            }
        }

        res->has_error = false;

        res->gf = nil;
        res->encode_async = nil;
        for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
            res->cmd_bufs[i].obj = nil;
        }

        res->cmd_bufs_ext = [[NSMutableArray alloc] init];
        res->buf_refs     = [[NSMutableArray alloc] init];

        // GGML_METAL_REPLAY_DISABLE: encode every graph. captures and fusion statistics need the encoders too
        res->use_replay = getenv("GGML_METAL_REPLAY_DISABLE") == NULL && getenv("GGML_METAL_CAPTURE_COMPUTE") == NULL &&
            !ggml_metal_fusion_info_stats(res->finfo);

        res->cmd_buf_last = nil;

        res->pipelines_ext = ggml_metal_pipelines_init();

        return res;
    }
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
        [ctx->buf_refs release];
    }

    if (ctx->pre_buf != nil) {
        [ctx->pre_buf release];
        ctx->pre_buf = nil;
    }
    ggml_metal_recording_free(ctx->rec);
    ctx->rec = NULL;

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (g_profile > 0) {
        for (int i = 0; i < GGML_METAL_PROFILE_BUCKETS; ++i) {
            if (g_profile_bucket_graphs[i] > 0) {
                fprintf(stderr, "ggml_metal_free: profile: graphs of %6lld nodes: %6lld, GPU %8.1f ms, %6.2f ms each\n",
                        (long long) g_profile_bucket_nodes[i], (long long) g_profile_bucket_graphs[i],
                        g_profile_bucket_gpu_ns[i]/1e6, g_profile_bucket_gpu_ns[i]/1e6/g_profile_bucket_graphs[i]);
            }
        }
        fprintf(stderr, "ggml_metal_free: profile: %lld graphs recorded, %lld executed from a recording\n", (long long) g_profile_n_recorded, (long long) g_profile_n_replayed);
        fprintf(stderr, "ggml_metal_free: profile: GPU idle between graphs %.1f ms, between the command buffers of a graph %.1f ms\n",
                g_profile_gap_graph_ns/1e6, g_profile_gap_cb_ns/1e6);
        if (g_profile_n_wake > 0) {
            fprintf(stderr, "ggml_metal_free: profile: a token: GPU end to CPU awake %.0f us, CPU to the next commit %.0f us (%d waits)\n",
                    g_profile_wake_s*1e6/g_profile_n_wake, g_profile_turnaround_s*1e6/g_profile_n_wake, g_profile_n_wake);
        }
        int64_t barriers, depth, kernels;
        ggml_metal_op_profile_get(&barriers, &depth, &kernels);
        fprintf(stderr, "ggml_metal_free: profile: %lld kernels, %lld barriers, depth %lld (summed over encoders)" "\n", (long long) kernels, (long long) barriers, (long long) depth);
        fprintf(stderr, "ggml_metal_free: profile: %lld graphs, %lld command buffers, GPU time %.1f ms, kernel-to-GPU wait %.1f ms\n",
                (long long) g_profile_n_graphs, (long long) g_profile_n_cb, g_profile_gpu_ns/1e6, g_profile_sched_ns/1e6);
    }

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);

        const int n_fusions = ggml_metal_fusion_info_n_fusions(ctx->finfo);
        for (int i = 0; i < n_fusions; i++) {
            const uint64_t count = ggml_metal_fusion_info_count(ctx->finfo, i);
            if (count == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_metal_fusion_info_label(ctx->finfo, i), count);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);
    ggml_metal_device_event_free(ctx->dev, ctx->ev_sync);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        // (polling the status instead brought the thread back in 60 us rather than 89 on an M5 Ultra, but only when it
        // polled for the whole execution: sleeping until shortly before the end and polling from there gained nothing)
        [ctx->cmd_buf_last waitUntilCompleted];
        if (getenv("GGML_METAL_PROFILE")) {
            g_profile_t_wake  = [[NSProcessInfo processInfo] systemUptime];
            g_profile_wake_s += g_profile_t_wake - [ctx->cmd_buf_last GPUEndTime];
            g_profile_n_wake++;
        }
        ctx->cmd_buf_last = nil;
    }

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ctx->has_error = true;
                return;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        // wrap the source data into a Metal buffer
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_src];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    @autoreleasepool {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        GGML_ASSERT(buf_dst);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:buf_dst
              destinationOffset:0
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_dst];
        [buf_dst release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        id<MTLCommandQueue> dst_queue = ggml_metal_device_get_queue(ctx_dst->dev);
        id<MTLCommandBuffer> sync_cmd_buf = [dst_queue commandBuffer];

        ggml_metal_event_encode_signal(ctx_dst->ev_sync, sync_cmd_buf);

        [sync_cmd_buf commit];

        [ctx_dst->cmd_bufs_ext addObject:sync_cmd_buf];
        ctx_dst->cmd_buf_last = sync_cmd_buf;

        [sync_cmd_buf retain];

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx_src->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_event_encode_wait(ctx_dst->ev_sync, cmd_buf);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

// what the kernels encoded for a graph depend on, folded to one number: its nodes, where they are allocated, their
// shapes and their parameters. a graph that did not go through ggml_metal_graph_optimize keeps all of this, but a
// caller that computes graphs without a scheduler reuses node memory for another graph without telling the backend
static uint64_t ggml_metal_graph_identity(const struct ggml_cgraph * gf) {
    uint64_t hs[4] = { 1469598103934665603ull, 1469598103934665603ull, 1469598103934665603ull, 1469598103934665603ull };
    for (int i = 0; i < gf->n_nodes; ++i) {
        const struct ggml_tensor * t = gf->nodes[i];
        uint64_t p[8];
        memcpy(p, t->op_params, sizeof(p));
        uint64_t x = (uint64_t) (uintptr_t) t ^ ((uint64_t) (uintptr_t) t->data << 1) ^ ((uint64_t) t->view_offs << 3) ^
            ((uint64_t) t->op << 56) ^ ((uint64_t) t->type << 48) ^ ((uint64_t) (t->flags & GGML_TENSOR_FLAG_COMPUTE) << 40) ^
            ((uint64_t) t->ne[0] << 8) ^ ((uint64_t) t->ne[1] << 24) ^ ((uint64_t) t->ne[2] << 36) ^ ((uint64_t) t->ne[3] << 44) ^
            p[0] ^ (p[1] << 7) ^ (p[2] << 13) ^ (p[3] << 19) ^ (p[4] << 23) ^ (p[5] << 29) ^ (p[6] << 31) ^ (p[7] << 37);
        for (int j = 0; j < GGML_MAX_SRC && t->src[j]; ++j) {
            x ^= ((uint64_t) (uintptr_t) t->src[j] << (5 + 3*j)) ^ ((uint64_t) (uintptr_t) t->src[j]->data >> 2);
        }
        // four interleaved chains: the multiplies of one do not wait for those of another
        hs[i & 3] = (hs[i & 3] ^ x)*1099511628211ull;
    }
    return hs[0] ^ (hs[1]*3) ^ (hs[2]*5) ^ (hs[3]*7);
}

// encode the whole graph in one command buffer, on this thread; the caller commits it
static id<MTLCommandBuffer> ggml_metal_encode_whole(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
    id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
    [cmd_buf retain];

    ggml_metal_op_t ctx_op = ggml_metal_op_init(ctx->dev, cmd_buf, gf, ctx->finfo, 0, gf->n_nodes, ctx->use_concurrency, false, ctx->debug_graph);
    for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
        const int res = ggml_metal_op_encode(ctx_op, idx);
        if (res == 0) {
            break;
        }
        idx += res - 1;
    }
    ggml_metal_op_free(ctx_op);

    return cmd_buf;
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    if (gf->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }

    // number of nodes encoded by the main thread (empirically determined)
    // GGML_METAL_N_MAIN: that number instead (a small first command buffer gets the GPU going sooner)
    static int n_main_env = -1;
    if (n_main_env < 0) {
        n_main_env = getenv("GGML_METAL_N_MAIN") ? atoi(getenv("GGML_METAL_N_MAIN")) : 0;
    }
    const int n_main = n_main_env > 0 ? n_main_env : MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_main nodes are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    if (ctx->use_replay && ctx->abort_callback == NULL) {
        const uint64_t hash = ggml_metal_graph_identity(gf);

        const bool recorded = ctx->rec != NULL && ctx->rec_hash == hash &&
            ctx->rec_nodes == (void *) gf->nodes && ctx->rec_n_nodes == gf->n_nodes && ctx->rec_gen == ctx->graph_gen;
        // GGML_METAL_REPLAY_FORCE: record every graph the first time it is seen (to test the recording itself)
        static int force = -1;
        if (force < 0) {
            force = getenv("GGML_METAL_REPLAY_FORCE") != NULL;
        }
        const bool seen = force || (ctx->seen_hash == hash &&
            ctx->seen_nodes == (void *) gf->nodes && ctx->seen_n_nodes == gf->n_nodes && ctx->seen_gen == ctx->graph_gen);

        static int preencode = -1;
        if (preencode < 0) {
            // on by default; GGML_METAL_PREENCODE_DISABLE falls back to the recording in an indirect command buffer
            preencode = getenv("GGML_METAL_PREENCODE_DISABLE") == NULL;
        }
        if (preencode && seen) {
            @autoreleasepool {
                ctx->gf = gf;

                id<MTLCommandBuffer> cmd_buf = nil;
                if (ctx->pre_buf != nil && ctx->pre_hash == hash && ctx->pre_nodes == (void *) gf->nodes &&
                    ctx->pre_n_nodes == gf->n_nodes && ctx->pre_gen == ctx->graph_gen) {
                    cmd_buf = ctx->pre_buf;
                    atomic_fetch_add(&g_profile_n_replayed, 1);
                } else {
                    if (ctx->pre_buf != nil) {
                        [ctx->pre_buf release];
                    }
                    cmd_buf = ggml_metal_encode_whole(ctx, gf);
                }
                ctx->pre_buf = nil;

                for (int i = 0; i <= ctx->n_cb; ++i) {
                    if (ctx->cmd_bufs[i].obj) {
                        [ctx->cmd_bufs[i].obj release];
                        ctx->cmd_bufs[i].obj = nil;
                    }
                }
                ctx->cmd_bufs[ctx->n_cb].obj = cmd_buf;

                ggml_metal_profile_cmd_buf(cmd_buf);
                ggml_metal_profile_cmd_buf_graph(cmd_buf, gf->n_nodes, true);
                atomic_fetch_add(&g_profile_n_graphs, 1);

                if (g_profile_t_wake > 0.0) {
                    g_profile_turnaround_s += [[NSProcessInfo processInfo] systemUptime] - g_profile_t_wake;
                    g_profile_t_wake = 0.0;
                }

                [cmd_buf commit];
                ctx->cmd_buf_last = cmd_buf;

                // the next one, while the GPU works
                ctx->pre_buf     = ggml_metal_encode_whole(ctx, gf);
                ctx->pre_nodes   = (void *) gf->nodes;
                ctx->pre_n_nodes = gf->n_nodes;
                ctx->pre_gen     = ctx->graph_gen;
                ctx->pre_hash    = hash;
            }

            return GGML_STATUS_SUCCESS;
        }

        if (!recorded && seen && !ctx->seen_failed) {
            // the second time in a row: record it, on this thread, with the encoders of the ops as they are
            @autoreleasepool {
                ggml_metal_recording_free(ctx->rec);
                ctx->rec = ggml_metal_recording_init(ctx->dev, MIN(gf->n_nodes, 16384));

                ctx->gf = gf;

                ggml_metal_recording_begin(ctx->rec);
                ggml_metal_op_t ctx_op = ggml_metal_op_init(ctx->dev, NULL, gf, ctx->finfo, 0, gf->n_nodes, ctx->use_concurrency, false, ctx->debug_graph);

                for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
                    const int res = ggml_metal_op_encode(ctx_op, idx);
                    if (res == 0) {
                        break;
                    }
                    idx += res - 1;
                }

                ggml_metal_op_free(ctx_op);

                if (ggml_metal_recording_ok(ctx->rec)) {
                    ctx->rec_nodes   = (void *) gf->nodes;
                    ctx->rec_n_nodes = gf->n_nodes;
                    ctx->rec_gen     = ctx->graph_gen;
                    ctx->rec_hash    = hash;
                    atomic_fetch_add(&g_profile_n_recorded, 1);
                } else {
                    ggml_metal_recording_free(ctx->rec);
                    ctx->rec = NULL;
                    ctx->seen_failed = true;
                }
            }
        }

        if (ctx->rec != NULL && ctx->rec_hash == hash &&
            ctx->rec_nodes == (void *) gf->nodes && ctx->rec_n_nodes == gf->n_nodes && ctx->rec_gen == ctx->graph_gen) {
            @autoreleasepool {
                ctx->gf = gf;

                ggml_metal_device_rsets_keep_alive(ctx->dev);

                id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

                id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
                [cmd_buf retain];

                // one command buffer for the graph, in the slot of the main thread
                for (int i = 0; i <= ctx->n_cb; ++i) {
                    if (ctx->cmd_bufs[i].obj) {
                        [ctx->cmd_bufs[i].obj release];
                        ctx->cmd_bufs[i].obj = nil;
                    }
                }
                ctx->cmd_bufs[ctx->n_cb].obj = cmd_buf;

                ggml_metal_profile_cmd_buf(cmd_buf);
                ggml_metal_profile_cmd_buf_graph(cmd_buf, gf->n_nodes, true);
                atomic_fetch_add(&g_profile_n_graphs, 1);
                atomic_fetch_add(&g_profile_n_replayed, 1);

                ggml_metal_recording_replay(ctx->rec, cmd_buf);

                [cmd_buf commit];

                ctx->cmd_buf_last = cmd_buf;
            }

            return GGML_STATUS_SUCCESS;
        }

        if (!seen) {
            ctx->seen_nodes   = (void *) gf->nodes;
            ctx->seen_n_nodes = gf->n_nodes;
            ctx->seen_gen     = ctx->graph_gen;
            ctx->seen_hash    = hash;
            ctx->seen_failed  = false;
        }
    }

    @autoreleasepool {
        ctx->gf = gf;

        // the main thread encodes [0, cb_start[0]) and the extra command buffer i encodes [cb_start[i], cb_start[i + 1])
        // a fusion cannot span two command buffers, so each boundary moves to the end of the fusable group it falls
        // into and the encoders fuse the same nodes as with a single command buffer
        {
            const int n_rest   = gf->n_nodes - MIN(n_main, gf->n_nodes);
            const int n_per_cb = n_cb > 0 ? (n_rest + n_cb - 1) / n_cb : 0;

            int i = 0;
            for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
                const int target = MIN(n_main + cb_idx*n_per_cb, gf->n_nodes);
                while (i < target) {
                    i += ggml_metal_fusion_max(gf, i);
                }
                ctx->cb_start[cb_idx] = MIN(i, gf->n_nodes);
            }
            ctx->cb_start[n_cb] = gf->n_nodes;
        }

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                const bool ok = [[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error];

                [descriptor release];

                if (!ok) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s' (did you set METAL_CAPTURE_ENABLED=1 ?)\n", __func__, [[error localizedDescription] UTF8String]);
                    [ctx->capture_scope release];
                    ctx->capture_scope = nil;
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            ggml_metal_profile_cmd_buf(cmd_buf);
            ggml_metal_profile_cmd_buf_graph(cmd_buf, gf->n_nodes, true);
            atomic_fetch_add(&g_profile_n_graphs, 1);

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            ggml_metal_profile_cmd_buf(cmd_buf);
            ggml_metal_profile_cmd_buf_graph(cmd_buf, gf->n_nodes, false);

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            enum ggml_status res = GGML_STATUS_SUCCESS;

            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    res = GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb && res == GGML_STATUS_SUCCESS; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }
                    res = GGML_STATUS_FAILED;
                    break;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    res = GGML_STATUS_ABORTED;
                    break;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            [ctx->capture_scope release];
            ctx->capture_scope = nil;

            ctx->capture_started = false;

            if (res != GGML_STATUS_SUCCESS) {
                return res;
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    // GGML_METAL_DUMP_RAW=file: the nodes in the order the graph was built in, before any reordering, for writing
    // fusion patterns (which match consecutive nodes). the last graph optimized wins
    {
        static const char * dump = NULL;
        static bool dump_init = false;
        if (!dump_init) {
            dump = getenv("GGML_METAL_DUMP_RAW");
            dump_init = true;
        }
        if (dump) {
            FILE * f = fopen(dump, "w");
            if (f) {
                for (int i = 0; i < gf->n_nodes; ++i) {
                    const struct ggml_tensor * t = gf->nodes[i];
                    fprintf(f, "%5d %-14s %-28s %s[%lld,%lld,%lld,%lld]", i, ggml_op_name(t->op), t->name, ggml_type_name(t->type),
                            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
                    for (int j = 0; j < GGML_MAX_SRC && t->src[j]; ++j) {
                        int k = i - 1;
                        while (k >= 0 && gf->nodes[k] != t->src[j]) {
                            k--;
                        }
                        fprintf(f, " <%d:%s", k, t->src[j]->name);
                    }
                    fprintf(f, "\n");
                }
                fclose(f);
            }
        }
    }

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    // a graph that comes through here is new or was allocated again: what was recorded for it no longer holds
    ctx->graph_gen++;

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    // when fusion stats are collected the graph must be encoded by a single thread so the
    // counters are race-free; override whatever the caller requested
    if (ggml_metal_fusion_info_stats(ctx->finfo)) {
        n_cb = 0;
    }

    // GGML_METAL_N_CB: the number of command buffers, each encoded by a thread of its own, after the first
    if (n_cb > 0 && getenv("GGML_METAL_N_CB") != NULL) {
        n_cb = atoi(getenv("GGML_METAL_N_CB"));
    }

    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        int idx_start = 0;
        int idx_end   = ctx->cb_start[0];

        if (cb_idx < n_cb_l) {
            idx_start = ctx->cb_start[cb_idx];
            idx_end   = ctx->cb_start[cb_idx + 1];
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            ctx->finfo,
            idx_start,
            idx_end,
            ctx->use_concurrency,
            ctx->capture_compute == 0,
            ctx->debug_graph);

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);

        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}
