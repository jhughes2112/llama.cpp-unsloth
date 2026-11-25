#include "ggml-metal-mega.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

// see ggml-metal-mega.h

namespace {

struct param {
    enum { BIND, TG, ATTR } what;
    int         idx;  // binding or threadgroup index
    std::string attr; // ATTR: the attribute name
    bool        cnst = false; // BIND: a constant-address-space parameter (must be set_bytes)
};

// the text of a parenthesised list starting at s[open] == '(', split at its top-level commas
static bool split_params(const std::string & s, size_t open, std::vector<std::string> & out, size_t & close) {
    int d_par = 0, d_ang = 0, d_brk = 0;
    std::string cur;
    for (size_t i = open; i < s.size(); ++i) {
        const char c = s[i];
        if (c == '(') { if (d_par++ == 0) continue; }
        if (c == ')') { if (--d_par == 0) { if (!cur.empty()) out.push_back(cur); close = i; return true; } }
        if (c == '<') d_ang++;
        if (c == '>') d_ang--;
        if (c == '[') d_brk++;
        if (c == ']') d_brk--;
        if (c == ',' && d_par == 1 && d_ang <= 0 && d_brk == 0) { out.push_back(cur); cur.clear(); continue; }
        cur += c;
    }
    return false;
}

static std::string trim(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
}

// the matching closing brace of the body that starts at the first '{' at or after pos
static size_t body_end(const std::string & s, size_t pos) {
    size_t i = s.find('{', pos);
    if (i == std::string::npos) return std::string::npos;
    int d = 0;
    for (; i < s.size(); ++i) {
        if (s[i] == '{') d++;
        if (s[i] == '}' && --d == 0) return i;
    }
    return std::string::npos;
}

// kernels that declare threadgroup memory in their body cannot be device functions: cut them out (with their template
// line), and the typedefs and instantiations that name them; also the ones that do not compile with coherent pointers
static std::string drop_kernels(const std::string & src) {
    static const std::regex re_kernel(R"((^|\n)([ \t]*template[ \t]*<[^\n]*\n)?[ \t]*kernel[ \t]+void[ \t]+(\w+)[ \t]*\()");
    static const std::regex re_tgdecl(R"(\n[ \t]*threadgroup[ \t]+[\w:<>, ]+[ \t]+\w+[ \t]*\[)");
    static const std::set<std::string> always = { "kernel_concat_q" };
    std::string out;
    std::set<std::string> dropped;
    size_t pos = 0;
    auto it = std::sregex_iterator(src.begin(), src.end(), re_kernel);
    for (; it != std::sregex_iterator(); ++it) {
        const auto & m = *it;
        const size_t start = m.position(0) + m.length(1);
        if (start < pos) continue;
        const size_t end = body_end(src, start);
        if (end == std::string::npos) break;
        const std::string body = src.substr(start, end - start + 1);
        const std::string name = m.str(3);
        if (always.count(name) || std::regex_search(body, re_tgdecl)) {
            out.append(src, pos, start - pos);
            pos = end + 1;
            dropped.insert(name);
        }
    }
    out.append(src, pos, std::string::npos);
    if (dropped.empty()) return out;
    // the lines that name a dropped kernel in a typedef or an instantiation
    std::string res;
    size_t a = 0;
    while (a < out.size()) {
        size_t b = out.find('\n', a);
        if (b == std::string::npos) b = out.size();
        const std::string line = out.substr(a, b - a);
        bool skip = false;
        const std::string t = trim(line);
        if (t.rfind("typedef", 0) == 0 || t.rfind("template", 0) == 0) {
            for (const auto & n : dropped) {
                std::regex re("\\b" + n + "\\b");
                if (std::regex_search(line, re)) { skip = true; break; }
            }
        }
        if (!skip) { res += line; res += '\n'; }
        a = b + 1;
    }
    return res;
}

// function constants become compile-time constants of the namespace
// GGML_METAL_MEGA_COHSTORE: every device pointer that is not const becomes coherent (stores), const ones stay plain
static std::string coherent_stores(const std::string & src) {
    // "const device T" is the same type as "device const T": spell it that way first
    static const std::regex re_cd(R"(\bconst(\s+)device\b)");
    static const std::regex re(R"(\bdevice\b(?!\s+const\b))");
    // "device const T * p = (device T *)(...)": a cast that only drops const to read, keep it plain
    static const std::regex re_rd(R"((device\s+const\s+[\w:<>]+\s*\*\s*\w+\s*=\s*\()coherent\(device\) device(\s+[\w:<>]+\s*\*\s*\)))");
    std::string s = std::regex_replace(std::regex_replace(src, re_cd, "device$1const"), re, "coherent(device) device");
    return std::regex_replace(s, re_rd, "$1device const$2");
}

static std::string rewrite_fc(const std::string & src) {
    static const std::regex re(R"(constant[ \t]+(\w+)[ \t]+(\w+)[ \t]*\[\[[ \t]*function_constant\((.*)\)[ \t]*\]\][ \t]*;)");
    return std::regex_replace(src, re, "constant constexpr $1 $2 = ($1) mega_fc($3);");
}

struct callee {
    std::string expr;           // what to call
    std::vector<param> params;
    bool barrier = true;        // a threadgroup barrier may be inside
    size_t at = std::string::npos; // where its call wrapper goes (right after the instantiation: macros in the
                                   // template arguments may be undefined at the end of the source)
};

// kernels known to have no threadgroup barrier: their threads past the end of a dispatch may sit out
static bool no_barrier(const std::string & host) {
    static const char * pre[] = { "kernel_unary_", "kernel_pad_", "kernel_bin_fuse_", "kernel_cpy_", "kernel_get_rows_",
                                  "kernel_set_rows_", "kernel_softcap_" };
    for (const char * p : pre) {
        if (host.rfind(p, 0) == 0) return true;
    }
    return false;
}

static bool find_callee(const std::string & src, const std::string & host, callee & c) {
    std::string fname;
    c.barrier = !no_barrier(host);
    {
        std::regex re("template[ \\t]*\\[\\[[ \\t]*host_name\\([ \\t]*\"" + host + "\"[ \\t]*\\)[ \\t]*\\]\\][ \\t]*kernel[ \\t]+\\w+[ \\t]+([^;]+);");
        std::smatch m;
        if (std::regex_search(src, m, re)) {
            c.expr = trim(m.str(1));
            fname  = c.expr.substr(0, c.expr.find('<'));
            fname  = trim(fname);
            c.at   = m.position(0) + m.length(0);
        } else {
            c.expr = host;
            fname  = host;
        }
    }
    std::regex re_decl("kernel[ \\t\\n]+void[ \\t\\n]+" + fname + "[ \\t\\n]*\\(");
    std::smatch m;
    if (!std::regex_search(src, m, re_decl)) {
        return false;
    }
    const size_t open = m.position(0) + m.length(0) - 1;
    std::vector<std::string> ps;
    size_t close = 0;
    if (!split_params(src, open, ps, close)) {
        return false;
    }
    int next = 0;
    for (const auto & p0 : ps) {
        const std::string p = trim(p0);
        const size_t a = p.find("[[");
        param q;
        if (a == std::string::npos) {
            q.what = param::BIND;
            q.idx  = next++;
            q.cnst = p.rfind("constant", 0) == 0;
        } else {
            const size_t b = p.find("]]", a);
            const std::string attr = trim(p.substr(a + 2, b - a - 2));
            int n = 0;
            if (sscanf(attr.c_str(), "threadgroup(%d)", &n) == 1) {
                q.what = param::TG;
                q.idx  = n;
            } else if (sscanf(attr.c_str(), "buffer(%d)", &n) == 1) {
                q.what = param::BIND;
                q.idx  = n;
                next   = n + 1;
            } else {
                q.what = param::ATTR;
                q.attr = attr;
            }
        }
        c.params.push_back(q);
    }
    return true;
}

static const char * attr_expr(const std::string & a) {
    static const std::map<std::string, const char *> m = {
        { "threadgroup_position_in_grid",   "mega_u3{c.tgpig}" },
        { "thread_position_in_threadgroup", "mega_u3{c.tpitg}" },
        { "thread_index_in_threadgroup",    "mega_u1{c.tiitg}" },
        { "simdgroup_index_in_threadgroup", "mega_u1{c.sgitg}" },
        { "thread_index_in_simdgroup",      "mega_u1{c.tiisg}" },
        { "threads_per_threadgroup",        "mega_u3{c.ntg}" },
        { "threadgroups_per_grid",          "mega_u3{c.ngrid}" },
        { "thread_position_in_grid",        "mega_u3{c.tgpig*c.ntg + c.tpitg}" },
        { "threads_per_grid",               "mega_u3{c.ngrid*c.ntg}" },
        { "simdgroups_per_threadgroup",     "mega_u1{c.nsg}" },
        { "threads_per_simdgroup",          "mega_u1{32u}" },
        { "thread_execution_width",         "mega_u1{32u}" },
    };
    auto it = m.find(a);
    return it == m.end() ? nullptr : it->second;
}

static const char * k_attrs[] = {
    "threadgroup_position_in_grid", "thread_position_in_threadgroup", "thread_index_in_threadgroup",
    "simdgroup_index_in_threadgroup", "thread_index_in_simdgroup", "threads_per_threadgroup", "threadgroups_per_grid",
    "thread_position_in_grid", "threads_per_grid", "simdgroups_per_threadgroup", "threads_per_simdgroup",
    "thread_execution_width", "dispatch_simdgroups_per_threadgroup", "quadgroup_index_in_threadgroup",
    "thread_index_in_quadgroup",
};

static const char * k_prelude = R"(
// ---- megakernel: the kernels below are device functions ----
struct mega_ctx {
    constant char * argblob;
    constant ggml_metal_mega_entry * e;
    threadgroup char * shmem0;
    threadgroup char * shmem1;
    uint3  tgpig;
    uint3  tpitg;
    uint3  ntg;
    uint3  ngrid;
    uint   tiitg;
    uint   sgitg;
    uint   tiisg;
    uint   nsg;
};
// a binding: converts to the type of the parameter it is passed to
struct mega_bind {
    constant char * a;
    device   char * p;
    template<typename T> operator constant T & () const { return *(constant T *) a; }
    template<typename T> operator constant T * () const { return (constant T *) a; }
    template<typename T> operator device   T * () const { return (device T *) p; }
    template<typename T> operator device const T * () const { return (device const T *) p; }
};
struct mega_tg {
    threadgroup char * p;
    template<typename T> operator threadgroup T * () const { return (threadgroup T *) p; }
};
struct mega_u3 {
    uint3 v;
    operator uint3  () const { return v; }
    operator ushort3() const { return ushort3(v); }
    operator int3   () const { return int3(v); }
    operator uint2  () const { return v.xy; }
    operator ushort2() const { return ushort2(v.xy); }
    operator uint   () const { return v.x; }
    operator ushort () const { return (ushort) v.x; }
    operator int    () const { return (int) v.x; }
};
struct mega_u1 {
    uint v;
    operator uint  () const { return v; }
    operator ushort() const { return (ushort) v; }
    operator int   () const { return (int) v; }
    operator short () const { return (short) v; }
};
static inline mega_bind mega_b(thread const mega_ctx & c, int i) {
    return mega_bind{ c.argblob + c.e->a[i], (device char *) c.e->p[i] };
}
#ifdef GGML_METAL_MEGA_NOINLINE_TEST
#define kernel
#else
#define kernel
#endif
#define host_name(x)
#define buffer(x)
#define threadgroup(x)
)";

static const char * k_kernel = R"(
#undef kernel
#undef host_name
#undef buffer
#undef threadgroup
MEGA_UNDEF_ATTRS

// the next item of a threadgroup: every R-th, or (dyn) the next free one of the step's counter (all threads of the
// threadgroup call it, also the ones that sat out an item)
static inline uint mega_next(uint g, uint R, bool dyn, device atomic_uint * ctr, threadgroup uint * tgn, ushort ti) {
    if (!dyn) {
        return g + R;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ti == 0) {
        *tgn = R + atomic_fetch_add_explicit(ctr, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return *tgn;
}

// the persistent kernel: walks the steps, the threadgroups of a step's dispatches shared out over its own
// threadgroups, a grid barrier between two steps
kernel void kernel_mega(
        constant char                         * argblob [[buffer(0)]],
        constant ggml_metal_mega_entry    * E       [[buffer(1)]],
        constant ggml_metal_mega_step     * S       [[buffer(2)]],
        device atomic_uint                    * bar     [[buffer(3)]],
        constant ggml_metal_mega_params       & P       [[buffer(4)]],
        threadgroup char                      * shmem   [[threadgroup(0)]],
        uint   tg [[threadgroup_position_in_grid]],
        uint   R  [[threadgroups_per_grid]],
        ushort ti [[thread_index_in_threadgroup]]) {
    // a wait that timed out (a threadgroup that never started: the GPU is shared) ends the whole dispatch at the
    // next barrier, so that it never holds the GPU for more than one wait
    threadgroup uint aborted;
    threadgroup uint next_item;
    if ((P.pad[1] & 0xff) == 1) {
        return; // GGML_METAL_MEGA_STOP=1: the dispatch alone
    }
    for (uint s = 0; s < P.n_steps; ++s) {
        const uint e0 = S[s].e0;
        const uint ni = S[s].n_items;
        const bool work = (P.pad[1] & 0xff) != 2 && (P.pad[2] == 0 || s + 1 < P.pad[2]); // STOP=2: steps and barriers only; UPTO
        // STOP=4: each threadgroup only its first item of a step; STOP=5: only item 0, on threadgroup 0
        const uint g_end = (P.pad[1] & 0xff) == 4 ? min(ni, tg + 1) : (P.pad[1] & 0xff) == 5 ? (tg == 0 ? min(ni, 1u) : 0u) : ni;
        // GGML_METAL_MEGA_DYNAMIC: after its first item a threadgroup takes the next free one from a counter of the
        // step (instead of every R-th), so that one that finishes early does more
        const bool dyn = (P.pad[1] & 0x400) != 0;
        for (uint g = tg; g < g_end && work; g = mega_next(g, R, dyn, bar + 64 + s, &next_item, ti)) {
            uint e  = e0;
            uint gi = g;
            while (gi >= (uint) E[e].n_items) { gi -= E[e].n_items; ++e; }
            constant ggml_metal_mega_entry * en = E + e;
            const uint w    = en->width;
            const uint lane = ti/w;
            if (lane >= (uint) en->lanes) {
                continue;
            }
            uint v = gi*en->lanes + lane;
            if (v >= (uint) en->n_vtg) {
                if (en->idle_ok) {
                    continue;
                }
                v = en->n_vtg - 1;
            }
            if ((P.pad[1] & 0xff) == 6) {
                v = 0; // STOP=6: the same work, every item on the data of threadgroup 0 (cache-resident)
            }
            const uint tt = en->tptg[0]*en->tptg[1]*en->tptg[2]; // threads the kernel sees
            for (uint rp = 0; rp < (uint) en->reps; ++rp) {
            const uint lt = ti - lane*w + rp*w;
            if (lt >= tt) {
                break;
            }
            mega_ctx c;
            c.argblob = argblob;
            c.e       = en;
            c.shmem0  = shmem + lane*en->smem_lane;
            c.shmem1  = c.shmem0 + en->smem0;
            c.ngrid   = uint3(en->grid[0], en->grid[1], en->grid[2]);
            c.tgpig   = uint3(v % c.ngrid.x, (v/c.ngrid.x) % c.ngrid.y, v/(c.ngrid.x*c.ngrid.y));
            c.ntg     = uint3(en->tptg[0], en->tptg[1], en->tptg[2]);
            c.tpitg   = uint3(lt % c.ntg.x, (lt/c.ntg.x) % c.ntg.y, lt/(c.ntg.x*c.ntg.y));
            c.tiitg   = lt;
            c.sgitg   = lt/32;
            c.tiisg   = lt%32;
            c.nsg     = (tt + 31)/32;
            if ((P.pad[1] & 0xff) != 3) switch (en->pipe) { // STOP=3: no kernel call
MEGA_CASES
                default: break;
            }
            }
        }
        if (s + 1 < P.n_steps) {
            threadgroup_barrier(mem_flags::mem_device);
            if (!(P.pad[1] & 0x200)) atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); // GGML_METAL_MEGA_NOFENCE: timing only
            if (ti == 0 && (P.pad[1] & 0x100)) {
                // GGML_METAL_MEGA_FLAGBAR: the last to arrive raises a flag (on a cache line of its own, written
                // once a step), the others watch only the flag instead of the counter everyone is adding to
                const uint old  = atomic_fetch_add_explicit(bar, 1u, memory_order_relaxed);
                const uint lim  = P.pad[0] ? P.pad[0] : (1u << 22);
                uint it = 0;
                aborted = 0;
                if (old + 1 == (s + 1)*R) {
                    atomic_store_explicit(bar + 32, s + 1, memory_order_relaxed);
                } else {
                    while (atomic_load_explicit(bar + 32, memory_order_relaxed) < s + 1) {
                        if (atomic_load_explicit(bar + 1, memory_order_relaxed) != 0 || ++it > lim) {
                            atomic_fetch_add_explicit(bar + 1, 1u, memory_order_relaxed);
                            aborted = 1;
                            break;
                        }
                    }
                }
            } else if (ti == 0) {
                atomic_fetch_add_explicit(bar, 1u, memory_order_relaxed);
                const uint want = (s + 1)*R;
                const uint lim  = P.pad[0] ? P.pad[0] : (1u << 22);
                uint it = 0;
                aborted = 0;
                while (atomic_load_explicit(bar, memory_order_relaxed) < want) {
                    if (atomic_load_explicit(bar + 1, memory_order_relaxed) != 0 || ++it > lim) {
                        atomic_fetch_add_explicit(bar + 1, 1u, memory_order_relaxed);
                        aborted = 1;
                        break;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_device);
            if (!(P.pad[1] & 0x200)) atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); // GGML_METAL_MEGA_NOFENCE: timing only
            if (aborted) {
                return;
            }
        }
    }
}

// the same code as the persistent kernel runs for an item, launched as an ordinary dispatch: a threadgroup an item of
// one entry (GGML_METAL_MEGA_ONESHOT, to tell the code it generates from the way the persistent kernel runs it)
kernel void kernel_mega_one(
        constant char                         * argblob [[buffer(0)]],
        constant ggml_metal_mega_entry    * en      [[buffer(1)]],
        threadgroup char                      * shmem   [[threadgroup(0)]],
        uint   gi [[threadgroup_position_in_grid]],
        ushort ti [[thread_index_in_threadgroup]]) {
    if (en->pad0 == 2) {
        return; // GGML_METAL_MEGA_ONE_STOP=2: nothing
    }
    const uint w    = en->width;
    const uint lane = ti/w;
    if (lane >= (uint) en->lanes) {
        return;
    }
    uint v = gi*en->lanes + lane;
    if (v >= (uint) en->n_vtg) {
        if (en->idle_ok) {
            return;
        }
        v = en->n_vtg - 1;
    }
    const uint tt = en->tptg[0]*en->tptg[1]*en->tptg[2];
    for (uint rp = 0; rp < (uint) en->reps; ++rp) {
        const uint lt = ti - lane*w + rp*w;
        if (lt >= tt) {
            break;
        }
        mega_ctx c;
        c.argblob = argblob;
        c.e       = en;
        c.shmem0  = shmem + lane*en->smem_lane;
        c.shmem1  = c.shmem0 + en->smem0;
        c.ngrid   = uint3(en->grid[0], en->grid[1], en->grid[2]);
        c.tgpig   = uint3(v % c.ngrid.x, (v/c.ngrid.x) % c.ngrid.y, v/(c.ngrid.x*c.ngrid.y));
        c.ntg     = uint3(en->tptg[0], en->tptg[1], en->tptg[2]);
        c.tpitg   = uint3(lt % c.ntg.x, (lt/c.ntg.x) % c.ntg.y, lt/(c.ntg.x*c.ntg.y));
        c.tiitg   = lt;
        c.sgitg   = lt/32;
        c.tiisg   = lt%32;
        c.nsg     = (tt + 31)/32;
        if (en->pad0 == 1) {
            if (c.tgpig.x == 0xffffffffu && (device char *) mega_b(c, 0).p == nullptr) shmem[0] = 0; // keep the prologue
            return; // GGML_METAL_MEGA_ONE_STOP=1: the prologue only
        }
        switch (en->pipe) {
MEGA_CASES
            default: break;
        }
    }
}
)";

static std::string replace_all(std::string s, const std::string & a, const std::string & b) {
    size_t p = 0;
    while ((p = s.find(a, p)) != std::string::npos) { s.replace(p, a.size(), b); p += b.size(); }
    return s;
}

// the call of each case, filled by ggml_metal_mega_source (for the baked schedule)
static std::vector<std::string> g_case_call;

// the kernel of each baked segment (kernel_seg_<n>)
static std::vector<int> g_seg_fn;

// the schedule ggml_metal_mega_source_baked writes out (null: none)
static struct {
    const ggml_metal_mega_sched_entry * ents;
    const int * step_e0;
    const int * seg_s0;
    int n_segs;
} g_sched = {};

// a grid barrier between two steps of a baked segment kernel (as in kernel_mega)
static const char * k_seg_barrier = R"(
#define MEGA_SEG_BARRIER(S) { \
    threadgroup_barrier(mem_flags::mem_device); \
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); \
    if (ti == 0) { \
        atomic_fetch_add_explicit(bar, 1u, memory_order_relaxed); \
        const uint want = (S)*R; \
        const uint lim  = P.pad[0] ? P.pad[0] : (1u << 22); \
        uint it = 0; \
        aborted = 0; \
        while (atomic_load_explicit(bar, memory_order_relaxed) < want) { \
            if (atomic_load_explicit(bar + 1, memory_order_relaxed) != 0 || ++it > lim) { \
                atomic_fetch_add_explicit(bar + 1, 1u, memory_order_relaxed); \
                aborted = 1; \
                break; \
            } \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_device); \
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); \
    if (aborted) { return; } \
}
// the same barrier in two halves, with work between them that does not need the other threadgroups (the prefetch of
// GGML_METAL_MEGA_PF): arrive, then wait
#define MEGA_SEG_ARRIVE() { \
    threadgroup_barrier(mem_flags::mem_device); \
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); \
    if (ti == 0) { atomic_fetch_add_explicit(bar, 1u, memory_order_relaxed); } \
}
#define MEGA_SEG_WAIT(S) { \
    if (ti == 0) { \
        const uint want = (S)*R; \
        const uint lim  = P.pad[0] ? P.pad[0] : (1u << 22); \
        uint it = 0; \
        aborted = 0; \
        while (atomic_load_explicit(bar, memory_order_relaxed) < want) { \
            if (atomic_load_explicit(bar + 1, memory_order_relaxed) != 0 || ++it > lim) { \
                atomic_fetch_add_explicit(bar + 1, 1u, memory_order_relaxed); \
                aborted = 1; \
                break; \
            } \
        } \
    } \
    threadgroup_barrier(mem_flags::mem_device); \
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device); \
    if (aborted) { return; } \
}
// touch the lines [b0, b1) of w: a load a 128-byte line, spread over the threads (the values only feed a store that
// never happens, so the loads are kept and nothing waits on them but the end of the prefetch)
#define MEGA_PF_LINE 128u
#ifdef MEGA_NOBAR
#undef MEGA_SEG_BARRIER
#define MEGA_SEG_BARRIER(S)
#undef MEGA_SEG_ARRIVE
#define MEGA_SEG_ARRIVE()
#undef MEGA_SEG_WAIT
#define MEGA_SEG_WAIT(S)
#endif
#ifdef MEGA_NOCALL
#define MEGA_CALL(x)
#else
#define MEGA_CALL(x) x
#endif
)";

// the prefetch for step s, before the barrier that opens it: every threadgroup loads the weight lines of its own items of
// the step (in the order it will run them), up to pf_max bytes
static std::string seg_prefetch(const ggml_metal_mega_sched_entry * ents, const int * step_e0, int s, int e_base,
                                uint64_t pf_max) {
    std::string k;
    bool any = false;
    int off = 0;
    for (int ei = step_e0[s]; ei < step_e0[s + 1]; ++ei) {
        const auto & e = ents[ei];
        if (e.pf_bind >= 0 && e.pf_slice > 0) {
            if (!any) {
                k += "    {\n        uint pf_acc = 0;\n        ulong pf_spent = 0;\n";
                any = true;
            }
            char buf[2048];
            snprintf(buf, sizeof(buf),
                "        {\n"
                "            device const uchar * w = (device const uchar *) E[%d].p[%d];\n"
                "            for (uint gi = (tg + R - (%du %% R)) %% R; gi < %du && pf_spent < %lluul; gi += R) {\n"
                "                const ulong b0 = (ulong) gi*%lluul;\n"
                "                const ulong b1 = min(b0 + %lluul, %lluul);\n"
                "                for (ulong o = b0 + (ulong) ti*MEGA_PF_LINE; o < b1; o += (ulong) T_*MEGA_PF_LINE) {\n"
                "                    pf_acc ^= *(device const uint *) (w + o);\n"
                "                }\n"
                "                pf_spent += b1 > b0 ? b1 - b0 : 0;\n"
                "            }\n"
                "        }\n",
                ei - e_base, e.pf_bind, off, e.n_items, (unsigned long long) pf_max,
                (unsigned long long) (e.pf_slice*e.vl), (unsigned long long) (e.pf_slice*e.vl),
                (unsigned long long) e.pf_total);
            k += buf;
        }
        off += e.n_items;
    }
    if (any) {
        k += "        if (pf_acc == 0x9e3779b9u && ti == 7) { atomic_fetch_add_explicit(bar + 2, 0u, memory_order_relaxed); }\n    }\n";
    }
    return k;
}

// a baked segment: its steps written out, every count and size a constant; the table E has only the bindings
static std::string seg_kernel(int si, const ggml_metal_mega_sched_entry * ents, const int * step_e0, int s0, int s1,
                              const std::vector<int> & case_of_pipe, const int * pipe_case_ok) {
    (void) pipe_case_ok;
    std::string k;
    k += "kernel void kernel_seg_" + std::to_string(si) + "(\n"
         "        constant char                      * argblob [[buffer(0)]],\n"
         "        constant ggml_metal_mega_entry     * E       [[buffer(1)]],\n"
         "        device atomic_uint                 * bar     [[buffer(3)]],\n"
         "        constant ggml_metal_mega_params    & P       [[buffer(4)]],\n"
         "        threadgroup char                   * shmem   [[threadgroup(0)]],\n"
         "        uint   tg [[threadgroup_position_in_grid]],\n"
         "        uint   R  [[threadgroups_per_grid]],\n"
         "        ushort ti [[thread_index_in_threadgroup]],\n"
         "        ushort T_ [[threads_per_threadgroup]]) {\n"
         "    threadgroup uint aborted;\n"
         "    uint pf_sink = 0;\n";
    // GGML_METAL_MEGA_SEGRET: every segment kernel returns at once (timing the launches alone)
    if (getenv("GGML_METAL_MEGA_SEGRET")) k += "    return;\n";
    static const bool pfn = getenv("GGML_METAL_MEGA_PFN") != nullptr;
    static const int  T   = getenv("GGML_METAL_MEGA_TPT") ? atoi(getenv("GGML_METAL_MEGA_TPT")) : 256;
    static const uint64_t pf_max = getenv("GGML_METAL_MEGA_PF") ? strtoull(getenv("GGML_METAL_MEGA_PF"), nullptr, 0) : 0;
    const int e_base = step_e0[s0];
    for (int s = s0; s < s1; ++s) {
        if (s > s0) {
            const std::string pf = pf_max ? seg_prefetch(ents, step_e0, s, e_base, pf_max) : std::string();
            if (pf.empty()) {
                k += "    MEGA_SEG_BARRIER(" + std::to_string(s - s0) + ")\n";
            } else {
                k += "    MEGA_SEG_ARRIVE()\n" + pf + "    MEGA_SEG_WAIT(" + std::to_string(s - s0) + ")\n";
            }
        }
        int off = 0; // items of the step before this entry
        for (int ei = step_e0[s]; ei < step_e0[s + 1]; ++ei) {
            const auto & e = ents[ei];
            const int tt = e.tptg[0]*e.tptg[1]*e.tptg[2];
            char buf[4096];
            snprintf(buf, sizeof(buf),
                "    for (uint gi = (tg + R - (%du %% R)) %% R; gi < %du; gi += R) {\n"
                "        const uint lane = ti/%du;\n"
                "        if (lane < %du) {\n"
                "            uint v = gi*%du + lane %% %du;\n"
                "            bool run = %s;\n"
                "            if (v >= %du) { %s }\n"
                "            for (uint rp = 0; run && rp < %du; ++rp) {\n"
                "                const uint lt = ti - lane*%du + rp*%du;\n"
                "                if (lt >= %du) { break; }\n"
                "                mega_ctx c;\n"
                "                c.argblob = argblob;\n"
                "                c.e       = E + %d;\n"
                "                c.shmem0  = shmem + lane*%du;\n"
                "                c.shmem1  = c.shmem0 + %du;\n"
                "                c.ngrid   = uint3(%du, %du, %du);\n"
                "                c.tgpig   = uint3(v %% %du, (v/%du) %% %du, v/%du);\n"
                "                c.ntg     = uint3(%du, %du, %du);\n"
                "                c.tpitg   = uint3(lt %% %du, (lt/%du) %% %du, lt/%du);\n"
                "                c.tiitg   = lt;\n"
                "                c.sgitg   = lt/32;\n"
                "                c.tiisg   = lt%%32;\n"
                "                c.nsg     = %du;\n"
                "                MEGA_CALL(%s(c));\n"
                "            }\n"
                "        }\n"
                "    }\n",
                off, e.n_items, e.width, e.lanes, e.vl, e.vl,
                (e.idle_ok && e.vl < e.lanes) ? ("lane < " + std::to_string(e.vl) + "u").c_str() : "true",
                e.n_vtg,
                e.idle_ok ? "run = false;" : ("v = " + std::to_string(e.n_vtg - 1) + "u;").c_str(),
                e.reps, e.width, e.width, tt, ei - e_base, e.smem_lane, e.smem0,
                e.grid[0], e.grid[1], e.grid[2], e.grid[0], e.grid[0], e.grid[1], e.grid[0]*e.grid[1],
                e.tptg[0], e.tptg[1], e.tptg[2], e.tptg[0], e.tptg[0], e.tptg[1], e.tptg[0]*e.tptg[1],
                (tt + 31)/32, g_case_call[case_of_pipe[e.pipe]].c_str());
            // GGML_METAL_MEGA_PFN: before an item, load the lines of the next item of this threadgroup (gi + R) into
            // registers that are only used after the item: the loads are in flight while it computes
            std::string item = buf;
            if (pfn && e.pf_bind >= 0 && e.pf_slice > 0 && e.n_items > 1) {
                const uint64_t ib = e.pf_slice*e.vl; // bytes an item
                int nl = (int) ((ib + 128ull*T - 1)/(128ull*T));
                if (nl > 8) nl = 8;
                std::string pre = "        uint pfr[" + std::to_string(nl) + "] = {};\n"
                                  "        if (gi + R < " + std::to_string(e.n_items) + "u) {\n"
                                  "            device const uchar * w = (device const uchar *) E[" + std::to_string(ei - e_base) + "].p[" + std::to_string(e.pf_bind) + "];\n"
                                  "            const ulong b0 = (ulong) (gi + R)*" + std::to_string(ib) + "ul;\n"
                                  "            const ulong b1 = min(b0 + " + std::to_string(ib) + "ul, " + std::to_string(e.pf_total) + "ul);\n";
                for (int j = 0; j < nl; ++j) {
                    pre += "            { const ulong o = b0 + (ulong) (ti + " + std::to_string(j*T) + "u)*128ul; if (o < b1) pfr[" +
                           std::to_string(j) + "] = *(device const uint *) (w + o); }\n";
                }
                pre += "        }\n";
                std::string post = "        pf_sink ^= pfr[0]";
                for (int j = 1; j < nl; ++j) post += " ^ pfr[" + std::to_string(j) + "]";
                post += ";\n";
                const size_t a = item.find('\n') + 1;          // after the loop header
                item.insert(a, pre);
                const size_t z = item.rfind("    }\n");          // before the end of the loop
                item.insert(z, post);
            }
            k += item;
            off += e.n_items;
        }
    }
    k += "    if (pf_sink == 0x9e3779b9u && ti == 7) { atomic_fetch_add_explicit(bar + 2, 0u, memory_order_relaxed); }\n";
    k += "}\n";
    return k;
}

} // namespace

char * ggml_metal_mega_source(const char * hdr, const char * const * kind_srcs, int n_kinds,
                              const struct ggml_metal_mega_pipe * pipes, int n_pipes, int * out_case, int * out_flags) {
    std::string src;
    g_case_call.clear();
    src += "#include <metal_stdlib>\nusing namespace metal;\n";
    // GGML_METAL_MEGA_NOCOH: plain pointers (results are wrong: a threadgroup can read a stale line of what another
    // wrote in the step before; for timing the cost of the coherent loads only)
    // GGML_METAL_MEGA_COHSTORE: only the pointers a kernel can write through are coherent; device const pointers are
    // plain, cached loads. measured (stepbench/coh3.m): plain stores leave a core with a line that has its own part new
    // and other cores' parts old (rows written by several threadgroups share lines), coherent stores do not, and plain
    // loads are then right as long as nothing was read or written at those addresses earlier in the dispatch
    // (GGML_ALLOC_NO_REUSE, and the stale-line cut of ggml_metal_mega_stale)
    const bool cohstore = getenv("GGML_METAL_MEGA_COHSTORE") != nullptr;
    if (!getenv("GGML_METAL_MEGA_NOCOH") && !cohstore) {
        src += "#define device coherent(device) device\n";
    }
    src += cohstore ? coherent_stores(hdr) : std::string(hdr);
    src += cohstore ? coherent_stores(k_prelude) : std::string(k_prelude);
    for (const char * a : k_attrs) {
        src += std::string("#define ") + a + "\n";
    }

    // a namespace for each kind and set of function constants
    std::map<std::string, int> ns_of;     // key -> namespace number
    std::vector<int> pipe_ns(n_pipes, -1);
    std::vector<std::string> ns_src;      // the processed kind source of each namespace (before the wrappers)
    std::vector<std::string> ns_fcs;
    std::vector<int> ns_kind;
    std::vector<std::string> kind_proc(n_kinds);

    for (int i = 0; i < n_pipes; ++i) {
        out_case[i]  = -1;
        out_flags[i] = 0;
        const auto & p = pipes[i];
        if (p.kind < 0 || p.kind >= n_kinds || !kind_srcs[p.kind]) continue;
        std::string key = std::to_string(p.kind) + ":";
        std::string fcs;
        for (int j = 0; j < p.n_fc; ++j) {
            char buf[64];
            snprintf(buf, sizeof(buf), "        case %d: return %lld;\n", p.fc_idx[j], (long long) p.fc_val[j]);
            fcs += buf;
            key += std::to_string(p.fc_idx[j]) + "=" + std::to_string((long long) p.fc_val[j]) + ",";
        }
        auto it = ns_of.find(key);
        if (it == ns_of.end()) {
            const int n = (int) ns_src.size();
            ns_of[key] = n;
            ns_fcs.push_back(fcs);
            ns_kind.push_back(p.kind);
            if (kind_proc[p.kind].empty()) {
                kind_proc[p.kind] = rewrite_fc(drop_kernels(kind_srcs[p.kind]));
                // kernel_depth_mix keeps n_embd/threads values a thread in yv[4] (3 with its 1024 threads); the persistent
                // kernel runs it with 256: 9 values, past the end of the array (garbage in a small library)
                kind_proc[p.kind] = replace_all(kind_proc[p.kind], "float yv[4];", "float yv[16];");
                if (getenv("GGML_METAL_MEGA_COHSTORE")) {
                    kind_proc[p.kind] = coherent_stores(kind_proc[p.kind]);
                }
            }
            ns_src.push_back(kind_proc[p.kind]);
            pipe_ns[i] = n;
        } else {
            pipe_ns[i] = it->second;
        }
    }

    std::vector<std::string> ns_wrap(ns_src.size());
    std::vector<std::vector<std::pair<size_t, std::string>>> ns_ins(ns_src.size()); // wrappers placed inside the source
    std::string cases;
    int n_case = 0;
    for (int i = 0; i < n_pipes; ++i) {
        if (pipe_ns[i] < 0) continue;
        callee c;
        if (!find_callee(ns_src[pipe_ns[i]], pipes[i].base, c)) {
            fprintf(stderr, "ggml_metal_mega: no kernel '%s' in its source\n", pipes[i].base);
            continue;
        }
        std::string args;
        bool ok = true;
        for (const auto & q : c.params) {
            if (!args.empty()) args += ", ";
            if (q.what == param::BIND) {
                args += "mega_b(c, " + std::to_string(q.idx) + ")";
            } else if (q.what == param::TG) {
                args += q.idx == 0 ? "mega_tg{c.shmem0}" : "mega_tg{c.shmem1}";
            } else {
                const char * e = attr_expr(q.attr);
                if (!e) { ok = false; break; }
                args += e;
            }
        }
        if (!ok) {
            fprintf(stderr, "ggml_metal_mega: '%s' has a parameter attribute it cannot supply\n", pipes[i].base);
            continue;
        }
        const int k = n_case++;
        const std::string w = "\nstatic inline void mega_call_" + std::to_string(k) + "(thread const mega_ctx & c) {\n    " +
                              c.expr + "(" + args + ");\n}\n";
        if (c.at != std::string::npos) {
            ns_ins[pipe_ns[i]].push_back({ c.at, w });
        } else {
            ns_wrap[pipe_ns[i]] += w;
        }
        cases += "                case " + std::to_string(k) + ": mega_ns" + std::to_string(pipe_ns[i]) + "::mega_call_" +
                 std::to_string(k) + "(c); break;\n";
        g_case_call.push_back("mega_ns" + std::to_string(pipe_ns[i]) + "::mega_call_" + std::to_string(k));
        out_case[i]  = k;
        int cmask = 0;
        for (const auto & q : c.params) {
            if (q.what == param::BIND && q.cnst && q.idx < 20) cmask |= 1 << q.idx;
        }
        out_flags[i] = GGML_METAL_MEGA_FLAG_OK | (c.barrier ? 0 : GGML_METAL_MEGA_FLAG_NO_BARRIER) |
                       (cmask << GGML_METAL_MEGA_FLAG_CONST_SHIFT);
    }

    for (size_t n = 0; n < ns_src.size(); ++n) {
        src += "namespace mega_ns" + std::to_string(n) + " {\n";
        src += "constexpr int64_t mega_fc(int i) {\n    switch (i) {\n" + ns_fcs[n] + "        default: return 0;\n    }\n}\n";
        std::string body = ns_src[n];
        auto & ins = ns_ins[n];
        std::sort(ins.begin(), ins.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        for (const auto & x : ins) {
            body.insert(x.first, x.second);
        }
        src += body;
        src += "\n" + ns_wrap[n];
        src += "} // namespace mega_ns" + std::to_string(n) + "\n";
    }

    std::string undef;
    for (const char * a : k_attrs) {
        undef += std::string("#undef ") + a + "\n";
    }
    std::string kern = replace_all(cohstore ? coherent_stores(k_kernel) : std::string(k_kernel), "MEGA_UNDEF_ATTRS", undef);
    kern = replace_all(kern, "MEGA_CASES\n", cases);
    src += kern;

    bool sched_ok = g_sched.ents != nullptr;
    for (int s = 0; sched_ok && s < g_sched.seg_s0[g_sched.n_segs]; ++s) {
        for (int e = g_sched.step_e0[s]; e < g_sched.step_e0[s + 1]; ++e) {
            if (out_case[g_sched.ents[e].pipe] < 0) sched_ok = false;
        }
    }
    if (sched_ok) {
        // the baked segments
        // GGML_METAL_MEGA_NOCALL: the segments without their kernel calls (timing the frame alone)
        std::string segs = getenv("GGML_METAL_MEGA_NOCALL") ? std::string("#define MEGA_NOCALL\n") + k_seg_barrier : std::string(k_seg_barrier);
        std::vector<int> case_of_pipe(n_pipes);
        for (int i = 0; i < n_pipes; ++i) case_of_pipe[i] = out_case[i];
        // segments with the same code (the same steps of another layer: only their part of E differs) share a kernel
        std::map<std::string, int> uniq;
        g_seg_fn.assign(g_sched.n_segs, -1);
        for (int si = 0; si < g_sched.n_segs; ++si) {
            std::string body = seg_kernel(0, g_sched.ents, g_sched.step_e0, g_sched.seg_s0[si], g_sched.seg_s0[si + 1], case_of_pipe, nullptr);
            auto it = uniq.find(body);
            if (it != uniq.end()) {
                g_seg_fn[si] = it->second;
                continue;
            }
            const int fi = (int) uniq.size();
            uniq[body] = fi;
            g_seg_fn[si] = fi;
            segs += replace_all(body, "kernel void kernel_seg_0(", "kernel void kernel_seg_" + std::to_string(fi) + "(");
        }
        src += cohstore ? coherent_stores(segs) : segs;
    }

    char * res = (char *) malloc(src.size() + 1);
    memcpy(res, src.c_str(), src.size() + 1);
    return res;
}

char * ggml_metal_mega_source_baked(const char * hdr, const char * const * kind_srcs, int n_kinds,
                                    const struct ggml_metal_mega_pipe * pipes, int n_pipes, int * out_case, int * out_flags,
                                    const struct ggml_metal_mega_sched_entry * ents, const int * step_e0,
                                    const int * seg_s0, int n_segs) {
    g_sched.ents    = ents;
    g_sched.step_e0 = step_e0;
    g_sched.seg_s0  = seg_s0;
    g_sched.n_segs  = n_segs;
    char * res = ggml_metal_mega_source(hdr, kind_srcs, n_kinds, pipes, n_pipes, out_case, out_flags);
    g_sched = {};
    // every entry needs a case
    for (int s = 0; res && s < seg_s0[n_segs]; ++s) {
        for (int e = step_e0[s]; e < step_e0[s + 1]; ++e) {
            if (out_case[ents[e].pipe] < 0) { free(res); return nullptr; }
        }
    }
    return res;
}

int ggml_metal_mega_seg_fn(int seg) {
    return seg >= 0 && seg < (int) g_seg_fn.size() ? g_seg_fn[seg] : -1;
}
