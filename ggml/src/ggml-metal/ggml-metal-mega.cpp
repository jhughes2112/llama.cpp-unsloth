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
    if (P.pad[1] == 1) {
        return; // GGML_METAL_MEGA_STOP=1: the dispatch alone
    }
    for (uint s = 0; s < P.n_steps; ++s) {
        const uint e0 = S[s].e0;
        const uint ni = S[s].n_items;
        const bool work = P.pad[1] != 2 && (P.pad[2] == 0 || s + 1 < P.pad[2]); // STOP=2: steps and barriers only; UPTO
        // STOP=4: each threadgroup only its first item of a step; STOP=5: only item 0, on threadgroup 0
        const uint g_end = P.pad[1] == 4 ? min(ni, tg + 1) : P.pad[1] == 5 ? (tg == 0 ? min(ni, 1u) : 0u) : ni;
        for (uint g = tg; g < g_end && work; g += R) {
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
            if (P.pad[1] == 6) {
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
            if (P.pad[1] != 3) switch (en->pipe) { // STOP=3: no kernel call
MEGA_CASES
                default: break;
            }
            }
        }
        if (s + 1 < P.n_steps) {
            threadgroup_barrier(mem_flags::mem_device);
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
            if (ti == 0) {
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
            atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
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

} // namespace

char * ggml_metal_mega_source(const char * hdr, const char * const * kind_srcs, int n_kinds,
                              const struct ggml_metal_mega_pipe * pipes, int n_pipes, int * out_case, int * out_flags) {
    std::string src;
    src += "#include <metal_stdlib>\nusing namespace metal;\n";
    // GGML_METAL_MEGA_NOCOH: plain pointers (results are wrong: a threadgroup can read a stale line of what another
    // wrote in the step before; for timing the cost of the coherent loads only)
    if (!getenv("GGML_METAL_MEGA_NOCOH")) {
        src += "#define device coherent(device) device\n";
    }
    src += hdr;
    src += k_prelude;
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
    std::string kern = replace_all(k_kernel, "MEGA_UNDEF_ATTRS", undef);
    kern = replace_all(kern, "MEGA_CASES\n", cases);
    src += kern;

    char * res = (char *) malloc(src.size() + 1);
    memcpy(res, src.c_str(), src.size() + 1);
    return res;
}
