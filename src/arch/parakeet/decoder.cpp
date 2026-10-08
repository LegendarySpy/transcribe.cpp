// arch/parakeet/decoder.cpp - Parakeet TDT decoder implementation.
//
// See decoder.h for the API contract. The predictor LSTM, encoder
// projection, and joint network all run as ggml backend graphs on a
// single shared CPU backend + persistent threadpool (owned by PredGraph).
// Decode weights are uploaded to resident ggml tensors at load and the
// host fp32 mirrors then freed.
//
// Numerical note: ggml's reduction order differs slightly from a naive
// host loop, so per-step values drift ~1e-7 vs the reference; over an
// utterance this stays within ~1e-4 and the greedy transcript is
// unchanged.

#include "decoder.h"

#include "parakeet.h"
#include "transcribe-batch-util.h"
#include "transcribe-debug.h"
#include "transcribe-log.h"
#include "transcribe-repetition-guard.h"
#include "weights.h"

// ggml-backend.h, not ggml-cpu.h: under GGML_BACKEND_DL the CPU backend
// is a loadable module, so backend-specific entry points must be reached
// through the registry (ggml_backend_init_by_type / get_proc_address),
// never by direct link-time reference.
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"  // ggml_threadpool_params_default (rest via registry)
#include "ggml.h"

#include <thread>

#if TRANSCRIBE_HAS_BLAS
#    ifdef __APPLE__
#        include <Accelerate/Accelerate.h>
#    else
#        include <cblas.h>
#    endif
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace transcribe::parakeet {

// ---------------------------------------------------------------------------
// Host weight extraction
// ---------------------------------------------------------------------------

namespace {

// Read all elements of a ggml_tensor into a host fp32 vector,
// dequantizing via ggml_get_type_traits()->to_float. Source dtype
// support is whatever ggml registers a to_float for (F32, F16, BF16, all
// Q*/IQ*). ggml_backend_tensor_get is universal (memcpy on host buffers,
// readback on discrete GPUs); we stage the raw bytes then materialize
// fp32 in `out`. Returns false on missing-trait / size errors.
bool read_tensor_to_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (t == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: read_tensor_to_f32: null tensor");
        return false;
    }
    const size_t nbytes = ggml_nbytes(t);
    if (nbytes == 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: tensor \"%s\" has 0 bytes", t->name);
        return false;
    }
    const int64_t nelem = ggml_nelements(t);
    if (nelem <= 0) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: tensor \"%s\" has nelem=%lld", t->name,
                static_cast<long long>(nelem));
        return false;
    }

    out.resize(static_cast<size_t>(nelem));

    // F32 fast path. type_traits[GGML_TYPE_F32] has no to_float (identity
    // is left null upstream), so short-circuit rather than dispatch.
    if (t->type == GGML_TYPE_F32) {
        if (nbytes != static_cast<size_t>(nelem) * sizeof(float)) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "parakeet decoder: tensor \"%s\" f32 nbytes %zu "
                    "!= nelem*sizeof(float) %zu",
                    t->name, nbytes, static_cast<size_t>(nelem) * sizeof(float));
            return false;
        }
        ggml_backend_tensor_get(t, out.data(), 0, nbytes);
        return true;
    }

    // Quantized / non-fp32 path: stage the raw bytes off the backend,
    // then walk the type's to_float to materialize fp32.
    const auto * tt = ggml_get_type_traits(t->type);
    if (tt == nullptr || tt->to_float == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: tensor \"%s\" type %s has no to_float", t->name,
                ggml_type_name(t->type));
        return false;
    }

    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    tt->to_float(raw.data(), out.data(), nelem);
    return true;
}

// Per-step GEMV weights keep a quantized GGUF type. The n=1 matmuls are
// memory-bound, and the quantized dot reads the GGUF's exact values at a
// fraction of the fp32 bytes; the step input is quantized to match.
// Cache-aware streaming models (Nemotron) stay fp32: Q8_0 step weights
// flipped Danish and Estonian words on Nemotron 3.5.
ggml_type step_weight_type(const ggml_tensor * src, bool keep_quantized) {
    return keep_quantized && ggml_is_quantized(src->type) ? src->type : GGML_TYPE_F32;
}

// Upload a step weight: the model tensor's raw bytes when the types match,
// else the fp32 values in `host`.
void set_step_weight(ggml_tensor * dst, const ggml_tensor * src, const std::vector<float> & host) {
    if (dst->type == src->type) {
        std::vector<uint8_t> raw(ggml_nbytes(src));
        ggml_backend_tensor_get(src, raw.data(), 0, raw.size());
        ggml_backend_tensor_set(dst, raw.data(), 0, raw.size());
    } else {
        ggml_backend_tensor_set(dst, host.data(), 0, host.size() * sizeof(float));
    }
}

// Make the joint network's weights resident as ggml tensors on the model:
// enc / pred / out projections. Built once at load. The per-step pred and
// out projections keep a quantized source type (step_weight_type); the rest
// are fp32 from the host mirrors, freed here once uploaded. On failure frees
// partial state and returns false (w_ready stays false → hard decode error).
bool build_joint_weight(HostJoint &         j,
                        const ggml_tensor * src_pred_w,
                        const ggml_tensor * src_out_w,
                        bool                keep_quantized) {
    const int joint_h = j.joint_h;
    const int joint_n = j.joint_n;

    auto fail = [&]() -> bool {
        if (j.w_buf != nullptr) {
            safe_buffer_free(j.w_buf);
            j.w_buf = nullptr;
        }
        if (j.w_ctx != nullptr) {
            ggml_free(j.w_ctx);
            j.w_ctx = nullptr;
        }
        if (j.w_backend != nullptr) {
            safe_backend_free(j.w_backend);
            j.w_backend = nullptr;
        }
        j.g_enc_w  = nullptr;
        j.g_enc_b  = nullptr;
        j.g_pred_w = nullptr;
        j.g_pred_b = nullptr;
        j.gw_w     = nullptr;
        j.gw_b     = nullptr;
        j.w_ready  = false;
        return false;
    };

    // CPU backend used ONLY to allocate the resident weight buffer —
    // never graph_compute'd, so it carries no per-step state.
    j.w_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (j.w_backend == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml CPU backend init failed");
        return fail();
    }

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 8;
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    j.w_ctx       = ggml_init(ip);
    if (j.w_ctx == nullptr) {
        return fail();
    }

    j.g_enc_w  = ggml_new_tensor_2d(j.w_ctx, GGML_TYPE_F32, j.d_enc, joint_h);
    j.g_enc_b  = ggml_new_tensor_1d(j.w_ctx, GGML_TYPE_F32, joint_h);
    j.g_pred_w = ggml_new_tensor_2d(j.w_ctx, step_weight_type(src_pred_w, keep_quantized), j.pred_hidden, joint_h);
    j.g_pred_b = ggml_new_tensor_1d(j.w_ctx, GGML_TYPE_F32, joint_h);
    j.gw_w     = ggml_new_tensor_2d(j.w_ctx, step_weight_type(src_out_w, keep_quantized), joint_h, joint_n);
    j.gw_b     = ggml_new_tensor_1d(j.w_ctx, GGML_TYPE_F32, joint_n);

    j.w_buf = ggml_backend_alloc_ctx_tensors(j.w_ctx, j.w_backend);
    if (j.w_buf == nullptr) {
        return fail();
    }

    // out_w has no host mirror; dequantize it only for an fp32 copy.
    {
        std::vector<float> tmp;
        if (j.gw_w->type != src_out_w->type && !read_tensor_to_f32(src_out_w, tmp)) {
            return fail();
        }
        set_step_weight(j.gw_w, src_out_w, tmp);
    }
    // The rest come from the host mirrors.
    ggml_backend_tensor_set(j.g_enc_w, j.enc_w.data(), 0, j.enc_w.size() * sizeof(float));
    ggml_backend_tensor_set(j.g_enc_b, j.enc_b.data(), 0, j.enc_b.size() * sizeof(float));
    set_step_weight(j.g_pred_w, src_pred_w, j.pred_w);
    ggml_backend_tensor_set(j.g_pred_b, j.pred_b.data(), 0, j.pred_b.size() * sizeof(float));
    ggml_backend_tensor_set(j.gw_b, j.out_b.data(), 0, j.out_b.size() * sizeof(float));

    // Host mirrors are now resident in ggml — release them (load-time scratch).
    std::vector<float>().swap(j.enc_w);
    std::vector<float>().swap(j.enc_b);
    std::vector<float>().swap(j.pred_w);
    std::vector<float>().swap(j.pred_b);
    std::vector<float>().swap(j.out_b);

    j.w_ready = true;
    return true;
}

// Make the predictor LSTM weights resident as ggml tensors for the per-call
// PredGraph. Built once at load. ne is [pred_hidden, 4*pred_hidden] for Wx/Wh
// (row-major [4*H, H] host bytes as a mul_mat operand; a quantized source
// type is kept, see step_weight_type) and [4*pred_hidden] for the fp32 bias.
// On failure frees partial state and returns false (lstm_ready stays false →
// hard decode error).
bool build_pred_weights(HostPredictor & p, const std::vector<ParakeetPredictor::LstmLayer> & src, bool keep_quantized) {
    const int H      = p.pred_hidden;
    const int four_H = 4 * H;
    const int L      = static_cast<int>(p.lstm.size());
    if (L == 0 || H == 0) {
        return false;
    }

    auto fail = [&]() -> bool {
        if (p.lstm_w_buf != nullptr) {
            safe_buffer_free(p.lstm_w_buf);
            p.lstm_w_buf = nullptr;
        }
        if (p.lstm_w_ctx != nullptr) {
            ggml_free(p.lstm_w_ctx);
            p.lstm_w_ctx = nullptr;
        }
        if (p.lstm_w_backend != nullptr) {
            safe_backend_free(p.lstm_w_backend);
            p.lstm_w_backend = nullptr;
        }
        for (auto & lh : p.lstm) {
            lh.g_Wx = nullptr;
            lh.g_Wh = nullptr;
            lh.g_b  = nullptr;
        }
        p.lstm_ready = false;
        return false;
    };

    // Alloc-only backend (never graph_compute'd), mirroring HostJoint::w_backend.
    p.lstm_w_backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (p.lstm_w_backend == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml CPU backend init failed (pred)");
        return fail();
    }

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * static_cast<size_t>(L * 3 + 1);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    p.lstm_w_ctx  = ggml_init(ip);
    if (p.lstm_w_ctx == nullptr) {
        return fail();
    }

    for (int l = 0; l < L; ++l) {
        auto & lh = p.lstm[l];
        lh.g_Wx   = ggml_new_tensor_2d(p.lstm_w_ctx, step_weight_type(src[l].Wx, keep_quantized), H, four_H);
        lh.g_Wh   = ggml_new_tensor_2d(p.lstm_w_ctx, step_weight_type(src[l].Wh, keep_quantized), H, four_H);
        lh.g_b    = ggml_new_tensor_1d(p.lstm_w_ctx, GGML_TYPE_F32, four_H);
    }

    p.lstm_w_buf = ggml_backend_alloc_ctx_tensors(p.lstm_w_ctx, p.lstm_w_backend);
    if (p.lstm_w_buf == nullptr) {
        return fail();
    }

    for (int l = 0; l < L; ++l) {
        auto & lh = p.lstm[l];
        set_step_weight(lh.g_Wx, src[l].Wx, lh.Wx);
        set_step_weight(lh.g_Wh, src[l].Wh, lh.Wh);
        ggml_backend_tensor_set(lh.g_b, lh.b.data(), 0, lh.b.size() * sizeof(float));
        // Host mirrors now resident in ggml — release them.
        std::vector<float>().swap(lh.Wx);
        std::vector<float>().swap(lh.Wh);
        std::vector<float>().swap(lh.b);
    }

    p.lstm_ready = true;
    return true;
}

}  // namespace

// Per-call joint compute graph (the mutable half of the joint network):
//   pred_proj = pred_w @ pred_in + pred_b      [joint_h]
//   summed    = enc_proj + pred_proj           [joint_h]
//   activated = activation(summed)             [joint_h]
//   logits    = out_w @ activated + out_b      [joint_n]
// Built on a BORROWED backend (PredGraph's shared pool) so the joint runs
// on the same threadpool as the pred LSTM and enc_proj. Inputs are the
// predictor output and this frame's precomputed enc projection; output is
// the logits. Built fresh per decode call (reentrant); weights are the
// shared resident HostJoint tensors.
namespace {

struct JointGraph {
    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;  // BORROWED (PredGraph's); NOT freed here
    ggml_backend_buffer_t buf     = nullptr;
    ggml_cgraph *         graph   = nullptr;
    ggml_tensor *         pred_in = nullptr;  // [pred_hidden] fp32 input (decoder out)
    ggml_tensor *         enc_in  = nullptr;  // [joint_h] fp32 input (enc_proj frame)
    ggml_tensor *         logits  = nullptr;  // [joint_n] fp32 output
    ggml_tensor *         probs   = nullptr;  // [n_probs] softmax of the token logits (n_probs > 0)
    ggml_tensor *         mean    = nullptr;  // [1] token logits averaged under probs (n_probs > 0)
    bool                  ready   = false;

    JointGraph() = default;

    ~JointGraph() {
        if (buf != nullptr) {
            safe_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        // backend is borrowed from PredGraph — do NOT free here.
    }

    JointGraph(const JointGraph &)             = delete;
    JointGraph & operator=(const JointGraph &) = delete;
};

// Build the full joint graph on the shared `backend` (PredGraph's pool).
// Returns false if the resident weights are absent or any ggml step
// fails (→ hard decode error).
bool build_joint_graph(JointGraph & g, const HostJoint & j, ggml_backend_t backend, int n_probs = 0) {
    if (backend == nullptr) {
        return false;
    }
    if (!j.w_ready || j.gw_w == nullptr || j.gw_b == nullptr || j.g_pred_w == nullptr || j.g_pred_b == nullptr) {
        return false;
    }

    g.backend = backend;  // borrowed

    auto fail = [&]() -> bool {
        if (g.buf != nullptr) {
            safe_buffer_free(g.buf);
            g.buf = nullptr;
        }
        if (g.ctx != nullptr) {
            ggml_free(g.ctx);
            g.ctx = nullptr;
        }
        g.pred_in = nullptr;
        g.enc_in  = nullptr;
        g.logits  = nullptr;
        g.graph   = nullptr;
        g.ready   = false;
        return false;
    };

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * 24 + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx         = ggml_init(ip);
    if (g.ctx == nullptr) {
        return fail();
    }

    g.pred_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, j.pred_hidden);
    ggml_set_input(g.pred_in);
    g.enc_in = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, j.joint_h);
    ggml_set_input(g.enc_in);

    // pred_proj = pred_w @ pred_in + pred_b   [joint_h]
    ggml_tensor * pred_proj = ggml_add(g.ctx, ggml_mul_mat(g.ctx, j.g_pred_w, g.pred_in), j.g_pred_b);
    // summed = enc_proj + pred_proj           [joint_h]
    ggml_tensor * summed    = ggml_add(g.ctx, g.enc_in, pred_proj);
    // activation — loader allow-list guarantees exactly one of relu/sigmoid/tanh.
    ggml_tensor * activated;
    if (j.activation == "relu") {
        activated = ggml_relu(g.ctx, summed);
    } else if (j.activation == "sigmoid") {
        activated = ggml_sigmoid(g.ctx, summed);
    } else {  // "tanh"
        activated = ggml_tanh(g.ctx, summed);
    }
    // logits = out_w @ activated + out_b      [joint_n]
    ggml_tensor * mm = ggml_mul_mat(g.ctx, j.gw_w, activated);
    g.logits         = ggml_add(g.ctx, mm, j.gw_b);
    ggml_set_output(g.logits);
    if (n_probs > 0) {
        ggml_tensor * tok = ggml_view_1d(g.ctx, g.logits, n_probs, 0);
        g.probs           = ggml_soft_max(g.ctx, tok);
        g.mean            = ggml_sum_rows(g.ctx, ggml_mul(g.ctx, g.probs, tok));
        ggml_set_output(g.probs);
        ggml_set_output(g.mean);
    }

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, g.backend);
    if (g.buf == nullptr) {
        return fail();
    }

    g.graph = ggml_new_graph(g.ctx);
    ggml_build_forward_expand(g.graph, g.logits);
    if (n_probs > 0) {
        ggml_build_forward_expand(g.graph, g.mean);
    }
    g.ready = true;
    return true;
}

// CPU-backend threadpool entry points, reached through the registry so
// the library stays DL-safe (under GGML_BACKEND_DL these symbols are not
// directly linkable). Resolved via ggml_backend_reg_get_proc_address.
typedef ggml_threadpool_t (*pfn_threadpool_new)(ggml_threadpool_params *);
typedef void (*pfn_threadpool_free)(ggml_threadpool_t);
typedef void (*pfn_set_threadpool)(ggml_backend_t, ggml_threadpool_t);

static void * cpu_backend_proc(ggml_backend_t backend, const char * name) {
    ggml_backend_dev_t dev = backend != nullptr ? ggml_backend_get_device(backend) : nullptr;
    ggml_backend_reg_t reg = dev != nullptr ? ggml_backend_dev_backend_reg(dev) : nullptr;
    return reg != nullptr ? ggml_backend_reg_get_proc_address(reg, name) : nullptr;
}

// Free a CPU-backend threadpool via the registry. Resolving the free fn needs
// the backend alive, so callers must invoke this BEFORE freeing the backend.
static void free_cpu_threadpool(ggml_backend_t backend, ggml_threadpool_t tp) {
    if (tp == nullptr || backend == nullptr) {
        return;
    }
    if (auto fn = (pfn_threadpool_free) cpu_backend_proc(backend, "ggml_threadpool_free")) {
        fn(tp);
    }
}

// One LSTM cell as a single graph node: dst = [h'; c'] from the gate
// matmuls src[0] = Wx@x and src[1] = Wh@h_prev, bias src[2] and c_prev
// src[3], gates in PyTorch [i, f, g, o] order. As separate ggml ops the
// pointwise math took ~15 nodes per layer, each a thread barrier.
void lstm_cell(ggml_tensor * dst, int ith, int nth, void * userdata) {
    (void) nth;
    (void) userdata;
    if (ith != 0) {
        return;
    }
    const auto * gx    = static_cast<const float *>(dst->src[0]->data);
    const auto * gh    = static_cast<const float *>(dst->src[1]->data);
    const auto * b     = static_cast<const float *>(dst->src[2]->data);
    const auto * c     = static_cast<const float *>(dst->src[3]->data);
    const int    H     = static_cast<int>(dst->src[3]->ne[0]);
    auto *       h_out = static_cast<float *>(dst->data);
    float *      c_out = h_out + H;
    auto         gate  = [&](int k, int j) {
        const int r = k * H + j;
        return (gx[r] + gh[r]) + b[r];
    };
    auto sigmoid = [](float v) {
        return 1.0f / (1.0f + std::exp(-v));
    };
    for (int j = 0; j < H; ++j) {
        const float i_ = sigmoid(gate(0, j));
        const float f_ = sigmoid(gate(1, j));
        const float g_ = std::tanh(gate(2, j));
        const float o_ = sigmoid(gate(3, j));
        c_out[j]       = f_ * c[j] + i_ * g_;
        h_out[j]       = o_ * std::tanh(c_out[j]);
    }
}

// Per-call predictor LSTM graph (the mutable half of the predictor):
// a single per-step graph built fresh per decode call around the
// model-resident HostPredictor::lstm weights, recomputed in place each
// step. Inputs are the embedding x and per-layer previous (h, c); outputs
// are the new (h, c) per layer. Reentrant: every concurrent decode owns
// its own PredGraph (backend + threadpool + I/O tensors).
struct PredGraph {
    ggml_context *             ctx     = nullptr;
    ggml_backend_t             backend = nullptr;
    ggml_backend_buffer_t      buf     = nullptr;
    ggml_cgraph *              graph   = nullptr;
    ggml_threadpool_t          tp      = nullptr;
    ggml_tensor *              x       = nullptr;  // [H] input embedding
    std::vector<ggml_tensor *> ph;                 // [H] prev hidden per layer (input)
    std::vector<ggml_tensor *> pc;                 // [H] prev cell   per layer (input)
    std::vector<ggml_tensor *> nh;                 // [H] new  hidden per layer (output)
    std::vector<ggml_tensor *> nc;                 // [H] new  cell   per layer (output)
    int                        H     = 0;
    int                        L     = 0;
    bool                       ready = false;

    PredGraph() = default;

    ~PredGraph() {
        if (buf != nullptr) {
            safe_buffer_free(buf);
        }
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
        free_cpu_threadpool(backend, tp);  // before safe_backend_free(backend)
        if (backend != nullptr) {
            safe_backend_free(backend);
        }
    }

    PredGraph(const PredGraph &)             = delete;
    PredGraph & operator=(const PredGraph &) = delete;
};

// Build the per-call LSTM graph around the resident p.lstm[*].g_Wx/g_Wh/g_b.
// Returns false if the resident weights are absent or any ggml step fails
// (→ hard decode error). n_threads is the resolved (>0) thread count.
bool build_pred_graph(PredGraph & g, const HostPredictor & p, int n_threads) {
    if (!p.lstm_ready) {
        return false;
    }
    const int H = p.pred_hidden;
    const int L = static_cast<int>(p.lstm.size());
    if (H == 0 || L == 0) {
        return false;
    }

    auto fail = [&]() -> bool {
        if (g.buf != nullptr) {
            safe_buffer_free(g.buf);
            g.buf = nullptr;
        }
        if (g.ctx != nullptr) {
            ggml_free(g.ctx);
            g.ctx = nullptr;
        }
        free_cpu_threadpool(g.backend, g.tp);
        g.tp = nullptr;  // before freeing backend
        if (g.backend != nullptr) {
            safe_backend_free(g.backend);
            g.backend = nullptr;
        }
        g.x     = nullptr;
        g.graph = nullptr;
        g.ready = false;
        g.ph.clear();
        g.pc.clear();
        g.nh.clear();
        g.nc.clear();
        return false;
    };

    g.H = H;
    g.L = L;

    g.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (g.backend == nullptr) {
        return fail();
    }
    if (ggml_backend_dev_t dev = ggml_backend_get_device(g.backend)) {
        auto set_n_threads = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(dev), "ggml_backend_set_n_threads");
        if (set_n_threads != nullptr) {
            set_n_threads(g.backend, std::max(1, n_threads));
        }
    }
    // Persistent threadpool so the per-step graph_compute reuses workers instead
    // of spawning a transient pool each call. ggml's default hybrid-polling
    // (poll=50) keeps the workers hot between the sub-millisecond per-step
    // dispatches; poll=0 (park-immediately) was measured to regress pred to
    // 70-110 ms because parked workers are slow to reschedule.
    {
        auto tp_new = (pfn_threadpool_new) cpu_backend_proc(g.backend, "ggml_threadpool_new");
        auto tp_set = (pfn_set_threadpool) cpu_backend_proc(g.backend, "ggml_backend_cpu_set_threadpool");
        if (tp_new != nullptr && tp_set != nullptr) {
            ggml_threadpool_params tpp = ggml_threadpool_params_default(std::max(1, n_threads));
            g.tp                       = tp_new(&tpp);
            if (g.tp != nullptr) {
                tp_set(g.backend, g.tp);
            }
        }
        // If unresolved (non-CPU exotic backend), g.tp stays null and each
        // graph_compute spawns a transient pool — correct, just less tuned.
    }

    ggml_init_params ip{};
    // 5 nodes/layer + (1 + 2L) input tensors; generous headroom.
    ip.mem_size   = ggml_tensor_overhead() * static_cast<size_t>(16 * L + 16) + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx         = ggml_init(ip);
    if (g.ctx == nullptr) {
        return fail();
    }

    g.x = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, H);
    ggml_set_input(g.x);
    g.ph.assign(static_cast<size_t>(L), nullptr);
    g.pc.assign(static_cast<size_t>(L), nullptr);
    g.nh.assign(static_cast<size_t>(L), nullptr);
    g.nc.assign(static_cast<size_t>(L), nullptr);
    for (int l = 0; l < L; ++l) {
        g.ph[l] = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, H);
        ggml_set_input(g.ph[l]);
        g.pc[l] = ggml_new_tensor_1d(g.ctx, GGML_TYPE_F32, H);
        ggml_set_input(g.pc[l]);
    }

    // gates = Wx@x + Wh@h_prev + b; split [i,f,g,o]; c' = f*c + i*g; h' = o*tanh(c').
    ggml_tensor * in = g.x;
    for (int l = 0; l < L; ++l) {
        const auto &  lh     = p.lstm[static_cast<size_t>(l)];
        ggml_tensor * args[] = { ggml_mul_mat(g.ctx, lh.g_Wx, in), ggml_mul_mat(g.ctx, lh.g_Wh, g.ph[l]), lh.g_b,
                                 g.pc[l] };
        ggml_tensor * cell   = ggml_custom_4d(g.ctx, GGML_TYPE_F32, 2 * H, 1, 1, 1, args, 4, lstm_cell, 1, nullptr);
        ggml_set_output(cell);
        g.nh[l] = ggml_view_1d(g.ctx, cell, H, 0);
        g.nc[l] = ggml_view_1d(g.ctx, cell, H, static_cast<size_t>(H) * sizeof(float));
        in      = g.nh[l];
    }

    g.buf = ggml_backend_alloc_ctx_tensors(g.ctx, g.backend);
    if (g.buf == nullptr) {
        return fail();
    }

    g.graph = ggml_new_graph(g.ctx);
    for (int l = 0; l < L; ++l) {
        ggml_build_forward_expand(g.graph, g.nh[l]);
        ggml_build_forward_expand(g.graph, g.nc[l]);
    }
    g.ready = true;
    return true;
}

}  // namespace

HostJoint::~HostJoint() {
    if (w_buf != nullptr) {
        safe_buffer_free(w_buf);
    }
    if (w_ctx != nullptr) {
        ggml_free(w_ctx);
    }
    if (w_backend != nullptr) {
        safe_backend_free(w_backend);
    }
}

HostPredictor::~HostPredictor() {
    if (lstm_w_buf != nullptr) {
        safe_buffer_free(lstm_w_buf);
    }
    if (lstm_w_ctx != nullptr) {
        ggml_free(lstm_w_ctx);
    }
    if (lstm_w_backend != nullptr) {
        safe_backend_free(lstm_w_backend);
    }
}

// Resolve a decode thread count: n_threads <= 0 means "auto" →
// default_n_threads() (min(8, usable cpus)), matching the encoder.
static int resolve_decode_threads(int n_threads) {
    return n_threads > 0 ? n_threads : transcribe::default_n_threads();
}

transcribe_status build_host_decoder_weights(const ParakeetModel & model, HostDecoderWeights & out) {
    const ParakeetHParams & hp = model.hparams;
    const ParakeetWeights & w  = model.weights;

    // Head-kind dispatch. CTC skips predictor/joint entirely; RNNT and
    // TDT share the predictor + joint mirror code below.
    switch (hp.head_kind) {
        case HeadKind::TDT:
            out.head_kind = HostHeadKind::TDT;
            break;
        case HeadKind::RNNT:
            out.head_kind = HostHeadKind::RNNT;
            break;
        case HeadKind::CTC:
            out.head_kind = HostHeadKind::CTC;
            break;
    }

    if (hp.head_kind == HeadKind::CTC) {
        // CTC head mirror. Source weight ne [1, d_model, n_classes]
        // (PyTorch [n_classes, d_model, 1]); bytes are already row-major
        // [n_classes, d_model] as the host decoder consumes them. Bias is
        // [n_classes].
        out.ctc_head.n_classes = hp.head_ctc_n_classes;
        out.ctc_head.blank_id  = hp.head_ctc_n_classes - 1;  // NeMo convention
        out.ctc_head.d_enc     = hp.enc_d_model;

        if (!read_tensor_to_f32(w.ctc_head.weight, out.ctc_head.weight)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (!read_tensor_to_f32(w.ctc_head.bias, out.ctc_head.bias)) {
            return TRANSCRIBE_ERR_GGUF;
        }

        const size_t expected_w = static_cast<size_t>(out.ctc_head.n_classes) * static_cast<size_t>(out.ctc_head.d_enc);
        if (out.ctc_head.weight.size() != expected_w) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "parakeet decoder: head.ctc.weight size %zu "
                    "!= n_classes*d_enc %zu",
                    out.ctc_head.weight.size(), expected_w);
            return TRANSCRIBE_ERR_GGUF;
        }
        if (static_cast<int>(out.ctc_head.bias.size()) != out.ctc_head.n_classes) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "parakeet decoder: head.ctc.bias size %zu "
                    "!= n_classes %d",
                    out.ctc_head.bias.size(), out.ctc_head.n_classes);
            return TRANSCRIBE_ERR_GGUF;
        }

        out.tdt_durations.clear();
        out.tdt_max_symbols = 0;
        out.blank_id        = out.ctc_head.blank_id;
        out.n_vocab         = out.ctc_head.n_classes - 1;
        return TRANSCRIBE_OK;
    }

    // ----- Predictor mirror (TDT, RNNT) -----
    const bool keep_quantized = hp.enc_att_context_style != ParakeetHParams::AttContextStyle::ChunkedLimited;
    out.predictor.pred_hidden = hp.pred_hidden;
    out.predictor.pred_vocab  = hp.pred_vocab;

    if (!read_tensor_to_f32(w.predictor.embed_w, out.predictor.embed_w)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    // Sanity-check the flattened size against the catalog shape.
    {
        const size_t expected = static_cast<size_t>(hp.pred_vocab) * static_cast<size_t>(hp.pred_hidden);
        if (out.predictor.embed_w.size() != expected) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                    "parakeet decoder: pred.embed.weight size %zu "
                    "!= pred_vocab*pred_hidden %zu",
                    out.predictor.embed_w.size(), expected);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    out.predictor.lstm.assign(hp.pred_n_layers, HostLstmLayer{});
    for (int i = 0; i < hp.pred_n_layers; ++i) {
        const auto & lw = w.predictor.lstm[i];
        auto &       lh = out.predictor.lstm[i];
        if (!read_tensor_to_f32(lw.Wx, lh.Wx)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (!read_tensor_to_f32(lw.Wh, lh.Wh)) {
            return TRANSCRIBE_ERR_GGUF;
        }
        if (!read_tensor_to_f32(lw.b, lh.b)) {
            return TRANSCRIBE_ERR_GGUF;
        }

        const size_t gates    = static_cast<size_t>(4 * hp.pred_hidden);
        const size_t mat_size = gates * static_cast<size_t>(hp.pred_hidden);
        if (lh.Wx.size() != mat_size || lh.Wh.size() != mat_size) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: pred.lstm.%d Wx/Wh wrong size", i);
            return TRANSCRIBE_ERR_GGUF;
        }
        if (lh.b.size() != gates) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: pred.lstm.%d bias wrong size", i);
            return TRANSCRIBE_ERR_GGUF;
        }
    }

    // Make the predictor LSTM weights resident. Fatal: a failure means
    // the model cannot decode — fail fast at load.
    if (!build_pred_weights(out.predictor, w.predictor.lstm, keep_quantized)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: predictor ggml weight build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // ----- Joint mirror -----
    out.joint.d_enc       = hp.enc_d_model;
    out.joint.pred_hidden = hp.pred_hidden;
    out.joint.joint_h     = hp.joint_hidden;
    out.joint.joint_n     = hp.joint_n_classes();
    out.joint.activation  = hp.joint_activation;

    // enc/pred/out_b are mirrored to host fp32; out_w is NOT mirrored
    // here (build_joint_weight reads it from the model tensor).
    if (!read_tensor_to_f32(w.joint.enc_w, out.joint.enc_w)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!read_tensor_to_f32(w.joint.enc_b, out.joint.enc_b)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!read_tensor_to_f32(w.joint.pred_w, out.joint.pred_w)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!read_tensor_to_f32(w.joint.pred_b, out.joint.pred_b)) {
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!read_tensor_to_f32(w.joint.out_b, out.joint.out_b)) {
        return TRANSCRIBE_ERR_GGUF;
    }

    // Make the joint weights resident as fp32 ggml tensors. Fatal: a
    // failure means the model cannot decode — fail fast at load.
    if (w.joint.out_w == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: joint out_w missing");
        return TRANSCRIBE_ERR_GGUF;
    }
    const int ne0 = static_cast<int>(w.joint.out_w->ne[0]);
    const int ne1 = static_cast<int>(w.joint.out_w->ne[1]);
    if (ne0 != out.joint.joint_h || ne1 != out.joint.joint_n) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: out_w ne [%d,%d] != [joint_h=%d, joint_n=%d]", ne0, ne1,
                out.joint.joint_h, out.joint.joint_n);
        return TRANSCRIBE_ERR_GGUF;
    }
    if (!build_joint_weight(out.joint, w.joint.pred_w, w.joint.out_w, keep_quantized)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: joint ggml weight build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // ----- TDT params -----
    out.tdt_durations   = hp.tdt_durations;
    out.tdt_max_symbols = hp.tdt_max_symbols;
    out.n_vocab         = hp.pred_vocab - 1;  // raw SP vocab size
    out.blank_id        = hp.pred_vocab - 1;  // blank lives at vocab_size

    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// LSTM state
// ---------------------------------------------------------------------------

void LstmState::reset(int n_layers, int pred_hidden) {
    h.resize(static_cast<size_t>(n_layers));
    c.resize(static_cast<size_t>(n_layers));
    for (int layer = 0; layer < n_layers; ++layer) {
        h[static_cast<size_t>(layer)].assign(static_cast<size_t>(pred_hidden), 0.0f);
        c[static_cast<size_t>(layer)].assign(static_cast<size_t>(pred_hidden), 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

namespace {

// One predictor step on the ggml graph: reads prev_state, writes the new
// step's (h, c) into new_state, returns a borrowed pointer into
// new_state.h.back(). Feeds prev state + embedding into the resident
// per-call graph, computes, reads new state back into the host LstmState.
// Caller guarantees g.ready and that new_state is sized to (L, H). Returns
// nullptr if the graph compute fails.
const float * predictor_step_ggml(const HostPredictor & predictor,
                                  PredGraph &           g,
                                  int                   last_token,
                                  const LstmState &     prev_state,
                                  LstmState &           new_state,
                                  std::vector<float> &  scratch_x) {
    const int H = predictor.pred_hidden;
    if (static_cast<int>(scratch_x.size()) < H) {
        scratch_x.resize(static_cast<size_t>(H));
    }

    // Embed lookup (or start-state zeros).
    if (last_token < 0) {
        std::fill_n(scratch_x.data(), H, 0.0f);
    } else {
        const size_t row_off = static_cast<size_t>(last_token) * static_cast<size_t>(H);
        std::memcpy(scratch_x.data(), predictor.embed_w.data() + row_off, static_cast<size_t>(H) * sizeof(float));
    }

    const size_t hb = static_cast<size_t>(H) * sizeof(float);
    ggml_backend_tensor_set(g.x, scratch_x.data(), 0, hb);
    for (int l = 0; l < g.L; ++l) {
        ggml_backend_tensor_set(g.ph[l], prev_state.h[static_cast<size_t>(l)].data(), 0, hb);
        ggml_backend_tensor_set(g.pc[l], prev_state.c[static_cast<size_t>(l)].data(), 0, hb);
    }

    if (const ggml_status gs = ggml_backend_graph_compute(g.backend, g.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: predictor compute failed (%d)", static_cast<int>(gs));
        return nullptr;
    }

    for (int l = 0; l < g.L; ++l) {
        ggml_backend_tensor_get(g.nh[l], new_state.h[static_cast<size_t>(l)].data(), 0, hb);
        ggml_backend_tensor_get(g.nc[l], new_state.c[static_cast<size_t>(l)].data(), 0, hb);
    }
    return new_state.h.back().data();
}

// Run the joint network for one decode step.
//
// `enc_proj` is the precomputed encoder projection for this frame
// (enc_w @ enc_frame + enc_b, joint_h floats); the caller batches all
// T_enc projections via one GEMM (precompute_enc_proj_ggml) before the
// loop. The graph computes:
//   pred_proj = pred_w @ pred_state + pred_b     [joint_h]
//   summed    = enc_proj + pred_proj             [joint_h]
//   activated = activation(summed)               [joint_h]
//   logits    = out_w @ activated   + out_b      [joint_n]
// Activation is one of {relu, sigmoid, tanh} (loader allow-list). Returns
// false if the graph compute fails.
bool joint_step(const HostJoint &    j,
                const JointGraph &   g,
                const float *        enc_proj,
                const float *        pred_state,
                std::vector<float> & out_logits) {
    if (static_cast<int>(out_logits.size()) < j.joint_n) {
        out_logits.resize(static_cast<size_t>(j.joint_n));
    }

    // Full joint on the shared decoder pool, one graph, one dispatch.
    ggml_backend_tensor_set(g.pred_in, pred_state, 0, static_cast<size_t>(j.pred_hidden) * sizeof(float));
    ggml_backend_tensor_set(g.enc_in, enc_proj, 0, static_cast<size_t>(j.joint_h) * sizeof(float));
    if (const ggml_status gs = ggml_backend_graph_compute(g.backend, g.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: joint compute failed (%d)", static_cast<int>(gs));
        return false;
    }
    ggml_backend_tensor_get(g.logits, out_logits.data(), 0, static_cast<size_t>(j.joint_n) * sizeof(float));

    // log_softmax over the full joint output matches NeMo's CPU-inference
    // representation of joint_after_projection. It is a uniform per-row
    // shift, so it leaves token/duration argmax AND token_confidence
    // (which re-softmaxes the token sub-range) invariant — the decode
    // output is bit-identical with or without it. It is only needed to
    // make the dec.joint.0 dump element-wise comparable to NeMo's
    // normalized reference, so skip it unless dumping.
    if (transcribe::debug::enabled()) {
        float max_v = out_logits[0];
        for (int i = 1; i < j.joint_n; ++i) {
            if (out_logits[i] > max_v) {
                max_v = out_logits[i];
            }
        }
        double sum = 0.0;
        for (int i = 0; i < j.joint_n; ++i) {
            sum += std::exp(static_cast<double>(out_logits[i] - max_v));
        }
        const float log_sum = static_cast<float>(std::log(sum)) + max_v;
        for (int i = 0; i < j.joint_n; ++i) {
            out_logits[i] -= log_sum;
        }
    }
    return true;
}

// Compute the per-utterance encoder projection out[T, joint_h] =
// enc_out[T, d_enc] @ enc_w^T + enc_b as a single GEMM on `backend`
// (the shared decoder pool). Weight + bias are the model-resident
// j.g_enc_w / j.g_enc_b; only the input and result are allocated here.
// At n = T the matmul is a real GEMM (tinyBLAS engages with
// GGML_LLAMAFILE=ON). Returns false on ggml failure (→ hard decode error).
bool precompute_enc_proj_ggml(const HostJoint &    j,
                              ggml_backend_t       backend,
                              const float *        enc_out,
                              int                  T,
                              int                  d_enc,
                              std::vector<float> & out) {
    if (backend == nullptr || j.g_enc_w == nullptr || j.g_enc_b == nullptr) {
        return false;
    }
    const int joint_h = j.joint_h;
    out.resize(static_cast<size_t>(T) * static_cast<size_t>(joint_h));

    ggml_init_params ip{};
    ip.mem_size        = ggml_tensor_overhead() * 6 + ggml_graph_overhead();
    ip.mem_buffer      = nullptr;
    ip.no_alloc        = true;
    ggml_context * ctx = ggml_init(ip);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_enc, T);
    ggml_set_input(in);
    ggml_tensor * mm  = ggml_mul_mat(ctx, j.g_enc_w, in);  // [joint_h, T]
    ggml_tensor * res = ggml_add(ctx, mm, j.g_enc_b);      // + [joint_h] (broadcast over T)
    ggml_set_output(res);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (buf == nullptr) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_set(in, enc_out, 0, static_cast<size_t>(T) * static_cast<size_t>(d_enc) * sizeof(float));

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, res);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        safe_buffer_free(buf);
        ggml_free(ctx);
        return false;
    }

    ggml_backend_tensor_get(res, out.data(), 0, static_cast<size_t>(T) * static_cast<size_t>(joint_h) * sizeof(float));

    safe_buffer_free(buf);
    ggml_free(ctx);
    return true;
}

// Argmax over a contiguous fp32 range. Returns the index of the
// largest value; ties go to the first occurrence (matches the numpy
// argmax convention). The decoder uses this for both the token and
// duration argmaxes.
int argmax_range(const float * data, int n) {
    int   best_i = 0;
    float best_v = data[0];
    for (int i = 1; i < n; ++i) {
        if (data[i] > best_v) {
            best_v = data[i];
            best_i = i;
        }
    }
    return best_i;
}

// Entropy-based confidence over a token-logit slice. Mirrors the
// reference ParakeetTDT.decode_greedy path:
//
//     token_probs = softmax(token_logits)
//     entropy     = -sum(p * log(p + 1e-10))
//     max_entropy = log(vocab_size + 1)
//     confidence  = 1 - entropy / max_entropy
//
// In our terms `vocab_size + 1 == pred_vocab == n_token_classes`.
// Computed in closed form without the +1e-10 epsilon, which moves the
// result by well under 1e-6.
float token_confidence(const float * token_logits, int n_token_classes) {
    const double max_entropy = std::log(static_cast<double>(n_token_classes));
    if (max_entropy <= 0.0) {
        return 1.0f;
    }
    // With e_i = exp(x_i - max) and S = sum e_i, log p_i = (x_i - max) - log S,
    // so entropy = log S - sum e_i (x_i - max) / S: one exp per class, no log.
#if TRANSCRIBE_HAS_BLAS && defined(__APPLE__)
    // Vectorized (max, shift, exp, sum, dot): the scalar version was ~21 us
    // per emitted token over 8193 classes, a seventh of the decoder.
    thread_local std::vector<float> xs, es;
    xs.resize(static_cast<size_t>(n_token_classes));
    es.resize(static_cast<size_t>(n_token_classes));
    const vDSP_Length n         = static_cast<vDSP_Length>(n_token_classes);
    float             max_logit = 0.0f;
    vDSP_maxv(token_logits, 1, &max_logit, n);
    const float neg_max = -max_logit;
    vDSP_vsadd(token_logits, 1, &neg_max, xs.data(), 1, n);
    vvexpf(es.data(), xs.data(), &n_token_classes);
    float sum_exp = 0.0f;
    float sum_ex  = 0.0f;
    vDSP_sve(es.data(), 1, &sum_exp, n);
    vDSP_dotpr(es.data(), 1, xs.data(), 1, &sum_ex, n);
    const double entropy = std::log(static_cast<double>(sum_exp)) - static_cast<double>(sum_ex) / sum_exp;
#else
    float max_logit = token_logits[0];
    for (int i = 1; i < n_token_classes; ++i) {
        if (token_logits[i] > max_logit) {
            max_logit = token_logits[i];
        }
    }
    double sum_exp = 0.0;
    double sum_ex  = 0.0;
    for (int i = 0; i < n_token_classes; ++i) {
        const float x = token_logits[i] - max_logit;
        const float e = std::exp(x);
        sum_exp += static_cast<double>(e);
        sum_ex += static_cast<double>(e * x);
    }
    const double entropy = std::log(sum_exp) - sum_ex / sum_exp;
#endif
    return static_cast<float>(1.0 - entropy / max_entropy);
}

// A TDT frame that emits max_symbols tokens without advancing is stuck. When
// its tokens end in a repeating block the model is looping: drop the loop
// from its first token (transcribe-repetition-guard.h). Returns true if
// tokens were dropped.
bool trim_stuck_frame(std::vector<TdtToken> & toks, int step, int max_symbols) {
    if (!repetition_guard_enabled()) {
        return false;
    }
    size_t start = toks.size();
    while (start > 0 && toks[start - 1].step_at_emit == step) {
        --start;
    }
    std::vector<int32_t> ids;
    for (size_t i = start; i < toks.size(); ++i) {
        ids.push_back(toks[i].id);
    }
    const int       n     = static_cast<int>(ids.size());
    const RepeatBar bar   = { max_symbols / 2, 2, 0, max_symbols, 2 };
    const int       block = repeating_tail_block(ids.data(), n, bar);
    if (block == 0) {
        return false;
    }
    int loop = n - 2 * block;
    while (loop > 0 && ids[static_cast<size_t>(loop - 1)] == ids[static_cast<size_t>(loop - 1 + block)]) {
        --loop;
    }
    toks.resize(start + static_cast<size_t>(loop));
    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "parakeet decoder: dropped a %d-token loop (%d tokens) at frame %d", block,
            n - loop, step);
    return true;
}

// Rewinds the predictor after trim_stuck_frame: replays the frame's kept
// tokens from where the frame started, so the next frame decodes as if the
// loop never happened. Returns false if a predictor step fails.
bool rewind_stuck_frame(const HostPredictor &         predictor,
                        PredGraph &                   pg,
                        const std::vector<TdtToken> & toks,
                        int                           step,
                        const FrameStart &            frame,
                        LstmState &                   state,
                        int &                         last_token,
                        LstmState &                   scratch_state,
                        std::vector<float> &          scratch_x) {
    size_t start = toks.size();
    while (start > 0 && toks[start - 1].step_at_emit == step) {
        --start;
    }
    state      = frame.state;
    last_token = frame.last_token;
    for (size_t i = start; i < toks.size(); ++i) {
        if (predictor_step_ggml(predictor, pg, last_token, state, scratch_state, scratch_x) == nullptr) {
            return false;
        }
        std::swap(state, scratch_state);
        last_token = toks[i].id;
    }
    return true;
}

// Greedy TDT / RNN-T decode with phrase boosting (boost.h). The step rules
// are the unboosted loops' own. `main` is the committed path; its tokens go
// straight to `out`. When the boosted pick differs from the model's token,
// the decode forks into bs.plain (model token) and bs.boosted (swapped
// token, then continuation-only boosting, no nested forks). Whichever
// branch is behind in frames steps next; the fork resolves at the boosted
// branch's next trie transition (see BoostTrie::fork_advance), after
// k_boost_fork_tokens tokens (reject), or at the end of the frames when
// `end_of_audio` (reject unless a phrase completed). A completed phrase is
// kept only if it ends at a word edge and the deletion check below passes.
// An open fork otherwise stays in `bs` for the next call, frames rebased
// to 0.
transcribe_status decode_boosted(const HostDecoderWeights & w,
                                 const float *              enc_out,
                                 int                        T_enc,
                                 int                        d_enc,
                                 int                        n_threads,
                                 const BoostTrie &          trie,
                                 GreedyCursor &             main,
                                 BoostDecodeState &         bs,
                                 int                        frame_offset,
                                 bool                       end_of_audio,
                                 std::vector<TdtToken> &    out) {
    const bool tdt         = w.head_kind == HostHeadKind::TDT;
    const int  n_token_cls = w.predictor.pred_vocab;
    const int  n_dur       = static_cast<int>(w.tdt_durations.size());
    const int  blank_id    = w.blank_id;
    const int  joint_h     = w.joint.joint_h;

    PredGraph  pg;
    JointGraph jg;
    build_pred_graph(pg, w.predictor, resolve_decode_threads(n_threads));
    if (pg.ready) {
        build_joint_graph(jg, w.joint, pg.backend);
    }
    if (!pg.ready || !jg.ready) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml decode graph build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    std::vector<float> enc_proj_all;
    if (!precompute_enc_proj_ggml(w.joint, pg.backend, enc_out, T_enc, d_enc, enc_proj_all)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: enc_proj graph failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    std::vector<float> scratch_x;
    std::vector<float> logits;

    // Two cursors may run over the same frames while a fork is open.
    const int max_iters = 32 * T_enc + 2048;
    int       iter      = 0;
    int       n_forks   = 0;
    int       n_accept  = 0;

    GreedyCursor * cursors[3] = { &main, &bs.plain, &bs.boosted };
    for (GreedyCursor * c : cursors) {
        c->new_symbols = 0;
        c->dirty       = true;
    }

    auto eval = [&](GreedyCursor & c) -> bool {
        ++iter;
        const float * decoder_out;
        if (c.dirty) {
            decoder_out = predictor_step_ggml(w.predictor, pg, c.last_token, c.state, c.next_state, scratch_x);
            if (decoder_out == nullptr) {
                return false;
            }
            c.dirty = false;
        } else {
            decoder_out = c.next_state.h.back().data();
        }
        const float * enc_proj = enc_proj_all.data() + static_cast<size_t>(c.step) * static_cast<size_t>(joint_h);
        return joint_step(w.joint, jg, enc_proj, decoder_out, logits);
    };

    // Emit `tok` (blank = none) for `c` from the last evaluated joint output
    // and advance its frame cursor with the unboosted loops' rules. Returns
    // false if a predictor step fails.
    auto commit = [&](GreedyCursor & c, int tok, std::vector<TdtToken> & dst) -> bool {
        const bool is_blank = tok == blank_id;
        const int  duration =
            tdt ? w.tdt_durations[static_cast<size_t>(argmax_range(logits.data() + n_token_cls, n_dur))] : 1;
        if (!is_blank) {
            TdtToken t;
            t.id              = tok;
            t.p               = token_confidence(logits.data(), n_token_cls);
            t.step_at_emit    = frame_offset + c.step;
            t.duration_frames = duration;
            if (tdt && duration == 0 && (dst.empty() || dst.back().step_at_emit != t.step_at_emit)) {
                c.frame.state      = c.state;
                c.frame.last_token = c.last_token;
            }
            dst.push_back(t);
            c.last_token = tok;
            std::swap(c.state, c.next_state);
            c.dirty = true;
        }
        if (!tdt) {
            if (is_blank) {
                c.step += 1;
                c.new_symbols = 0;
            } else if (w.tdt_max_symbols > 0 && ++c.new_symbols >= w.tdt_max_symbols) {
                c.step += 1;
                c.new_symbols = 0;
            }
            return true;
        }
        c.step += duration;
        c.new_symbols += 1;
        if (duration != 0) {
            c.new_symbols = 0;
        } else if (w.tdt_max_symbols > 0 && c.new_symbols >= w.tdt_max_symbols) {
            if (trim_stuck_frame(dst, frame_offset + c.step, w.tdt_max_symbols)) {
                if (!rewind_stuck_frame(w.predictor, pg, dst, frame_offset + c.step, c.frame, c.state, c.last_token,
                                        c.next_state, scratch_x)) {
                    return false;
                }
                c.dirty = true;
            }
            c.step += 1;
            c.new_symbols = 0;
        } else if (is_blank && w.tdt_max_symbols > 0) {
            const int skip = w.tdt_max_symbols - c.new_symbols;
            if (skip > 0 && iter + skip < max_iters) {
                iter += skip;
                c.step += 1;
                c.new_symbols = 0;
            }
        }
        return true;
    };

    // A completed phrase is checked for derailment first: both branches run
    // k_boost_guard_frames more frames (the plain one from the same frame)
    // and the boosted branch is rejected if it falls silent there.
    auto start_guard = [&]() {
        bs.guard_from    = bs.boosted.step;
        bs.guard_until   = bs.boosted.step + k_boost_guard_frames;
        bs.guard_plain   = 0;
        bs.guard_boosted = 0;
        bs.guard_next    = false;
    };

    auto resolve = [&](bool accept) {
        GreedyCursor & win = accept ? bs.boosted : bs.plain;
        out.insert(out.end(), win.held.begin(), win.held.end());
        win.held.clear();
        std::swap(main, win);
        bs.node        = 0;
        bs.fork_open   = false;
        bs.guard_until = -1;
        n_accept += accept ? 1 : 0;
    };

    while (iter < max_iters) {
        if (!bs.fork_open) {
            if (main.step >= T_enc) {
                break;
            }
            if (!eval(main)) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            const int u = argmax_range(logits.data(), n_token_cls);
            const int b = u == blank_id ? u : trie.pick(bs.node, logits.data(), n_token_cls, u, false);
            if (b == u) {
                if (!commit(main, u, out)) {
                    return TRANSCRIBE_ERR_BACKEND;
                }
                if (u != blank_id) {
                    bs.node = trie.next(bs.node, u);
                }
                continue;
            }
            ++n_forks;
            bs.plain   = main;
            bs.boosted = main;
            if (!commit(bs.plain, u, bs.plain.held)) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            if (!commit(bs.boosted, b, bs.boosted.held)) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            bs.fork_open   = true;
            bs.fork_node   = bs.node;
            bs.fork_tokens = 1;
            if (trie.fork_begin(bs.fork_node, bs.fork_completed, b) == BoostVerdict::Accept) {
                start_guard();
            }
            continue;
        }

        GreedyCursor & a = bs.plain;
        GreedyCursor & b = bs.boosted;
        if (bs.guard_until >= 0) {
            const int  end    = std::min(bs.guard_until, T_enc);
            const bool a_live = a.step < end;
            const bool b_live = b.step < end;
            if (!a_live && !b_live) {
                if (bs.guard_until > T_enc && !end_of_audio) {
                    break;  // the window continues in the next call
                }
                resolve(boost_guard_passed(bs.guard_plain, bs.guard_boosted));
                continue;
            }
            GreedyCursor & c = a_live && (!b_live || a.step < b.step) ? a : b;
            if (!eval(c)) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            const int u    = argmax_range(logits.data(), n_token_cls);
            const int from = c.step;
            if (!commit(c, u, c.held)) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            if (u != blank_id && from >= bs.guard_from &&
                (trie.token_flags[static_cast<size_t>(u)] & k_boost_token_word) != 0) {
                ++(&c == &a ? bs.guard_plain : bs.guard_boosted);
            }
            // A phrase glued onto the next word piece is not the phrase.
            if (&c == &b && u != blank_id && !bs.guard_next) {
                bs.guard_next = true;
                if ((trie.token_flags[static_cast<size_t>(u)] & k_boost_token_glue) != 0) {
                    resolve(false);
                }
            }
            continue;
        }
        const bool a_live = a.step < T_enc;
        const bool b_live = b.step < T_enc;
        if (!b_live && end_of_audio) {
            if (bs.fork_completed) {
                start_guard();
            } else {
                resolve(false);
            }
            continue;
        }
        if (!a_live && !b_live) {
            break;
        }
        const bool step_plain = a_live && (!b_live || a.step < b.step);
        if (!eval(step_plain ? a : b)) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        const int u = argmax_range(logits.data(), n_token_cls);
        if (step_plain) {
            if (!commit(a, u, a.held)) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            continue;
        }
        const int tok    = u == blank_id ? u : trie.pick(bs.fork_node, logits.data(), n_token_cls, u, true);
        const int b_from = b.step;
        if (!commit(b, tok, b.held)) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        if (tok == blank_id) {
            continue;
        }
        BoostVerdict verdict = trie.fork_advance(bs.fork_node, bs.fork_completed, tok);
        if (verdict == BoostVerdict::Continue && ++bs.fork_tokens >= k_boost_fork_tokens) {
            verdict = bs.fork_completed ? BoostVerdict::Accept : BoostVerdict::Reject;
        }
        if (verdict == BoostVerdict::Accept) {
            start_guard();
            bs.guard_next = trie.nodes[static_cast<size_t>(bs.fork_node)].n_children > 0;  // left the match
            // The token that left the match sits at guard_from; the plain
            // branch's copy of it is counted, so count this one too.
            if (b_from >= bs.guard_from && (trie.token_flags[static_cast<size_t>(tok)] & k_boost_token_word) != 0) {
                ++bs.guard_boosted;
            }
        } else if (verdict == BoostVerdict::Reject) {
            resolve(false);
        }
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "decoder (boost): %d iters, %d forks, %d accepted, T_enc=%d", iter, n_forks,
            n_accept, T_enc);
    if (iter >= max_iters) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder (boost): hit iteration cap (%d)", max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }
    if (bs.fork_open) {
        bs.plain.step -= T_enc;
        bs.boosted.step -= T_enc;
        if (bs.guard_until >= 0) {
            bs.guard_from -= T_enc;
            bs.guard_until -= T_enc;
        }
    }
    return TRANSCRIBE_OK;
}

transcribe_status decode_boosted_offline(const HostDecoderWeights & w,
                                         const float *              enc_out,
                                         int                        T_enc,
                                         int                        d_enc,
                                         int                        n_threads,
                                         const BoostTrie &          trie,
                                         std::vector<TdtToken> &    out_tokens) {
    const int        n_layers = static_cast<int>(w.predictor.lstm.size());
    GreedyCursor     main;
    BoostDecodeState bs;
    main.state.reset(n_layers, w.predictor.pred_hidden);
    main.next_state.reset(n_layers, w.predictor.pred_hidden);
    return decode_boosted(w, enc_out, T_enc, d_enc, n_threads, trie, main, bs, 0, /*end_of_audio=*/true, out_tokens);
}

// TDT beam search with shallow-fusion phrase boosting, used for offline
// decodes when phrases are set. Each hypothesis takes the greedy choice and,
// on a non-blank step, also forks to the trie's next tokens (probability at
// least k_boost_floor). Hypotheses wait at their frame; a frame's group is
// cut to k_boost_beam within k_boost_beam_margin of the best, the greedy one
// kept. A hypothesis scores log p(token) + log p(duration) plus the beam's
// boost weight times the trie's score change (BoostTrie::advance). A partial
// match is taken back when it stalls for k_boost_guard_frames or the audio
// ends; a phrase earns nothing restarting in the frame it completed in or
// repeating itself. When all hypotheses share a frame with no match open or
// fresh, the best one that did not drop words against the greedy one
// (boost_guard_passed) wins. A frame that hits max_symbols on a repeating
// tail hands the decode to greedy boosting, which carries the repeat guard.
transcribe_status decode_beam_offline(const HostDecoderWeights & w,
                                      const float *              enc_out,
                                      int                        T_enc,
                                      int                        d_enc,
                                      int                        n_threads,
                                      const BoostTrie &          trie,
                                      std::vector<TdtToken> &    out_tokens) {
    const int   n_token_cls = w.predictor.pred_vocab;
    const int   n_dur       = static_cast<int>(w.tdt_durations.size());
    const int   blank_id    = w.blank_id;
    const int   joint_h     = w.joint.joint_h;
    const int   max_sym     = w.tdt_max_symbols;
    const int   beam        = k_boost_beam;
    const float lambda      = trie.lambda * k_boost_beam_weight;

    PredGraph  pg;
    JointGraph jg;
    build_pred_graph(pg, w.predictor, resolve_decode_threads(n_threads));
    if (pg.ready) {
        build_joint_graph(jg, w.joint, pg.backend, n_token_cls);
    }
    if (!pg.ready || !jg.ready) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml decode graph build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    std::vector<float> enc_proj_all;
    if (!precompute_enc_proj_ggml(w.joint, pg.backend, enc_out, T_enc, d_enc, enc_proj_all)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: enc_proj graph failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Emitted tokens as a tree; a hypothesis points at its last one.
    struct Emitted {
        TdtToken tok;
        int      prev;
    };

    struct Hyp {
        std::shared_ptr<const LstmState> state;  // after last_token, or before it while dirty
        float                            score       = 0.0f;
        int                              step        = 0;
        int                              new_symbols = 0;
        int                              last_token  = -1;
        int                              node        = 0;
        int                              matched_at  = 0;     // frame of the last trie arc
        int                              ended_at    = -1;    // frame of the last completed phrase
        int                              last_end    = -1;    // that phrase's node until a plain word follows
        int                              words       = 0;
        bool                             plain       = true;  // took only greedy choices
        int                              tail        = -1;
        uint64_t                         hash        = 1469598103934665603ULL;
        bool                             dirty       = true;
    };

    std::vector<Emitted> emitted;
    auto                 start = std::make_shared<LstmState>();
    start->reset(static_cast<int>(w.predictor.lstm.size()), w.predictor.pred_hidden);
    std::vector<std::vector<Hyp>> at(static_cast<size_t>(T_enc));
    std::vector<Hyp>              done;
    at[0].push_back(Hyp{ start });

    // Takes back a partial match.
    auto unmatch = [&](Hyp & h) {
        const BoostTrie::Node & n = trie.nodes[static_cast<size_t>(h.node)];
        if (!n.end) {
            h.score += lambda * n.chain_backoff;
        }
        h.node = 0;
    };
    // Same-sequence hypotheses at one place merge (the better score wins).
    auto place = [](std::vector<Hyp> & dst, Hyp && h) {
        for (Hyp & o : dst) {
            if (o.hash == h.hash && o.node == h.node && o.step == h.step) {
                const bool plain = o.plain || h.plain;
                if (h.score > o.score) {
                    o = std::move(h);
                }
                o.plain = plain;
                return;
            }
        }
        dst.push_back(std::move(h));
    };
    auto by_score = [](const Hyp & a, const Hyp & b) {
        return a.score > b.score;
    };
    // The best hypothesis that did not drop words against the greedy one (boost_guard_passed).
    auto best_of = [&](std::vector<Hyp> & v) -> Hyp * {
        int plain_words = 0;
        for (const Hyp & h : v) {
            plain_words = h.plain ? h.words : plain_words;
        }
        Hyp * best = nullptr;
        for (Hyp & h : v) {
            if (boost_guard_passed(plain_words, h.words) && (best == nullptr || h.score > best->score)) {
                best = &h;
            }
        }
        return best;
    };
    // Whether a TDT frame's tokens, ending in `tail`, loop.
    auto loops = [&](int tail, int step) {
        std::vector<int32_t> ids;
        for (int e = tail; e >= 0 && emitted[static_cast<size_t>(e)].tok.step_at_emit == step;
             e     = emitted[static_cast<size_t>(e)].prev) {
            ids.push_back(emitted[static_cast<size_t>(e)].tok.id);
        }
        std::reverse(ids.begin(), ids.end());
        const RepeatBar bar = { max_sym / 2, 2, 0, max_sym, 2 };
        return repeating_tail_block(ids.data(), static_cast<int>(ids.size()), bar) != 0;
    };

    const float        log_floor = std::log(k_boost_floor);
    const float        log_n     = std::log(static_cast<float>(n_token_cls));
    const size_t       hb        = static_cast<size_t>(w.predictor.pred_hidden) * sizeof(float);
    std::vector<float> scratch_x, logits(static_cast<size_t>(w.joint.joint_n)), probs(static_cast<size_t>(n_token_cls));
    std::vector<Hyp>   cur, next;
    std::vector<int>   cand;
    const int          max_rounds = 16 * T_enc + 1024;
    int                max_jump   = 1;
    for (const int d : w.tdt_durations) {
        max_jump = std::max(max_jump, d);
    }
    int rounds = 0;

    for (int t = 0; t < T_enc; ++t) {
        cur.swap(at[static_cast<size_t>(t)]);
        at[static_cast<size_t>(t)].clear();
        // With every hypothesis here and no match open or fresh, the best one is the decode.
        bool settled = done.empty();
        for (int s = t + 1; settled && s < std::min(T_enc, t + max_jump + 1); ++s) {
            settled = at[static_cast<size_t>(s)].empty();
        }
        for (const Hyp & h : cur) {
            settled = settled && (h.node == 0 || trie.nodes[static_cast<size_t>(h.node)].end) &&
                      t - h.matched_at > k_boost_guard_frames;
        }
        if (settled && cur.size() > 1) {
            Hyp keep = std::move(*best_of(cur));
            cur.assign(1, std::move(keep));
        }
        if (settled && !cur.empty()) {
            cur[0].plain = true;  // the greedy baseline from here on
        }
        while (!cur.empty()) {
            if (++rounds > max_rounds) {
                log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder (beam): hit round cap (%d)", max_rounds);
                return TRANSCRIBE_ERR_BACKEND;
            }
            std::sort(cur.begin(), cur.end(), by_score);
            size_t keep = 1;
            while (keep < cur.size() && static_cast<int>(keep) < beam &&
                   cur[keep].score >= cur[0].score - k_boost_beam_margin) {
                ++keep;
            }
            for (size_t i = keep; i < cur.size(); ++i) {
                if (cur[i].plain) {  // the greedy one stays for the word check
                    std::swap(cur[keep++], cur[i]);
                    break;
                }
            }
            cur.resize(keep);
            next.clear();
            for (Hyp & h : cur) {
                if (h.dirty) {
                    auto fresh = std::make_shared<LstmState>(*h.state);
                    if (predictor_step_ggml(w.predictor, pg, h.last_token, *h.state, *fresh, scratch_x) == nullptr) {
                        return TRANSCRIBE_ERR_BACKEND;
                    }
                    h.state = std::move(fresh);
                    h.dirty = false;
                }
                float mean = 0.0f;
                ggml_backend_tensor_set(jg.pred_in, h.state->h.back().data(), 0, hb);
                ggml_backend_tensor_set(jg.enc_in, enc_proj_all.data() + static_cast<size_t>(t) * joint_h, 0,
                                        static_cast<size_t>(joint_h) * sizeof(float));
                if (const ggml_status gs = ggml_backend_graph_compute(jg.backend, jg.graph);
                    gs != GGML_STATUS_SUCCESS) {
                    log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: joint compute failed (%d)",
                            static_cast<int>(gs));
                    return TRANSCRIBE_ERR_BACKEND;
                }
                ggml_backend_tensor_get(jg.logits, logits.data(), 0, logits.size() * sizeof(float));
                ggml_backend_tensor_get(jg.probs, probs.data(), 0, probs.size() * sizeof(float));
                ggml_backend_tensor_get(jg.mean, &mean, 0, sizeof(float));
                const float * z = logits.data();
                const float * p = probs.data();

                const int   top  = argmax_range(z, n_token_cls);
                const float lse  = z[top] - std::log(p[top]);
                const float conf = 1.0f - (lse - mean) / log_n;

                const float * zd = z + n_token_cls;
                const int     d  = argmax_range(zd, n_dur);
                double        sd = 0.0;
                for (int k = 0; k < n_dur; ++k) {
                    sd += std::exp(static_cast<double>(zd[k] - zd[d]));
                }
                const float lpd = -static_cast<float>(std::log(sd));
                const int   dur = w.tdt_durations[static_cast<size_t>(d)];

                auto land = [&](Hyp && c) {
                    if (c.node != 0 && c.step - c.matched_at > k_boost_guard_frames) {
                        unmatch(c);
                    }
                    if (c.step >= T_enc) {
                        place(done, std::move(c));
                    } else if (c.step == t) {
                        place(next, std::move(c));
                    } else {
                        place(at[static_cast<size_t>(c.step)], std::move(c));
                    }
                };

                // The greedy choice; a non-blank one also forks to the trie's next tokens.
                cand.clear();
                if (top == blank_id) {
                    Hyp b = h;
                    b.score += z[blank_id] - lse + lpd;
                    b.step += std::max(dur, 1);  // a zero-duration blank ends up one frame on
                    b.new_symbols = 0;
                    land(std::move(b));
                } else {
                    cand.push_back(top);
                }
                for (int f = h.node; top != blank_id && h.ended_at != t; f = trie.nodes[static_cast<size_t>(f)].fail) {
                    const BoostTrie::Node & fn = trie.nodes[static_cast<size_t>(f)];
                    for (int k = 0; k < fn.n_children; ++k) {
                        const int tok = trie.child_tok[static_cast<size_t>(fn.first_child + k)];
                        if (z[tok] - lse >= log_floor && std::find(cand.begin(), cand.end(), tok) == cand.end() &&
                            (trie.token_flags[static_cast<size_t>(tok)] & k_boost_token_special) == 0) {
                            cand.push_back(tok);
                        }
                    }
                    if (f == 0) {
                        break;
                    }
                }

                for (const int tok : cand) {
                    Hyp c = h;
                    c.score += z[tok] - lse + lpd;
                    const float             gain = trie.advance(c.node, tok);
                    const BoostTrie::Node & n    = trie.nodes[static_cast<size_t>(c.node)];
                    if (c.node != 0 && h.ended_at == t && trie.child(0, tok) == c.node) {
                        c.score += lambda * (gain - n.score);  // no restart in the frame a phrase completed in
                        c.node = 0;
                    } else if (n.end && c.node == h.last_end) {
                        c.score += lambda * (gain - n.score);  // no bonus for a phrase repeating itself
                        c.node = 0;
                    } else {
                        c.score += lambda * gain;
                    }
                    if (c.node != 0) {
                        c.matched_at = t;
                        if (n.end) {
                            c.ended_at = t;
                            c.last_end = c.node;
                        }
                    } else if ((trie.token_flags[static_cast<size_t>(tok)] & k_boost_token_word) != 0) {
                        c.last_end = -1;
                    }
                    c.words += (trie.token_flags[static_cast<size_t>(tok)] & k_boost_token_word) != 0;
                    c.plain      = h.plain && tok == top;
                    c.dirty      = true;
                    c.last_token = tok;
                    c.hash       = (c.hash ^ static_cast<uint64_t>(tok + 1)) * 1099511628211ULL;
                    TdtToken e;
                    e.id              = tok;
                    e.p               = conf;
                    e.step_at_emit    = t;
                    e.duration_frames = dur;
                    emitted.push_back({ e, h.tail });
                    c.tail = static_cast<int>(emitted.size()) - 1;
                    if (dur > 0) {
                        c.step += dur;
                        c.new_symbols = 0;
                    } else if (max_sym > 0 && ++c.new_symbols >= max_sym) {
                        if (repetition_guard_enabled() && loops(c.tail, t)) {
                            // Rare; greedy boosting trims the loop and rewinds.
                            return decode_boosted_offline(w, enc_out, T_enc, d_enc, n_threads, trie, out_tokens);
                        }
                        c.step += 1;
                        c.new_symbols = 0;
                    }
                    land(std::move(c));
                }
            }
            cur.swap(next);
        }
    }

    for (Hyp & h : done) {
        unmatch(h);
    }
    const Hyp *  best  = done.empty() ? nullptr : best_of(done);
    const size_t first = out_tokens.size();
    for (int e = best != nullptr ? best->tail : -1; e >= 0; e = emitted[static_cast<size_t>(e)].prev) {
        out_tokens.push_back(emitted[static_cast<size_t>(e)].tok);
    }
    std::reverse(out_tokens.begin() + static_cast<std::ptrdiff_t>(first), out_tokens.end());
    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "decoder (beam): %d rounds, %zu tokens, T_enc=%d", rounds,
            out_tokens.size() - first, T_enc);
    return TRANSCRIBE_OK;
}

}  // namespace

// ---------------------------------------------------------------------------
// Greedy decode driver
// ---------------------------------------------------------------------------

transcribe_status decode_tdt_greedy(const HostDecoderWeights & w,
                                    const float *              enc_out,
                                    int                        T_enc,
                                    int                        d_enc,
                                    int                        n_threads,
                                    const BoostTrie *          boost,
                                    std::vector<TdtToken> &    out_tokens) {
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.joint.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder: enc d_model mismatch (got %d, "
                "expected %d)",
                d_enc, w.joint.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (boost != nullptr && !boost->empty()) {
        return decode_beam_offline(w, enc_out, T_enc, d_enc, n_threads, *boost, out_tokens);
    }

    const int64_t t_setup_start = ggml_time_us();
    const int     nt            = resolve_decode_threads(n_threads);
    const int     n_layers      = static_cast<int>(w.predictor.lstm.size());
    const int     H             = w.predictor.pred_hidden;
    const int     n_token_cls   = w.predictor.pred_vocab;  // == vocab_size + 1
    const int     n_dur         = static_cast<int>(w.tdt_durations.size());
    const int     blank_id      = w.blank_id;

    // All-ggml decode on ONE shared threadpool: PredGraph owns the
    // backend + pool; enc_proj and the joint graph borrow it. pg is
    // declared first so it is destroyed LAST (jg's buffer lives on pg's
    // backend). Built per decode call, reentrant. A build failure is a
    // hard decode error.
    PredGraph  pg;
    JointGraph jg;
    build_pred_graph(pg, w.predictor, nt);
    if (pg.ready) {
        build_joint_graph(jg, w.joint, pg.backend);
    }
    if (!pg.ready || !jg.ready) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml decode graph build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    const int64_t t_setup_us = ggml_time_us() - t_setup_start;

    // Two LSTM states, both pre-sized: `state` is the committed
    // state we read from each iteration; `next_state` is where the
    // predictor writes the new step's outputs. On a non-blank
    // emission we swap; on a blank emission we just discard the
    // contents of `next_state` (overwritten on the next iteration).
    LstmState state;
    LstmState next_state;
    state.reset(n_layers, H);
    next_state.reset(n_layers, H);

    // Precompute encoder projections for all T_enc frames (decode-state
    // independent; one sgemm before the loop). See precompute_enc_proj_ggml.
    const int          joint_h = w.joint.joint_h;
    std::vector<float> enc_proj_all;
    const int64_t      t_enc_proj_start = ggml_time_us();
    if (!precompute_enc_proj_ggml(w.joint, pg.backend, enc_out, T_enc, d_enc, enc_proj_all)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: enc_proj graph failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    const int64_t t_enc_proj_us = ggml_time_us() - t_enc_proj_start;

    // Per-call scratch reused across every decode step.
    std::vector<float> scratch_x;
    std::vector<float> logits;

    int        last_token  = -1;  // sentinel: no previous token (start state)
    int        step        = 0;
    int        new_symbols = 0;
    FrameStart frame;

    // Runaway-protection cap; legitimate transcriptions never approach it.
    const int max_iters = 16 * T_enc + 1024;
    int       iter      = 0;
    int64_t   t_pred_us = 0, t_joint_us = 0, t_conf_us = 0;

    // On a blank emission (last_token, state) are preserved, so the
    // predictor would write the same next_state next iteration. Skip the
    // LSTM unroll and reuse the previous decoder_out; recompute on a
    // non-blank emission (state.swap + last_token change). Bit-exact with
    // the unconditional path.
    bool predictor_dirty = true;

    while (step < T_enc && iter < max_iters) {
        ++iter;

        // ----- Predictor (one LSTM step) -----
        const int64_t t0 = ggml_time_us();
        const float * decoder_out;
        if (predictor_dirty) {
            decoder_out = predictor_step_ggml(w.predictor, pg, last_token, state, next_state, scratch_x);
            if (decoder_out == nullptr) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            predictor_dirty = false;
        } else {
            decoder_out = next_state.h.back().data();
        }
        const int64_t t1 = ggml_time_us();

        // ----- Joint (using precomputed encoder projection) -----
        const float * enc_proj = enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h);
        if (!joint_step(w.joint, jg, enc_proj, decoder_out, logits)) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        const int64_t t2 = ggml_time_us();
        t_pred_us += t1 - t0;
        t_joint_us += t2 - t1;

        // ----- Argmax (token + duration) -----
        const float * token_logits    = logits.data();
        const float * duration_logits = logits.data() + n_token_cls;

        const int pred_token = argmax_range(token_logits, n_token_cls);
        const int decision   = argmax_range(duration_logits, n_dur);
        const int duration   = w.tdt_durations[static_cast<size_t>(decision)];

        // Optional dump of the first decode step (start state, encoder
        // frame 0): embed input, per-layer LSTM h, joint logits. Mirrors
        // the Python reference dumper's decode subcommand.
        if (iter == 1 && transcribe::debug::enabled()) {
            // Predictor scratch_x at start: zeros (vector of length H).
            const long long s_h = H;
            transcribe::debug::dump_host_f32("dec.embed.0", scratch_x.data(), s_h, &s_h, 1, "decoder.embed");

            for (int layer = 0; layer < n_layers; ++layer) {
                char name_h[64];
                char name_c[64];
                std::snprintf(name_h, sizeof(name_h), "dec.lstm.%d.h.0", layer);
                std::snprintf(name_c, sizeof(name_c), "dec.lstm.%d.c.0", layer);
                transcribe::debug::dump_host_f32(name_h, next_state.h[layer].data(), s_h, &s_h, 1, "decoder.lstm");
                transcribe::debug::dump_host_f32(name_c, next_state.c[layer].data(), s_h, &s_h, 1, "decoder.lstm");
            }

            const long long s_n = w.joint.joint_n;
            transcribe::debug::dump_host_f32("dec.joint.0", logits.data(), s_n, &s_n, 1, "decoder.joint");
        }

        // ----- TDT emit + state advance -----
        const bool is_blank = (pred_token == blank_id);
        if (!is_blank) {
            const int64_t tc0 = ggml_time_us();
            const float   p   = token_confidence(token_logits, n_token_cls);
            t_conf_us += ggml_time_us() - tc0;
            TdtToken tok;
            tok.id              = pred_token;
            tok.p               = p;
            tok.step_at_emit    = step;
            tok.duration_frames = duration;
            if (duration == 0 && (out_tokens.empty() || out_tokens.back().step_at_emit != step)) {
                frame.state      = state;
                frame.last_token = last_token;
            }
            out_tokens.push_back(tok);

            last_token = pred_token;
            std::swap(state, next_state);  // commit
            predictor_dirty = true;
        }

        // Step / stuck advance. Matches the reference:
        //   step += duration
        //   new_symbols += 1
        //   if duration != 0: new_symbols = 0
        //   elif max_symbols and new_symbols >= max_symbols:
        //       step += 1; new_symbols = 0
        step += duration;
        new_symbols += 1;
        if (duration != 0) {
            new_symbols = 0;
        } else if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
            if (trim_stuck_frame(out_tokens, step, w.tdt_max_symbols)) {
                if (!rewind_stuck_frame(w.predictor, pg, out_tokens, step, frame, state, last_token, next_state,
                                        scratch_x)) {
                    return TRANSCRIBE_ERR_BACKEND;
                }
                predictor_dirty = true;
            }
            step += 1;
            new_symbols = 0;
        } else if (is_blank && w.tdt_max_symbols > 0) {
            // A blank with duration 0 leaves (step, last_token, state)
            // unchanged, so every following iteration is identical until
            // max_symbols forces the step to advance. Fast-forward those
            // repeated no-op loops without changing the observable result.
            const int skip = w.tdt_max_symbols - new_symbols;
            if (skip > 0 && iter + skip < max_iters) {
                iter += skip;
                step += 1;
                new_symbols = 0;
            }
        }
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
            "decoder: %d iters, %d tokens, T_enc=%d  "
            "setup=%.1f ms  enc_proj=%.1f ms  pred=%.1f ms  joint=%.1f ms  conf=%.1f ms  "
            "total=%.1f ms  per_iter=%.0f us",
            iter, static_cast<int>(out_tokens.size()), T_enc, t_setup_us / 1000.0, t_enc_proj_us / 1000.0,
            t_pred_us / 1000.0, t_joint_us / 1000.0, t_conf_us / 1000.0,
            (t_setup_us + t_enc_proj_us + t_pred_us + t_joint_us + t_conf_us) / 1000.0,
            static_cast<double>(t_pred_us + t_joint_us) / std::max(iter, 1));

    if (iter >= max_iters) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder: hit iteration cap (%d) — pathological "
                "logits or loop bug",
                max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// RNNT greedy decode
// ---------------------------------------------------------------------------
//
// Same predictor + joint forward as TDT, but the joint emits exactly
// `vocab+1` logits (no duration extras). Step rule:
//
//   per iteration:
//     run predictor (one LSTM step)
//     run joint -> token_logits[vocab+1]
//     argmax -> pred_token
//     if pred_token == blank:
//         step += 1
//         new_symbols = 0
//     else:
//         emit token, swap predictor state, last_token = pred_token
//         new_symbols += 1
//         if max_symbols and new_symbols >= max_symbols:
//             step += 1; new_symbols = 0   # break out of stuck-on-frame loop
//
// Mirrors NeMo's `RNNTGreedyDecodeInfer.step_per_frame` (without
// duration). Matches the reference dump points (`dec.embed.0`,
// `dec.lstm.{layer}.{h,c}.0`, `dec.joint.0`) emitted on iter 1.

transcribe_status decode_rnnt_greedy(const HostDecoderWeights & w,
                                     const float *              enc_out,
                                     int                        T_enc,
                                     int                        d_enc,
                                     int                        n_threads,
                                     const BoostTrie *          boost,
                                     std::vector<TdtToken> &    out_tokens) {
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.joint.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (rnnt): enc d_model mismatch (got %d, "
                "expected %d)",
                d_enc, w.joint.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (boost != nullptr && !boost->empty()) {
        return decode_boosted_offline(w, enc_out, T_enc, d_enc, n_threads, *boost, out_tokens);
    }

    const int nt          = resolve_decode_threads(n_threads);
    const int n_layers    = static_cast<int>(w.predictor.lstm.size());
    const int H           = w.predictor.pred_hidden;
    const int n_token_cls = w.predictor.pred_vocab;  // == vocab + 1
    const int blank_id    = w.blank_id;

    // Per-call decode graphs (see decode_tdt_greedy for the lifecycle).
    PredGraph  pg;
    JointGraph jg;
    build_pred_graph(pg, w.predictor, nt);
    if (pg.ready) {
        build_joint_graph(jg, w.joint, pg.backend);
    }
    if (!pg.ready || !jg.ready) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml decode graph build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    LstmState state;
    LstmState next_state;
    state.reset(n_layers, H);
    next_state.reset(n_layers, H);

    // Precompute encoder projections for all T_enc frames (same as TDT).
    const int          joint_h = w.joint.joint_h;
    std::vector<float> enc_proj_all;
    const int64_t      t_enc_proj_start = ggml_time_us();
    if (!precompute_enc_proj_ggml(w.joint, pg.backend, enc_out, T_enc, d_enc, enc_proj_all)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: enc_proj graph failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    const int64_t t_enc_proj_us = ggml_time_us() - t_enc_proj_start;

    std::vector<float> scratch_x;
    std::vector<float> logits;

    int last_token  = -1;
    int step        = 0;
    int new_symbols = 0;

    const int max_iters = 16 * T_enc + 1024;
    int       iter      = 0;
    int64_t   t_pred_us = 0, t_joint_us = 0, t_conf_us = 0;

    // Predictor-step cache (see decode_tdt_greedy). Bit-exact with the
    // unconditional path; elides 60-80% of predictor work on a typical
    // RNN-T iters/token ratio.
    bool predictor_dirty = true;

    while (step < T_enc && iter < max_iters) {
        ++iter;

        const int64_t t0 = ggml_time_us();
        const float * decoder_out;
        if (predictor_dirty) {
            decoder_out = predictor_step_ggml(w.predictor, pg, last_token, state, next_state, scratch_x);
            if (decoder_out == nullptr) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            predictor_dirty = false;
        } else {
            decoder_out = next_state.h.back().data();
        }
        const int64_t t1 = ggml_time_us();

        const float * enc_proj = enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h);
        if (!joint_step(w.joint, jg, enc_proj, decoder_out, logits)) {
            return TRANSCRIBE_ERR_BACKEND;
        }
        const int64_t t2 = ggml_time_us();
        t_pred_us += t1 - t0;
        t_joint_us += t2 - t1;

        // RNNT joint output is just `n_token_cls` floats (no duration extras).
        const float * token_logits = logits.data();
        const int     pred_token   = argmax_range(token_logits, n_token_cls);

        if (iter == 1 && transcribe::debug::enabled()) {
            const long long s_h = H;
            transcribe::debug::dump_host_f32("dec.embed.0", scratch_x.data(), s_h, &s_h, 1, "decoder.embed");
            for (int layer = 0; layer < n_layers; ++layer) {
                char name_h[64];
                char name_c[64];
                std::snprintf(name_h, sizeof(name_h), "dec.lstm.%d.h.0", layer);
                std::snprintf(name_c, sizeof(name_c), "dec.lstm.%d.c.0", layer);
                transcribe::debug::dump_host_f32(name_h, next_state.h[layer].data(), s_h, &s_h, 1, "decoder.lstm");
                transcribe::debug::dump_host_f32(name_c, next_state.c[layer].data(), s_h, &s_h, 1, "decoder.lstm");
            }
            const long long s_n = w.joint.joint_n;
            transcribe::debug::dump_host_f32("dec.joint.0", logits.data(), s_n, &s_n, 1, "decoder.joint");
        }

        const bool is_blank = (pred_token == blank_id);
        if (is_blank) {
            step += 1;
            new_symbols = 0;
        } else {
            const int64_t tc0 = ggml_time_us();
            const float   p   = token_confidence(token_logits, n_token_cls);
            t_conf_us += ggml_time_us() - tc0;
            TdtToken tok;
            tok.id              = pred_token;
            tok.p               = p;
            tok.step_at_emit    = step;
            tok.duration_frames = 1;
            out_tokens.push_back(tok);

            last_token = pred_token;
            std::swap(state, next_state);
            predictor_dirty = true;

            new_symbols += 1;
            if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
                step += 1;
                new_symbols = 0;
            }
        }
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG,
            "decoder (rnnt): %d iters, %d tokens, T_enc=%d  "
            "enc_proj=%.1f ms  pred=%.1f ms  joint=%.1f ms  conf=%.1f ms  "
            "total=%.1f ms",
            iter, static_cast<int>(out_tokens.size()), T_enc, t_enc_proj_us / 1000.0, t_pred_us / 1000.0,
            t_joint_us / 1000.0, t_conf_us / 1000.0, (t_enc_proj_us + t_pred_us + t_joint_us + t_conf_us) / 1000.0);

    if (iter >= max_iters) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (rnnt): hit iteration cap (%d) — "
                "pathological logits or loop bug",
                max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// RNNT greedy decode — streaming
// ---------------------------------------------------------------------------
//
// Same algorithm as decode_rnnt_greedy but with externally-managed
// LSTM state (state_io) and previous token (last_token_io). The
// chunk's encoder frames are decoded in stream-wide coordinates
// (step_at_emit = frame_offset + local_step). No timing log.
transcribe_status decode_rnnt_greedy_streaming(const HostDecoderWeights & w,
                                               const float *              enc_out,
                                               int                        T_enc_new,
                                               int                        d_enc,
                                               LstmState &                state_io,
                                               int &                      last_token_io,
                                               int                        frame_offset,
                                               int                        n_threads,
                                               const BoostTrie *          boost,
                                               BoostDecodeState &         boost_io,
                                               std::vector<TdtToken> &    out_tokens) {
    if (enc_out == nullptr || T_enc_new <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    // Pure RNN-T greedy only: a TDT head's durations would be silently
    // ignored and a CTC head carries no predictor state. Fail loud rather
    // than mis-decode a hypothetical future TDT/CTC streaming model.
    if (w.head_kind != HostHeadKind::RNNT) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (rnnt-stream): streaming decode "
                "requires an RNN-T head; this model's head is not "
                "RNN-T");
        return TRANSCRIBE_ERR_NOT_IMPLEMENTED;
    }
    if (d_enc != w.joint.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (rnnt-stream): enc d_model mismatch "
                "(got %d, expected %d)",
                d_enc, w.joint.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int nt          = resolve_decode_threads(n_threads);
    const int n_layers    = static_cast<int>(w.predictor.lstm.size());
    const int H           = w.predictor.pred_hidden;
    const int n_token_cls = w.predictor.pred_vocab;
    const int blank_id    = w.blank_id;

    // Validate state_io shape; reset if degenerate.
    if (static_cast<int>(state_io.h.size()) != n_layers || static_cast<int>(state_io.c.size()) != n_layers) {
        state_io.reset(n_layers, H);
        last_token_io = -1;
    }
    for (int l = 0; l < n_layers; ++l) {
        if (static_cast<int>(state_io.h[l].size()) != H || static_cast<int>(state_io.c[l].size()) != H) {
            state_io.reset(n_layers, H);
            last_token_io = -1;
            break;
        }
    }

    if (boost != nullptr && !boost->empty()) {
        GreedyCursor main;
        main.state      = state_io;
        main.last_token = last_token_io;
        main.next_state.reset(n_layers, H);
        const transcribe_status st = decode_boosted(w, enc_out, T_enc_new, d_enc, n_threads, *boost, main, boost_io,
                                                    frame_offset, /*end_of_audio=*/false, out_tokens);
        if (st == TRANSCRIBE_OK) {
            state_io      = std::move(main.state);
            last_token_io = main.last_token;
        }
        return st;
    }

    // Per-call decode graphs (see decode_tdt_greedy for the lifecycle).
    PredGraph  pg;
    JointGraph jg;
    build_pred_graph(pg, w.predictor, nt);
    if (pg.ready) {
        build_joint_graph(jg, w.joint, pg.backend);
    }
    if (!pg.ready || !jg.ready) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder: ggml decode graph build failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Precompute encoder projections for this chunk only.
    const int          joint_h = w.joint.joint_h;
    std::vector<float> enc_proj_all;
    if (!precompute_enc_proj_ggml(w.joint, pg.backend, enc_out, T_enc_new, d_enc, enc_proj_all)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder (rnnt-stream): enc_proj graph failed");
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Working state. On return, `state` (committed after the last
    // non-blank emission, or unchanged for an all-blank chunk) and
    // last_token are copied back into state_io / last_token_io.
    LstmState state = state_io;
    LstmState next_state;
    next_state.reset(n_layers, H);

    std::vector<float> scratch_x;
    std::vector<float> logits;

    int last_token  = last_token_io;
    int step        = 0;
    int new_symbols = 0;

    const int max_iters = 16 * T_enc_new + 1024;
    int       iter      = 0;

    bool predictor_dirty = true;

    while (step < T_enc_new && iter < max_iters) {
        ++iter;

        const float * decoder_out;
        if (predictor_dirty) {
            decoder_out = predictor_step_ggml(w.predictor, pg, last_token, state, next_state, scratch_x);
            if (decoder_out == nullptr) {
                return TRANSCRIBE_ERR_BACKEND;
            }
            predictor_dirty = false;
        } else {
            decoder_out = next_state.h.back().data();
        }

        const float * enc_proj = enc_proj_all.data() + static_cast<size_t>(step) * static_cast<size_t>(joint_h);
        if (!joint_step(w.joint, jg, enc_proj, decoder_out, logits)) {
            return TRANSCRIBE_ERR_BACKEND;
        }

        const float * token_logits = logits.data();
        const int     pred_token   = argmax_range(token_logits, n_token_cls);

        const bool is_blank = (pred_token == blank_id);
        if (is_blank) {
            step += 1;
            new_symbols = 0;
        } else {
            const float p = token_confidence(token_logits, n_token_cls);
            TdtToken    tok;
            tok.id              = pred_token;
            tok.p               = p;
            tok.step_at_emit    = frame_offset + step;
            tok.duration_frames = 1;
            out_tokens.push_back(tok);

            last_token = pred_token;
            std::swap(state, next_state);
            predictor_dirty = true;

            new_symbols += 1;
            if (w.tdt_max_symbols > 0 && new_symbols >= w.tdt_max_symbols) {
                step += 1;
                new_symbols = 0;
            }
        }
    }

    if (iter >= max_iters) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet decoder (rnnt-stream): hit iteration cap (%d)", max_iters);
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Commit final state. After the loop, `state` is the post-last-
    // non-blank state (the predictor's "current" since the last emit).
    state_io      = std::move(state);
    last_token_io = last_token;
    return TRANSCRIBE_OK;
}

void resolve_boost_fork(BoostDecodeState &      boost_io,
                        LstmState &             state_io,
                        int &                   last_token_io,
                        std::vector<TdtToken> & out_tokens) {
    if (!boost_io.fork_open) {
        return;
    }
    const bool accept  = boost_io.guard_until >= 0 ? boost_guard_passed(boost_io.guard_plain, boost_io.guard_boosted) :
                                                     boost_io.fork_completed;
    GreedyCursor & win = accept ? boost_io.boosted : boost_io.plain;
    out_tokens.insert(out_tokens.end(), win.held.begin(), win.held.end());
    state_io      = win.state;
    last_token_io = win.last_token;
    boost_io      = BoostDecodeState{};
}

// CTC greedy decode.
//
// Per-frame: logits[t] = W @ enc[t] + b -> log_softmax -> argmax.
// Collapse: drop adjacent duplicates ("aaab" -> "ab"), then drop blanks
// (standard CTC greedy, cf. NeMo GreedyCTCInfer).
transcribe_status decode_ctc_greedy(const HostDecoderWeights & w,
                                    const float *              enc_out,
                                    int                        T_enc,
                                    int                        d_enc,
                                    int                        n_threads,
                                    std::vector<TdtToken> &    out_tokens) {
    if (enc_out == nullptr || T_enc <= 0 || d_enc <= 0) {
        return TRANSCRIBE_ERR_INVALID_ARG;
    }
    if (d_enc != w.ctc_head.d_enc) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet decoder (ctc): enc d_model mismatch "
                "(got %d, expected %d)",
                d_enc, w.ctc_head.d_enc);
        return TRANSCRIBE_ERR_INVALID_ARG;
    }

    const int n_classes = w.ctc_head.n_classes;
    const int blank_id  = w.ctc_head.blank_id;

    // Compute T_enc frames of logits: [T_enc, n_classes] = enc_out[T_enc, d_enc] @ W^T + b.
    std::vector<float> logits_all(static_cast<size_t>(T_enc) * static_cast<size_t>(n_classes));

    const int64_t t_proj_start = ggml_time_us();
#if TRANSCRIBE_HAS_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, T_enc, n_classes, d_enc, 1.0f, enc_out, d_enc,
                w.ctc_head.weight.data(), d_enc, 0.0f, logits_all.data(), n_classes);
#else
    // No BLAS: project all T frames in parallel over parallel_for_all
    // (rows within a frame are a serial, auto-vectorized dot). Bias folded
    // in below.
    {
        const int     nt = resolve_decode_threads(n_threads);
        const float * Wc = w.ctc_head.weight.data();
        transcribe::parallel_for_all(T_enc, nt, [&](int t) {
            const float * frame = enc_out + static_cast<size_t>(t) * static_cast<size_t>(d_enc);
            float *       row   = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
            for (int r = 0; r < n_classes; ++r) {
                const float * wr  = Wc + static_cast<size_t>(r) * static_cast<size_t>(d_enc);
                float         acc = 0.0f;
                for (int c = 0; c < d_enc; ++c) {
                    acc += wr[c] * frame[c];
                }
                row[r] = acc;
            }
            return true;
        });
    }
#endif
    for (int t = 0; t < T_enc; ++t) {
        float * row = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
        for (int c = 0; c < n_classes; ++c) {
            row[c] += w.ctc_head.bias[static_cast<size_t>(c)];
        }
    }
    const int64_t t_proj_us = ggml_time_us() - t_proj_start;

    // Per-frame log-softmax in place. The reference dumps log-probs,
    // not raw logits.
    for (int t = 0; t < T_enc; ++t) {
        float * row   = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
        float   max_v = row[0];
        for (int c = 1; c < n_classes; ++c) {
            if (row[c] > max_v) {
                max_v = row[c];
            }
        }
        double sum = 0.0;
        for (int c = 0; c < n_classes; ++c) {
            sum += std::exp(static_cast<double>(row[c] - max_v));
        }
        const float log_sum = static_cast<float>(std::log(sum)) + max_v;
        for (int c = 0; c < n_classes; ++c) {
            row[c] -= log_sum;
        }
    }

    // Optional dumps. Frame 0 is shape [n_classes]; full is
    // [T_enc, n_classes] in row-major (== ggml ne fast-to-slow
    // [n_classes, T_enc]).
    if (transcribe::debug::enabled()) {
        const long long s_n = n_classes;
        transcribe::debug::dump_host_f32("dec.ctc.logprobs.0", logits_all.data(), s_n, &s_n, 1,
                                         "decoder.ctc.logprobs.0");
        const long long shape_full[2] = { T_enc, n_classes };
        transcribe::debug::dump_host_f32("dec.ctc.logprobs", logits_all.data(),
                                         static_cast<long long>(logits_all.size()), shape_full, 2,
                                         "decoder.ctc.logprobs");
    }

    // Greedy collapse: drop runs of identical labels, then drop blanks.
    // Per-emit `step_at_emit` is the encoder frame at which the label
    // first changed; duration_frames = 1.
    int prev_label = -1;
    for (int t = 0; t < T_enc; ++t) {
        const float * row   = logits_all.data() + static_cast<size_t>(t) * static_cast<size_t>(n_classes);
        const int     label = argmax_range(row, n_classes);
        if (label == prev_label) {
            continue;  // run-length collapse
        }
        prev_label = label;
        if (label == blank_id) {
            continue;
        }
        TdtToken tok;
        tok.id              = label;
        // CTC greedy confidence = exp(top log-prob); this is the
        // model's own probability for the chosen label, in [0, 1].
        tok.p               = std::exp(row[label]);
        tok.step_at_emit    = t;
        tok.duration_frames = 1;
        out_tokens.push_back(tok);
    }

    log_msg(TRANSCRIBE_LOG_LEVEL_DEBUG, "decoder (ctc): T_enc=%d  proj=%.1f ms  emitted=%d", T_enc, t_proj_us / 1000.0,
            static_cast<int>(out_tokens.size()));

    return TRANSCRIBE_OK;
}

}  // namespace transcribe::parakeet
