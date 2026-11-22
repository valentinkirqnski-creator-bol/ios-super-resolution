// srsr_cpu.cpp -- SRSR backend for the PC test harness (NOT in the iOS target).
// Runs the SAME SRSRKernels.metal source on the CPU through srsr_cpu_shim.h:
// one adapter per kernel turns the bound slots into the kernel's arguments and
// loops the grid (OpenMP). Batches execute synchronously at dispatch time.
#include "srsr_cpu_shim.h"
#include "SRSRKernels.metal"
#undef kernel
#undef thread
#undef device
#undef constant

#include "srsr_backend.h"
#include <cstdio>
#include <memory>
#include <vector>

namespace hhsr {
namespace srsr {
namespace {

struct CTex : Tex {
    metal::CpuTex t;
    CTex(metal::CpuFmt f, int w_, int h_) : t(f, w_, h_) {}
};
struct CBuf : Buf {
    std::vector<uint8_t> d;
    void* contents() override { return d.data(); }
};

static metal::CpuFmt to_cpu(Fmt f) {
    switch (f) {
        case Fmt::R8Unorm: return metal::CpuFmt::R8Unorm;
        case Fmt::R32Float: return metal::CpuFmt::R32Float;
        case Fmt::RGBA16Float: return metal::CpuFmt::RGBA16Float;
        case Fmt::RGBA16Uint: return metal::CpuFmt::RGBA16Uint;
        case Fmt::RGBA32Float: return metal::CpuFmt::RGBA32Float;
    }
    return metal::CpuFmt::R32Float;
}

class CpuBackend : public Backend {
public:
    const char* name() const override { return "CPU-emulated Metal"; }
    std::unique_ptr<Tex> make_tex(Fmt f, int w, int h, const char*) override {
        auto t = std::make_unique<CTex>(to_cpu(f), w, h);
        t->w = w; t->h = h; t->fmt = f;
        return t;
    }
    std::unique_ptr<Buf> make_buf(size_t bytes, const char*) override {
        auto b = std::make_unique<CBuf>();
        b->d.assign(bytes, 0);
        b->size = bytes;
        return b;
    }
    bool begin() override { return true; }
    void tex(int slot, Tex* t) override { if (slot >= 0 && slot < 80) tex_[slot] = static_cast<CTex*>(t); }
    void bytes(int slot, const void* p, size_t n) override {
        if (slot < 0 || slot >= 8) return;
        bytes_[slot].assign((const uint8_t*)p, (const uint8_t*)p + n);
    }
    void buf(int slot, Buf* b) override { if (slot >= 0 && slot < 8) buf_[slot] = b; }
    uint64_t commit(const char*) override { return ++ticket_; }
    bool wait(uint64_t) override { return !failed_; }
    bool wait_all() override { return !failed_; }
    bool failed() const override { return failed_; }

    void dispatch_bm(int tx, int ty, int, size_t, size_t) override {
        const auto& u = U<SRSRBlockMatchingUniforms>(0);
        auto ref = R(0), mov = R(1), ini = R(2);
        auto out = Wt(3);
#pragma omp parallel
        {
            std::vector<float> tile(64 * 64);
#pragma omp for schedule(dynamic, 8)
            for (int i = 0; i < tx * ty; ++i)
                srsr_block_matching_tile(ref, mov, ini, out, u, uint2((uint)(i % tx), (uint)(i / tx)), tile.data());
        }
    }

    void dispatch(K k, int w, int h) override {
        switch (k) {
            case K::prepare: {
                const float* src = (const float*)buf_[1]->contents();
                auto q = WU(1); auto g = Wt(2); const auto& u = U<SRSRPrepareUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_prepare(src, q, g, u, p); });
                break;
            }
            case K::unblocker: {
                auto q = RU(0); auto o = Wt(1); const auto& u = U<SRSRUnblockerUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_unblocker(q, o, u, p); });
                break;
            }
            case K::clear_flow: {
                auto o = Wt(0); const auto& u = U<SRSRFlowMedianUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_clear_flow(o, u, p); });
                break;
            }
            case K::downsample: {
                auto a = R(0); auto o = Wt(1); const auto& u = U<SRSRDownsampleUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_downsample(a, o, u, p); });
                break;
            }
            case K::gradients: {
                auto a = R(0); auto gx = Wt(1), gy = Wt(2); const auto& u = U<SRSRGradientUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_gradients(a, gx, gy, u, p); });
                break;
            }
            case K::gradient_tile_sums: {
                auto gx = R(0), gy = R(1); auto o = Wt(2); const auto& u = U<SRSRGradientUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_gradient_tile_sums(gx, gy, o, u, p); });
                break;
            }
            case K::block_matching: break;   // dispatch_bm
            case K::ica_iteration: {
                auto a = R(0), b = R(1), c = R(2), d = R(3), e = R(4), f = R(5); auto o = Wt(6);
                const auto& u = U<SRSRICAUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_ica_iteration(a, b, c, d, e, f, o, u, p); });
                break;
            }
            case K::flow_transfer: {
                auto a = R(0), b = R(1), c = R(2); auto o = Wt(3); const auto& u = U<SRSRFlowTransferUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_flow_transfer(a, b, c, o, u, p); });
                break;
            }
            case K::flow_median: {
                auto a = R(0); auto o = Wt(1); const auto& u = U<SRSRFlowMedianUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_flow_median(a, o, u, p); });
                break;
            }
            case K::flow_pack: {
                auto a = R(0); auto o = Wt(1); const auto& u = U<SRSRFlowPackUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_flow_pack(a, o, u, p); });
                break;
            }
            case K::guide: {
                auto q = RU(0); auto g = Wt(1), c = Wt(2);
                const auto& u = U<SRSRGuideUniforms>(0); const auto& lut = U<SRSRNoiseLUT>(1);
                grid(w, h, [&](uint2 p) { srsr_guide(q, g, c, u, lut, p); });
                break;
            }
            case K::rejection: {
                auto bg = S(0), ag = S(1); auto fl = R(2); auto ub = S(3); auto o = Wt(4);
                const auto& u = U<SRSRRejectionUniforms>(0); const auto& lut = U<SRSRNoiseLUT>(1);
                grid(w, h, [&](uint2 p) { srsr_rejection(bg, ag, fl, ub, o, u, lut, p); });
                break;
            }
            case K::dilate_h: {
                auto a = R(0); auto o = Wt(1); const auto& u = U<SRSRRejectionUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_dilate_horizontal(a, o, u, p); });
                break;
            }
            case K::dilate_v: {
                auto a = R(0); auto o = Wt(1); const auto& u = U<SRSRRejectionUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_dilate_vertical(a, o, u, p); });
                break;
            }
            case K::accumulate_weights: {
                metal::array<texture2d<float, access::read>, kSRSRMaxFrames> d;
                for (int i = 0; i < kSRSRMaxFrames; ++i) d[i] = R(i);
                auto o = Wt(kSRSRMaxFrames); const auto& u = U<SRSRAccumulateUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_accumulate_weights(d, o, u, p); });
                break;
            }
            case K::merge: {
                auto o = Wt(0); auto acc = S(1);
                metal::array<texture2d<uint, access::read>, kSRSRMaxFrames> q;
                metal::array<texture2d<float>, kSRSRMaxFrames> c, rj;
                metal::array<texture2d<float, access::read>, kSRSRMaxFrames> fl;
                for (int i = 0; i < kSRSRMaxFrames; ++i) {
                    q[i] = RU(8 + i); c[i] = S(24 + i); fl[i] = R(40 + i); rj[i] = S(56 + i);
                }
                const auto& u = U<SRSRMergeUniforms>(0);
                const SRSRFrameNormalization* norms = (const SRSRFrameNormalization*)bytes_[1].data();
                const SRSRFrameGeometry* geo = (const SRSRFrameGeometry*)bytes_[2].data();
                grid(w, h, [&](uint2 p) { srsr_merge(o, acc, q, c, fl, rj, u, norms, geo, p); });
                break;
            }
            case K::denoise: {
                auto in = R(0); auto acc = S(1); auto o = Wt(2);
                ushort* out = (ushort*)buf_[1]->contents();
                const auto& u = U<SRSRDenoiseUniforms>(0);
                grid(w, h, [&](uint2 p) { srsr_denoise(in, acc, o, out, u, p); });
                break;
            }
            case K::count: break;
        }
    }

private:
    CTex* tex_[80] = {};
    Buf* buf_[8] = {};
    std::vector<uint8_t> bytes_[8];
    uint64_t ticket_ = 0;
    bool failed_ = false;

    template <class T> const T& U(int slot) const { return *(const T*)bytes_[slot].data(); }
    texture2d<float, access::read> R(int s) const { return texture2d<float, access::read>(&tex_[s]->t); }
    texture2d<float, access::write> Wt(int s) const { return texture2d<float, access::write>(&tex_[s]->t); }
    texture2d<float> S(int s) const { return texture2d<float>(&tex_[s]->t); }
    texture2d<uint, access::read> RU(int s) const { return texture2d<uint, access::read>(&tex_[s]->t); }
    texture2d<uint, access::write> WU(int s) const { return texture2d<uint, access::write>(&tex_[s]->t); }

    template <class F> static void grid(int w, int h, const F& f) {
#pragma omp parallel for schedule(dynamic, 4)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) f(uint2((uint)x, (uint)y));
    }
};

} // namespace

std::unique_ptr<Backend> make_backend() { return std::make_unique<CpuBackend>(); }

} // namespace srsr
} // namespace hhsr
