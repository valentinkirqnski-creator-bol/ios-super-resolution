// srsr_dump <burst_dir> <out_dir> [scale]
//
// Runs one synthetic burst through the SRSR pipeline (PC CPU backend: the same
// SRSRKernels.metal the app runs) and dumps every input of the merge, plus the
// merged image, so tools/srsr_nn can rebuild the merge exactly in PyTorch.
//
// burst_dir (written by gen.py):
//   meta.txt     H W N cfa00 cfa01 cfa10 cfa11 a_r a_g a_b b_r b_g b_b wb_r wb_g wb_b prewhitened
//                then one line of N exposures (shutter x ISO, any unit; ratios matter)
//   frame_XX.f32 H x W float32, the loader's raw: black-subtracted, white = 1, pre-white-balanced
//                when prewhitened = 1 (as the app's loader delivers an iPhone DNG)
// out_dir:
//   index.txt    one line per file: name w h nc dtype
//   uniforms.bin SRSRMergeUniforms, norms.bin SRSRFrameNormalization[n], geometry.bin SRSRFrameGeometry[n]
//   quad_SS.u16 (RGBA16Uint codes), cov_SS.f32 (RGBA, half-quantised), flow_SS.f32 (RGBA32Float),
//   mask_SS.f32 (dilated keep, R8-quantised; slots >= 1), accw.f32, merged.f32 (outH x outW x 3,
//   before the denoiser). Slot 0 is the reference; slots 1.. the comparison frames in burst order.
#include "pipeline.h"
#include "srsr_pipeline.h"
#include "srsr_cpu_hook.h"
#include "srsr_types.h"
#include "types.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace hhsr;
namespace fs = std::filesystem;

static std::string g_out;
static std::vector<float> g_merged;     // outH x outW x 3, assembled from the bands
static int g_outW = 0, g_outH = 0;
static bool g_inputs_written = false;
static std::ofstream g_index;

static void write_file(const std::string& name, const void* p, size_t n) {
    FILE* f = std::fopen((g_out + "/" + name).c_str(), "wb");
    if (!f) { std::fprintf(stderr, "cannot write %s\n", name.c_str()); std::exit(3); }
    std::fwrite(p, 1, n, f);
    std::fclose(f);
}

static void dump_slot(const std::string& name, const srsr::CpuSlotView& s) {
    if (!s.bound) return;
    const size_t n = (size_t)s.w * s.h * s.nc;
    if (s.u) {
        write_file(name + ".u16", s.u, n * 2);
        g_index << name << ".u16 " << s.w << " " << s.h << " " << s.nc << " u16\n";
    } else {
        write_file(name + ".f32", s.f, n * 4);
        g_index << name << ".f32 " << s.w << " " << s.h << " " << s.nc << " f32\n";
    }
}

static void hook(const srsr::CpuDispatchView& v) {
    if (v.kernel != srsr::K::merge) return;
    SRSRMergeUniforms u;
    std::memcpy(&u, v.bytes[0].data(), sizeof u);
    if (!g_inputs_written) {
        g_inputs_written = true;
        write_file("uniforms.bin", v.bytes[0].data(), v.bytes[0].size());
        write_file("norms.bin", v.bytes[1].data(), v.bytes[1].size());
        write_file("geometry.bin", v.bytes[2].data(), v.bytes[2].size());
        char nm[32];
        for (int f = 0; f < u.frameCount; ++f) {
            std::snprintf(nm, sizeof nm, "quad_%02d", f);  dump_slot(nm, v.tex(8 + f));
            std::snprintf(nm, sizeof nm, "cov_%02d", f);   dump_slot(nm, v.tex(24 + f));
            if (f == 0) continue;
            std::snprintf(nm, sizeof nm, "flow_%02d", f);  dump_slot(nm, v.tex(40 + f));
            std::snprintf(nm, sizeof nm, "mask_%02d", f);  dump_slot(nm, v.tex(56 + f));
        }
        dump_slot("accw", v.tex(1));
        g_outW = u.outWidth; g_outH = u.outHeight;
        g_merged.assign((size_t)g_outW * g_outH * 3, 0.f);
    }
    // The band texture: rows 0..bandRows-1 hold output rows bandY0.. (RGBA16Float).
    const srsr::CpuSlotView b = v.tex(0);
    for (int r = 0; r < u.bandRows; ++r) {
        const int y = u.bandY0 + r;
        if (y < 0 || y >= g_outH || r >= b.h) continue;
        for (int x = 0; x < g_outW && x < b.w; ++x)
            for (int c = 0; c < 3; ++c)
                g_merged[((size_t)y * g_outW + x) * 3 + c] = b.f[((size_t)r * b.w + x) * 4 + c];
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: srsr_dump <burst_dir> <out_dir> [scale]\n");
        return 1;
    }
    const std::string in = argv[1];
    g_out = argv[2];
    fs::create_directories(g_out);
    const float scale = argc > 3 ? (float)std::atof(argv[3]) : 1.f;

    std::ifstream mf(in + "/meta.txt");
    int H, W, N, c00, c01, c10, c11, prew;
    float a[3], b[3], wb[3];
    mf >> H >> W >> N >> c00 >> c01 >> c10 >> c11 >> a[0] >> a[1] >> a[2] >> b[0] >> b[1] >> b[2] >> wb[0] >> wb[1] >> wb[2] >> prew;
    std::vector<float> expo(N, 1.f);
    for (int i = 0; i < N; ++i) mf >> expo[i];
    if (!mf) { std::fprintf(stderr, "bad meta.txt\n"); return 2; }

    Config cfg;
    cfg.srsr_enabled = true;
    cfg.srsr_denoise = false;          // the dump is the merge; the denoiser is not part of training (yet)
    cfg.srsr_unblocker = false;
    cfg.robustness_save_mask = false;
    cfg.scale = scale;
    // SRSR_OV=0/1: override Config::srsr_align_overlap16 (A/B comparisons of the aligner).
    if (const char* ov = std::getenv("SRSR_OV")) cfg.srsr_align_overlap16 = std::atoi(ov) != 0;

    auto loader = [&](int k, Config& c, bool, int, int) -> Image {
        c.cfa.p[0][0] = (uint8_t)c00; c.cfa.p[0][1] = (uint8_t)c01;
        c.cfa.p[1][0] = (uint8_t)c10; c.cfa.p[1][1] = (uint8_t)c11;
        for (int i = 0; i < 3; ++i) { c.alpha_dng[i] = a[i]; c.beta_dng[i] = b[i]; c.white_balance[i] = wb[i]; }
        c.raw_prewhitened = prew != 0;
        Image im(H, W, 1);
        char nm[32];
        std::snprintf(nm, sizeof nm, "/frame_%02d.f32", k);
        FILE* f = std::fopen((in + nm).c_str(), "rb");
        if (!f || std::fread(im.data.data(), 4, im.data.size(), f) != im.data.size()) {
            if (f) std::fclose(f);
            return Image();
        }
        std::fclose(f);
        im.exposure = expo[k];
        return im;
    };
    // The loader fills the Config fields the pipeline reads at the start, too.
    { Config c = cfg; loader(0, c, true, 0, 0); cfg = c; }

    g_index.open(g_out + "/index.txt");
    srsr::g_cpu_dispatch_hook = hook;
    std::string why;
    Image prev = process_burst_srsr_to_dng(N, loader, cfg, g_out + "/out.dng", nullptr, 64, &why);
    srsr::g_cpu_dispatch_hook = nullptr;
    if (prev.h <= 0) { std::fprintf(stderr, "SRSR failed: %s\n", why.c_str()); return 4; }
    write_file("merged.f32", g_merged.data(), g_merged.size() * 4);
    g_index << "merged.f32 " << g_outW << " " << g_outH << " 3 f32\n";
    g_index.close();
    fs::remove(g_out + "/out.dng");
    std::printf("ok %d x %d, %d frames\n", g_outW, g_outH, N);
    return 0;
}
