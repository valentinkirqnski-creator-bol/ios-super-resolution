// wronski_dump: synthetic bursts through the app's own Wronski stages (core defaults).
//
//   wronski_dump <jobs.txt>        each line: <in.bin> <out.bin>
//
// The stages are exactly process_burst's (core/pipeline.cpp): grey + pyramid, align,
// flow_to_raw_tile_grid, init_robustness / compute_robustness (the analytic mask),
// estimate_kernels, merge_comp / merge_ref. Everything the fusion network needs that is not
// the network itself is written out: per comparison frame the tile flow and the analytic R,
// per frame the kernel covariances, and the merged image (for the Python merge's parity check).
//
// in.bin  : int32 N, H, W, cfa[4] ; f32 alpha, beta, wb[3], scale ; f32 frames[N][H][W]
// out.bin : int32 magic 0x57524431, N, tile_size, flags (1 dense, 2 affine, 4 homography)
//           per comparison frame k = 1..N-1: int32 ny, nx ; f32 flow[ny][nx][2]
//                                            int32 rh, rw, rc ; f32 R[rh][rw][rc]
//           per frame k = 0..N-1:            int32 ch, cw ; f32 cov[ch][cw][4]
//           int32 Hs, Ws ; f32 merged[Hs][Ws][3]
#include "pipeline.h"
#include "snr_tuning.h"
#include "stages.h"
#include "types.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace hhsr;

template <class T> static void put(FILE* f, const T* p, size_t n) { std::fwrite(p, sizeof(T), n, f); }
static void put_i(FILE* f, int v) { put(f, &v, 1); }

static bool run(const std::string& in, const std::string& out) {
    FILE* fi = std::fopen(in.c_str(), "rb");
    if (!fi) { std::fprintf(stderr, "cannot open %s\n", in.c_str()); return false; }
    int hdr[7];
    float par[6];
    if (std::fread(hdr, 4, 7, fi) != 7 || std::fread(par, 4, 6, fi) != 6) { std::fclose(fi); return false; }
    const int N = hdr[0], H = hdr[1], W = hdr[2];
    std::vector<Image> burst(N);
    for (int k = 0; k < N; ++k) {
        burst[k] = Image(H, W, 1);
        if (std::fread(burst[k].data.data(), 4, (size_t)H * W, fi) != (size_t)H * W) { std::fclose(fi); return false; }
    }
    std::fclose(fi);

    Config cfg;
    cfg.num_threads = 1;                       // the generator runs one process per core
    if (const char* t = std::getenv("WD_THREADS")) cfg.num_threads = std::atoi(t);   // full-size real bursts
    cfg.scale = par[5];
    cfg.cfa.p[0][0] = (uint8_t)hdr[3]; cfg.cfa.p[0][1] = (uint8_t)hdr[4];
    cfg.cfa.p[1][0] = (uint8_t)hdr[5]; cfg.cfa.p[1][1] = (uint8_t)hdr[6];
    for (int i = 0; i < 3; ++i) { cfg.alpha_dng[i] = par[0]; cfg.beta_dng[i] = par[1]; cfg.white_balance[i] = par[2 + i]; }
    cfg.raw_prewhitened = true;

    // ---- process_burst (core/pipeline.cpp), stage by stage
    Config work = cfg;
    work.burst_frame_count = N;
    tune_config_snr(burst[0], work);
    const Image& ref = burst[0];
    const int tile_size = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];
    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    Image ref_grey_padded = pad_image_circular(ref_grey, work.grey_tile_size(tile_size));
    Pyramid ref_pyr = build_pyramid(ref_grey_padded, work.bm_factors);
    RefStats ref_stats = init_robustness(ref, work);
    CovField ref_covs = estimate_kernels(ref, work);
    const int Hs = (int)std::lround(work.scale * H), Ws = (int)std::lround(work.scale * W);
    Image num(Hs, Ws, 3), den(Hs, Ws, 3);

    FILE* fo = std::fopen(out.c_str(), "wb");
    if (!fo) { std::fprintf(stderr, "cannot write %s\n", out.c_str()); return false; }
    std::vector<CovField> covs(N);
    covs[0] = ref_covs;
    int flags = 0;
    std::vector<FlowField> flows(N);
    std::vector<Image> robs(N);
    for (int k = 1; k < N; ++k) {
        const Image& comp = burst[k];
        Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);
        FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, tile_size);
        flow = flow_to_raw_tile_grid(flow, comp.h, comp.w, comp_grey.h, comp_grey.w, tile_size,
                                     work.r_Mt, work.num_threads, work.grey_tile_size(tile_size));
        if (flow.has_dense()) flags |= 1;
        if (flow.has_affine()) flags |= 2;
        if (flow.has_global_h) flags |= 4;
        robs[k] = compute_robustness(comp, ref_stats, flow, tile_size, work);
        covs[k] = estimate_kernels(comp, work);
        merge_comp(comp, flow, covs[k], robs[k], tile_size, num, den, work);
        flows[k] = std::move(flow);
    }
    merge_ref(ref, ref_covs, num, den, work, nullptr);

    put_i(fo, 0x57524431); put_i(fo, N); put_i(fo, tile_size); put_i(fo, flags);
    for (int k = 1; k < N; ++k) {
        put_i(fo, flows[k].ny); put_i(fo, flows[k].nx);
        put(fo, flows[k].flow.data(), (size_t)flows[k].ny * flows[k].nx * 2);
        put_i(fo, robs[k].h); put_i(fo, robs[k].w); put_i(fo, robs[k].c);
        put(fo, robs[k].data.data(), robs[k].data.size());
    }
    for (int k = 0; k < N; ++k) {
        put_i(fo, covs[k].h); put_i(fo, covs[k].w);
        put(fo, covs[k].cov.data(), covs[k].cov.size());
    }
    put_i(fo, Hs); put_i(fo, Ws);
    std::vector<f32> m((size_t)Hs * Ws * 3);
    for (size_t i = 0; i < m.size(); ++i) m[i] = den.data[i] > 0.f ? num.data[i] / den.data[i] : 0.f;
    put(fo, m.data(), m.size());
    std::fclose(fo);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: wronski_dump <jobs.txt>\n"); return 1; }
    std::ifstream jobs(argv[1]);
    std::string line;
    int bad = 0;
    while (std::getline(jobs, line)) {
        std::istringstream ss(line);
        std::string in, out;
        if (!(ss >> in >> out)) continue;
        if (!run(in, out)) ++bad;
    }
    return bad ? 2 : 0;
}
