// Run the REAL block matcher over a synthesised burst and emit its flow field.
//
// The training set previously corrupted the true flow with a hand-chosen noise
// model. That teaches the mask to recognise errors nobody's aligner makes. The
// errors that matter are the ones THIS aligner produces -- its failure modes on
// repeated texture, on low contrast, at tile boundaries, under rotation -- and
// the only way to get that distribution is to run it.
//
// So: Python synthesises the burst and writes the raw planes here; this runs
// compute_grey -> pad -> build_pyramid -> align -> flow_to_raw_tile_grid,
// exactly as core/pipeline_paths.cpp does, and writes the per-tile flow back.
// Whatever the matcher gets wrong is then what the network trains against.
//
//   in:  i32 h, i32 w, i32 tile_size, i32 n_frames, i32 grey_method,
//        then n_frames * h*w f32   (frame 0 is the reference)
//   out: i32 n_comp, i32 ny, i32 nx,
//        then n_comp * ny*nx*2 f32 (dx, dy per tile, RAW pixels)
//
// Whole burst per invocation: process startup dominates otherwise.
#include "stages.h"
#include "types.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace hhsr;

static bool rd(std::FILE* f, void* p, size_t n) {
    return std::fread(p, 1, n, f) == n;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: align_tool in.bin out.bin\n");
        return 2;
    }
    std::FILE* fi = std::fopen(argv[1], "rb");
    if (!fi) { std::printf("cannot open %s\n", argv[1]); return 1; }
    int32_t h = 0, w = 0, ts = 0, nf = 0, gm = 0;
    if (!rd(fi, &h, 4) || !rd(fi, &w, 4) || !rd(fi, &ts, 4) ||
        !rd(fi, &nf, 4) || !rd(fi, &gm, 4)) {
        std::printf("short header\n"); return 1;
    }
    if (h <= 0 || w <= 0 || ts <= 0 || nf < 2) {
        std::printf("bad header %d %d %d %d\n", h, w, ts, nf); return 1;
    }
    std::vector<Image> raws((size_t)nf);
    for (int i = 0; i < nf; ++i) {
        raws[(size_t)i] = Image(h, w, 1);
        if (!rd(fi, raws[(size_t)i].data.data(),
                (size_t)h * (size_t)w * sizeof(f32))) {
            std::printf("short frame %d\n", i); return 1;
        }
    }
    std::fclose(fi);

    Config cfg;
    cfg.bayer_mode = true;
    cfg.grey_method = (gm == 1) ? GreyMethod::Decimate : GreyMethod::FFT;
    cfg.bm_tile_sizes = {ts, ts, ts, std::max(8, ts / 2)};
    cfg.num_threads = 0;

    Image ref_grey = compute_grey(raws[0], cfg.bayer_mode, cfg.grey_method);
    Image ref_padded = pad_image_circular(ref_grey, cfg.grey_tile_size(ts));
    Pyramid ref_pyr = build_pyramid(ref_padded, cfg.bm_factors);

    std::vector<FlowField> flows;
    flows.reserve((size_t)nf - 1);
    int ny = 0, nx = 0;
    for (int i = 1; i < nf; ++i) {
        Image cg = compute_grey(raws[(size_t)i], cfg.bayer_mode,
                                cfg.grey_method);
        FlowField fl = align(ref_pyr, ref_grey, cg, cfg, ts);
        // The grey lattice may be coarser than raw; the mask and merge index a
        // raw-pixel tile grid, so convert exactly as the pipeline does.
        fl = flow_to_raw_tile_grid(fl, h, w, ref_grey.h, ref_grey.w, ts,
                                   cfg.r_Mt, cfg.num_threads,
                                   cfg.grey_tile_size(ts));
        if (i == 1) { ny = fl.ny; nx = fl.nx; }
        if (fl.ny != ny || fl.nx != nx) {
            std::printf("flow grid changed between frames\n"); return 1;
        }
        flows.push_back(std::move(fl));
    }
    clear_align_ref_ica_cache();

    std::FILE* fo = std::fopen(argv[2], "wb");
    if (!fo) { std::printf("cannot write %s\n", argv[2]); return 1; }
    int32_t nc = (int32_t)flows.size();
    std::fwrite(&nc, 4, 1, fo);
    std::fwrite(&ny, 4, 1, fo);
    std::fwrite(&nx, 4, 1, fo);
    std::vector<f32> buf((size_t)ny * (size_t)nx * 2);
    for (const FlowField& fl : flows) {
        for (int y = 0; y < ny; ++y) {
            for (int x = 0; x < nx; ++x) {
                buf[((size_t)y * (size_t)nx + (size_t)x) * 2 + 0] = fl.dx(y, x);
                buf[((size_t)y * (size_t)nx + (size_t)x) * 2 + 1] = fl.dy(y, x);
            }
        }
        std::fwrite(buf.data(), sizeof(f32), buf.size(), fo);
    }
    std::fclose(fo);
    return 0;
}
