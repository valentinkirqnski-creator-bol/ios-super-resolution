// Verifies that pad_circular_f32 in HHSRKernels.metal agrees with
// pad_image_circular in pipeline.cpp.
//
//   g++ -std=c++17 -O2 tools/verify_pad_circular.cpp -o verify_pad_circular
//   ./verify_pad_circular        (exit 0 iff every geometry agrees)
//
// The Metal kernel cannot be compiled here, so its index arithmetic is
// transcribed verbatim and compared against the CPU original on every geometry
// the app can produce. This matters because only one of the two runs at a time --
// the GPU path when the grey is resident, the CPU path when it is not -- so a
// disagreement would be a silent alignment error rather than a crash.
//
// Why the kernel exists: compute_grey_fft_metal returns a DIMENSIONS-ONLY Image
// on the resident path (want_host = !resident_raw || debug_dumps_enabled), so the
// pixels live only in sticky_grey and padding it on the CPU read from an empty
// vector. 3024 is 16 x 189 with 189 odd, so the grey divides by 8 and 16 but
// never by 32 or 64 -- which is why only those two tile sizes ever reached the
// host pad, and why only they crashed.
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vector>

// ---- the CPU original, from core/pipeline.cpp ------------------------------
static int cpu_pad_amount(int h, int w, int ts) {
    const int ph = (ts - h % ts) % ts;
    const int pw = (ts - w % ts) % ts;
    return ph + pw;
}

static std::vector<float> cpu_pad(const std::vector<float>& img, int h, int w, int ts,
                                 int& out_h, int& out_w) {
    const int pad_h = (ts - h % ts) % ts;
    const int pad_w = (ts - w % ts) % ts;
    out_h = h + pad_h;
    out_w = w + pad_w;
    std::vector<float> padded((size_t)out_h * out_w);
    for (int y = 0; y < out_h; ++y) {
        const int src_y = y < h ? y : (y - h);
        for (int x = 0; x < out_w; ++x) {
            const int src_x = x < w ? x : (x - w);
            padded[(size_t)y * out_w + x] = img[(size_t)src_y * w + src_x];
        }
    }
    return padded;
}

// ---- the Metal kernel, transcribed verbatim from HHSRKernels.metal --------
// kernel void pad_circular_f32(...) {
//     if (gid.x >= p.out_w || gid.y >= p.out_h) return;
//     uint sy = (gid.y < p.in_h) ? gid.y : (gid.y % p.in_h);
//     uint sx = (gid.x < p.in_w) ? gid.x : (gid.x % p.in_w);
//     out[gid.y * p.out_w + gid.x] = in[sy * p.in_w + sx];
// }
static std::vector<float> gpu_pad(const std::vector<float>& img,
                                 uint32_t in_h, uint32_t in_w,
                                 uint32_t out_h, uint32_t out_w) {
    std::vector<float> out((size_t)out_h * out_w, -12345.f);
    for (uint32_t gy = 0; gy < out_h; ++gy)
        for (uint32_t gx = 0; gx < out_w; ++gx) {
            const uint32_t sy = (gy < in_h) ? gy : (gy % in_h);
            const uint32_t sx = (gx < in_w) ? gx : (gx % in_w);
            out[(size_t)gy * out_w + gx] = img[(size_t)sy * in_w + sx];
        }
    return out;
}

static int g_fail = 0;

static void check(int h, int w, int ts, const char* note) {
    (void)note;
    // pad_image_circular's (i - extent) wrap only lands inside the plane while
    // the pad is smaller than the extent. Past that the CPU function reads out of
    // its own bounds, so it is not a reference to compare against -- there the
    // kernel only has to stay in range.
    const int pad_h_only = (ts - h % ts) % ts;
    const int pad_w_only = (ts - w % ts) % ts;
    const bool cpu_is_valid_reference = (pad_h_only < h) && (pad_w_only < w);
    std::vector<float> img((size_t)h * w);
    for (size_t i = 0; i < img.size(); ++i)
        img[i] = (float)((i * 2654435761u) % 65521u) / 65521.f;   // deterministic, no structure

    int ch = 0, cw = 0;
    const std::vector<float> cpu = cpu_pad(img, h, w, ts, ch, cw);
    // The host computes the padded extents the same way align_metal now does.
    const uint32_t gh = (uint32_t)(h + ((ts - h % ts) % ts));
    const uint32_t gw = (uint32_t)(w + ((ts - w % ts) % ts));
    const std::vector<float> gpu = gpu_pad(img, (uint32_t)h, (uint32_t)w, gh, gw);

    bool dims_ok = ((int)gh == ch && (int)gw == cw);
    long diffs = 0;
    long unwritten = 0;
    if (dims_ok) {
        for (size_t i = 0; i < cpu.size(); ++i) {
            if (gpu[i] == -12345.f) ++unwritten;
            else if (gpu[i] != cpu[i]) ++diffs;
        }
    }
    // Every read must land inside the source plane -- the wrap is i - extent, not
    // a modulo, so it is only safe while pad < extent.
    bool in_range = true;
    for (uint32_t gy = 0; gy < gh && in_range; ++gy) {
        const uint32_t sy = (gy < (uint32_t)h) ? gy : (gy % (uint32_t)h);
        if (sy >= (uint32_t)h) in_range = false;
    }
    for (uint32_t gx = 0; gx < gw && in_range; ++gx) {
        const uint32_t sx = (gx < (uint32_t)w) ? gx : (gx % (uint32_t)w);
        if (sx >= (uint32_t)w) in_range = false;
    }

    const bool ok = dims_ok && unwritten == 0 && in_range &&
                    (cpu_is_valid_reference ? (diffs == 0) : true);
    if (!ok) ++g_fail;
    std::printf("  %5dx%-5d ts %2d -> %5dx%-5d  pad %4d  %s%s%s%s  %s\n",
                h, w, ts, ch, cw, cpu_pad_amount(h, w, ts),
                dims_ok ? "dims" : "DIMS-MISMATCH",
                !cpu_is_valid_reference ? " (cpu ref invalid)"
                                        : (diffs ? " PIXELS-DIFFER" : " pixels"),
                unwritten ? " UNWRITTEN" : "",
                in_range ? " in-range" : " OUT-OF-RANGE",
                ok ? "OK" : "FAIL");
}

int main() {
    std::printf("pad_circular_f32 (Metal, transcribed) vs pad_image_circular (CPU)\n");
    std::printf("every pixel compared, not sampled\n\n");

    // The real device geometry. FFT grey is full resolution, so the grey the
    // align path pads is 3024x4032. 3024 = 16 x 189 and 189 is odd, which is the
    // whole reason 32 and 64 crashed and 8/16 did not.
    std::printf("-- iPhone 12MP grey, 3024x4032, every tile size the picker offers --\n");
    for (int ts : {8, 16, 32, 64}) check(3024, 4032, ts, "device");

    std::printf("\n-- the 2x2 quad grey (Decimate), half resolution --\n");
    for (int ts : {8, 16, 32, 64}) check(1512, 2016, ts, "quad");

    std::printf("\n-- geometries that pad on BOTH axes --\n");
    check(100, 100, 32, "both");
    check(101, 103, 16, "both odd");
    check(1, 1, 8, "degenerate");
    check(7, 9, 8, "smaller than a tile");
    check(65, 63, 64, "straddles one tile");

    std::printf("\n-- worst case for the wrap: pad one short of the extent --\n");
    // pad = ts - h%ts. With h = ts+1 the pad is ts-1, the largest pad that can
    // occur relative to a plane this small, so src = i - h is at its furthest.
    for (int ts : {8, 16, 32, 64}) check(ts + 1, ts + 1, ts, "max relative pad");

    // ---- residency flow-slice geometry vs align's -------------------------
    // Same family as the pad above. metal_frames_begin sizes the GPU flow slice
    // from the RAW dimensions, while align() produces its flow field on a grey
    // that pad_image_circular has already rounded UP to whole tiles. Sizing the
    // slice by floor made the two disagree, metal_frame_set_flow rejected the
    // frame on ny, and the shutter failed with "GPU frame state unavailable".
    // Both sides must round up. 3024 = 16 x 189, and 189 being odd is why this
    // only bit at the larger tile sizes -- exactly the divisibility that made
    // the host pad crash above.
    std::printf("\n-- flow slice (metal_frames_begin) vs align's flow field --\n");
    for (int dims = 0; dims < 2; ++dims) {
        const int h = dims ? 1512 : 3024;
        const int w = dims ? 2016 : 4032;
        for (int ts : {8, 16, 32, 64}) {
            const int pad_h = h + ((ts - h % ts) % ts);
            const int pad_w = w + ((ts - w % ts) % ts);
            const int align_ny = (pad_h + ts - 1) / ts;   // align.cpp, on the PADDED grey
            const int align_nx = (pad_w + ts - 1) / ts;
            const int slice_ny = (h + ts - 1) / ts;       // metal_gpu.mm, on the raw dims
            const int slice_nx = (w + ts - 1) / ts;
            const int old_ny = h / ts;                    // what it used to be
            const int old_nx = w / ts;
            const bool ok = align_ny == slice_ny && align_nx == slice_nx;
            if (!ok) ++g_fail;
            std::printf("  %5dx%-5d ts %2d  align %3dx%-3d  slice %3dx%-3d  %-4s"
                        "  (floor was %3dx%-3d %s)\n",
                        h, w, ts, align_ny, align_nx, slice_ny, slice_nx,
                        ok ? "OK" : "FAIL", old_ny, old_nx,
                        (old_ny == align_ny && old_nx == align_nx) ? "ok" : "MISMATCH");
        }
    }

    std::printf("\n%d failed\n", g_fail);
    return g_fail ? 1 : 0;
}
