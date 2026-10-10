// See shift_consensus_pipeline.h.
#include "shift_consensus_pipeline.h"
#include "shift_consensus.h"
#include "stages.h"
#include "global_homography.h"
#include "isa_prealign.h"
#if defined(__APPLE__)
#include "metal_gpu.h"
#endif
#include <array>
#include <cmath>
#include <utility>

namespace hhsr {

std::vector<FlowField> compute_consensus_flows(
    const Image& ref_grey, int ref_index, int frame_count,
    const std::function<Image(int)>& grey_of_frame,
    const Config& cfg, int tile_size) {

    std::vector<FlowField> empty;
    if (!cfg.shift_consensus_enabled) return empty;
    if (frame_count < 3) return empty;               // needs >=3 for any redundancy
    if (ref_index < 0 || ref_index >= frame_count) return empty;
    if (ref_grey.h <= 0 || ref_grey.w <= 0) return empty;
    // This pre-pass holds every frame's warped grey + pyramid simultaneously
    // (~160MB/frame at 12MP). Above the cap it would exhaust memory and the OS
    // kills the app, so bail to the normal per-frame align instead.
    const int max_frames = (cfg.shift_consensus_max_frames > 0) ? cfg.shift_consensus_max_frames : 6;
    if (frame_count > max_frames) return empty;

    // Block-match the ad-hoc warped greys on the CPU path: they were never
    // uploaded as resident GPU frames, so align_metal (the default on iOS) is
    // driven out of its model and can crash. Slower, but it cannot crash.
    Config ccfg = cfg;
    ccfg.align_force_cpu = true;

    // Every grey is sampled on the CPU (pre-align + block match), so it MUST be
    // fully host-resident. On the Metal path a grey can carry dims with empty
    // host data (GPU-only); feeding that to resize_blur reads a null buffer and
    // crashes. Bail to the normal per-frame align instead of dereferencing it.
    auto host_ready = [](const Image& im) {
        return im.h > 0 && im.w > 0 && im.c > 0 && !im.data.empty() &&
               im.data.size() == (size_t)im.h * (size_t)im.w * (size_t)im.c;
    };
    if (!host_ready(ref_grey)) return empty;

    const int H = ref_grey.h, W = ref_grey.w;
    const int gts = cfg.grey_tile_size(tile_size);

    // 1) Global pre-align every frame into the reference's coordinate frame and
    //    keep the warped grey + its pyramid. Same rigid-seed-then-homography
    //    refine the per-frame isa_warp path uses, so the residual tracks below
    //    live in the same domain (comp_pos = H * (lr + residual_flow)).
    std::vector<Image> wgrey(frame_count);
    std::vector<Image> wpad(frame_count);
    std::vector<Pyramid> wpyr(frame_count);
    std::vector<std::array<f32, 9>> Hframe(frame_count);
    for (int f = 0; f < frame_count; ++f) {
        Hframe[f] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
        Image g = grey_of_frame(f);
        // Precondition: full-res, matching dims, and fully host-resident (see above).
        if (g.h != H || g.w != W || !host_ready(g)) return empty;
        if (f == ref_index) {
            wgrey[f] = g;
        } else {
            f32 idx = 0.f, idy = 0.f, irot = 0.f;
            if (estimate_isa_prealign(ref_grey, g, cfg, idx, idy, irot)) {
                const f32 cx = 0.5f * (f32)(W - 1), cy = 0.5f * (f32)(H - 1);
                const f32 cs = std::cos(irot), sn = std::sin(irot);
                f32 Hm[9] = { cs, -sn, cx + idx - cs * cx + sn * cy,
                              sn,  cs, cy + idy - sn * cx - cs * cy, 0.f, 0.f, 1.f };
                refine_global_homography_seed(ref_grey, g, cfg, Hm);
                for (int i = 0; i < 9; ++i) Hframe[f][i] = Hm[i];
                wgrey[f] = warp_grey_by_homography(g, Hm);
            } else {
                wgrey[f] = g;                          // no seed: track from identity
            }
        }
        wpad[f] = pad_image_circular(wgrey[f], gts);
        wpyr[f] = build_pyramid(wpad[f], cfg.bm_factors);
    }

    // 2) Measure the redundant pairwise residual tracks. Full strategy unless a
    //    Blocks window is requested.
    TrackingStrategy strat = TrackingStrategy::Full;
    int block = frame_count - 1;
    if (cfg.shift_consensus_block_size > 0 && cfg.shift_consensus_block_size < frame_count - 1) {
        strat = TrackingStrategy::Blocks;
        block = cfg.shift_consensus_block_size;
    }
    ShiftConsensus sc(frame_count, ref_index, strat, block);
    sc.set_reject_threshold_px2(cfg.shift_consensus_reject_px2);
    const auto& pairs = sc.pairs();
    const int m = sc.shift_count();

    std::vector<ShiftGrid> meas(m);
    int gny = 0, gnx = 0;
    for (int k = 0; k < m; ++k) {
        const int i = pairs[k].reference, j = pairs[k].toTrack;
#if defined(__APPLE__)
        // align()'s Metal path pins the moving grey by dimensions; every pair
        // shares dims, so drop the pin between aligns or frame j's grey gets
        // reused for j'. Also clear the ref ICA cache when the ref pyramid
        // changes (it is keyed by pyramid address).
        metal_invalidate_sticky_grey();
#endif
        FlowField fl = align(wpyr[i], wpad[i], wgrey[j], ccfg, tile_size, 0.f, 0.f, 0.f);
        if (k == 0) { gny = fl.ny; gnx = fl.nx; }
        if (fl.ny != gny || fl.nx != gnx || fl.flow.empty()) return empty;
        meas[k].ny = fl.ny; meas[k].nx = fl.nx; meas[k].v = std::move(fl.flow);
    }
#if defined(__APPLE__)
    clear_align_ref_ica_cache();
#endif
    if (gny <= 0 || gnx <= 0) return empty;

    // 3) Solve the consensus: outlier-free residual frame->reference flow.
    std::vector<ShiftGrid> cons;
    if (!sc.minimize(meas, cons)) return empty;

    // 4) Package each comparison frame's cleaned flow on the raw tile grid with
    //    its global homography, exactly like the per-frame isa_warp path.
    std::vector<FlowField> out(frame_count);
    for (int f = 0; f < frame_count; ++f) {
        if (f == ref_index) continue;
        FlowField flow(gny, gnx);
        flow.flow = cons[f].v;
        flow = flow_to_raw_tile_grid(flow, H, W, ref_grey.h, ref_grey.w,
                                     tile_size, cfg.r_Mt, cfg.num_threads,
                                     cfg.grey_tile_size(tile_size));
        for (int i = 0; i < 9; ++i) flow.global_h[i] = Hframe[f][i];
        flow.has_global_h = true;
        out[f] = std::move(flow);
    }
    return out;
}

} // namespace hhsr
