// Burst processing for memory-constrained mobile targets (iOS).
// Full-res (1×) and large bursts: spill comparison Bayer to disk after analyze.
// Lighter 2× crops keep RAW+analysis in RAM. Apple: GPU merge prefetch after analyze.
#include "pipeline.h"
#include "stages.h"
#include "dng_writer.h"
#include "snr_tuning.h"
#include "raw_io.h"
#include "parallel.h"
#include "debug_utils.h"
#include "prof.h"
#if defined(__APPLE__)
#include "metal_gpu.h"
#endif
#include <vector>
#include <array>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cstdio>
#include <cmath>
#include <future>
#include <memory>
#include <limits>
#include <sstream>
#if defined(__APPLE__)
#include <thread>
#endif

namespace fs = std::filesystem;

namespace hhsr {

namespace {

static bool save_image(const fs::path& p, const Image& im) {
    int32_t hdr[3] = {im.h, im.w, im.c};
    std::ofstream out(p, std::ios::binary);
    if (!out) return false;
    out.write((const char*)hdr, sizeof(hdr));
    out.write((const char*)im.data.data(), (std::streamsize)(im.data.size() * sizeof(f32)));
    return out.good();
}

static bool load_image(const fs::path& p, Image& im) {
    int32_t hdr[3];
    std::ifstream in(p, std::ios::binary);
    if (!in || !in.read((char*)hdr, sizeof(hdr))) return false;
    // Reuse storage when shape matches — avoids alloc+zero on every reload.
    if (im.h != hdr[0] || im.w != hdr[1] || im.c != hdr[2])
        im = Image(hdr[0], hdr[1], hdr[2]);
    return (bool)in.read((char*)im.data.data(), (std::streamsize)(im.data.size() * sizeof(f32)));
}

struct CachedCompFrame {
    FlowField flow;
    Image rob;
    CovField covs;
    Image comp;
    std::vector<uint8_t> rob_rows_nonzero;
    bool rob_has_nonzero = true;
    int index = 0;
};

struct CachedCompMeta {
    FlowField flow;
    Image rob;
    CovField covs;
    std::vector<uint8_t> rob_rows_nonzero;
    bool rob_has_nonzero = true;
    int index = 0;
};

static bool load_cached_comp_raw(const fs::path& cache, int k, Image& comp) {
    return load_image(cache / ("f" + std::to_string(k) + ".raw"), comp) && comp.h > 0;
}

static bool robustness_row_activity(const Image& rob, std::vector<uint8_t>& rows) {
    rows.assign((size_t)std::max(0, rob.h), 0u);
    if (rob.h <= 0 || rob.w <= 0 || rob.c <= 0 || rob.data.empty())
        return false;
    bool any = false;
    for (int y = 0; y < rob.h; ++y) {
        bool row_any = false;
        const size_t base = (size_t)y * (size_t)rob.w * (size_t)std::max(1, rob.c);
        const size_t count = (size_t)rob.w * (size_t)std::max(1, rob.c);
        for (size_t i = 0; i < count; ++i) {
            if (rob.data[base + i] != 0.f) {
                row_any = true;
                break;
            }
        }
        rows[(size_t)y] = row_any ? 1u : 0u;
        any = any || row_any;
    }
    return any;
}

static inline int round_away_to_int(f32 x) {
    return (int)std::lround(x);
}

static bool robustness_band_can_contribute(const std::vector<uint8_t>& rows,
                                           int y0, int bh, float scale,
                                           bool bayer_mode) {
    if (rows.empty()) return true;
    const int rh = (int)rows.size();
    for (int local_y = 0; local_y < bh; ++local_y) {
        const f32 lr_y = (f32)(y0 + local_y) / scale;
        int ry;
        if (bayer_mode)
            ry = std::min(std::max(round_away_to_int((lr_y - 0.5f) / 2.f), 0), rh - 1);
        else
            ry = std::min(std::max(round_away_to_int(lr_y), 0), rh - 1);
        if (rows[(size_t)ry]) return true;
    }
    return false;
}

struct PrealignTransform {
    f32 dx = 0.f;      // reference raw/grey coordinate -> moving coordinate
    f32 dy = 0.f;
    f32 angle = 0.f;   // radians, positive = counter-clockwise about image center
    f32 score = -1.f;
    bool valid = false;
};

struct PrealignPlan {
    int reference_index = 0;
    std::vector<PrealignTransform> to_first;
    std::vector<PrealignTransform> from_reference;
};

static inline f32 sample_bilinear_clamped(const Image& im, f32 y, f32 x) {
    if (im.h <= 0 || im.w <= 0) return 0.f;
    x = clampf(x, 0.f, (f32)(im.w - 1));
    y = clampf(y, 0.f, (f32)(im.h - 1));
    const int fx = (int)std::floor(x);
    const int fy = (int)std::floor(y);
    const int cx = std::min(fx + 1, im.w - 1);
    const int cy = std::min(fy + 1, im.h - 1);
    const f32 tx = x - (f32)fx;
    const f32 ty = y - (f32)fy;
    const f32 top = im.at(fy, fx) + tx * (im.at(fy, cx) - im.at(fy, fx));
    const f32 bot = im.at(cy, fx) + tx * (im.at(cy, cx) - im.at(cy, fx));
    return top + ty * (bot - top);
}

static bool sample_bilinear_inside(const Image& im, f32 y, f32 x, f32& out) {
    if (im.h <= 1 || im.w <= 1 || x < 0.f || y < 0.f ||
        x > (f32)(im.w - 1) || y > (f32)(im.h - 1))
        return false;
    out = sample_bilinear_clamped(im, y, x);
    return true;
}

static Image make_prealign_thumbnail(const Image& raw, const Config& cfg) {
    if (raw.h <= 0 || raw.w <= 0) return Image();
    const int max_dim = std::max(96, std::min(512, cfg.global_prealignment_thumb_max_dim));
    int tw = max_dim;
    int th = max_dim;
    if (raw.w >= raw.h) {
        tw = max_dim;
        th = std::max(64, (int)std::lround((double)raw.h * (double)max_dim / (double)raw.w));
    } else {
        th = max_dim;
        tw = std::max(64, (int)std::lround((double)raw.w * (double)max_dim / (double)raw.h));
    }
    tw = std::min(tw, raw.w);
    th = std::min(th, raw.h);
    Image thumb(th, tw, 1);
    const f32 sx = (f32)raw.w / (f32)tw;
    const f32 sy = (f32)raw.h / (f32)th;
    constexpr int S = 4;
    parallel_rows(th, 0, [&](int y) {
        for (int x = 0; x < tw; ++x) {
            f32 sum = 0.f;
            for (int yy = 0; yy < S; ++yy) {
                for (int xx = 0; xx < S; ++xx) {
                    const f32 rx = ((f32)x + ((f32)xx + 0.5f) / (f32)S) * sx - 0.5f;
                    const f32 ry = ((f32)y + ((f32)yy + 0.5f) / (f32)S) * sy - 0.5f;
                    sum += sample_bilinear_clamped(raw, ry, rx);
                }
            }
            thumb.at(y, x) = sum * (1.f / (f32)(S * S));
        }
    });
    return gaussian_blur(thumb, 0.6f);
}

static f32 prealign_ncc_score(const Image& ref, const Image& mov,
                              f32 dx, f32 dy, f32 angle, int stride) {
    if (ref.h != mov.h || ref.w != mov.w || ref.h <= 8 || ref.w <= 8)
        return -std::numeric_limits<f32>::infinity();
    const f32 ca = std::cos(angle);
    const f32 sa = std::sin(angle);
    const f32 cx = 0.5f * (f32)(ref.w - 1);
    const f32 cy = 0.5f * (f32)(ref.h - 1);
    const int mx = std::max(4, ref.w / 12);
    const int my = std::max(4, ref.h / 12);
    double sr = 0.0, sm = 0.0, srr = 0.0, smm = 0.0, srm = 0.0;
    int count = 0;
    for (int y = my; y < ref.h - my; y += stride) {
        for (int x = mx; x < ref.w - mx; x += stride) {
            const f32 rx = (f32)x - cx;
            const f32 ry = (f32)y - cy;
            const f32 qx = cx + ca * rx - sa * ry + dx;
            const f32 qy = cy + sa * rx + ca * ry + dy;
            f32 mv = 0.f;
            if (!sample_bilinear_inside(mov, qy, qx, mv)) continue;
            const f32 rv = ref.at(y, x);
            sr += rv;
            sm += mv;
            srr += (double)rv * (double)rv;
            smm += (double)mv * (double)mv;
            srm += (double)rv * (double)mv;
            ++count;
        }
    }
    if (count < 64) return -std::numeric_limits<f32>::infinity();
    const double n = (double)count;
    const double cov = srm - (sr * sm) / n;
    const double vr = srr - (sr * sr) / n;
    const double vm = smm - (sm * sm) / n;
    const double denom = std::sqrt(std::max(0.0, vr) * std::max(0.0, vm));
    if (!(denom > 1e-12) || !std::isfinite(denom))
        return -std::numeric_limits<f32>::infinity();
    return (f32)(cov / denom);
}

static PrealignTransform estimate_prealign_transform(const Image& ref_thumb,
                                                     const Image& mov_thumb,
                                                     const Config& cfg) {
    PrealignTransform best;
    const int max_shift = std::max(0, std::min(64, cfg.global_prealignment_max_shift));
    const f32 range_deg = std::max(0.f, std::min(2.f, cfg.global_prealignment_rotation_range_deg));
    const f32 step_deg = std::max(0.05f, std::min(1.f, cfg.global_prealignment_rotation_step_deg));
    const int stride = std::max(1, (int)std::ceil(std::sqrt(
        (double)std::max(1, ref_thumb.h * ref_thumb.w) / 7000.0)));
    const f32 deg_to_rad = 3.14159265358979323846f / 180.f;
    auto consider = [&](f32 dx, f32 dy, f32 angle) {
        const f32 score = prealign_ncc_score(ref_thumb, mov_thumb, dx, dy, angle, stride);
        if (score > best.score) {
            best.dx = dx;
            best.dy = dy;
            best.angle = angle;
            best.score = score;
        }
    };

    std::vector<f32> angles;
    if (range_deg <= 0.0001f) {
        angles.push_back(0.f);
    } else {
        for (f32 deg = -range_deg; deg <= range_deg + 0.5f * step_deg; deg += step_deg)
            angles.push_back(deg * deg_to_rad);
        angles.push_back(0.f);
    }

    const int coarse_step = max_shift >= 8 ? 2 : 1;
    for (f32 a : angles) {
        for (int sy = -max_shift; sy <= max_shift; sy += coarse_step) {
            for (int sx = -max_shift; sx <= max_shift; sx += coarse_step) {
                consider((f32)sx, (f32)sy, a);
            }
        }
    }

    const PrealignTransform coarse = best;
    const f32 fine_angle_step = range_deg > 0.f ? 0.5f * step_deg * deg_to_rad : 1.f;
    const int fine_shift = std::max(1, coarse_step);
    for (f32 da = -fine_angle_step; da <= fine_angle_step + 0.25f * fine_angle_step;
         da += fine_angle_step) {
        const f32 a = (range_deg > 0.f)
            ? std::max(-range_deg * deg_to_rad,
                       std::min(range_deg * deg_to_rad, coarse.angle + da))
            : 0.f;
        for (int sy = (int)std::floor(coarse.dy) - fine_shift;
             sy <= (int)std::ceil(coarse.dy) + fine_shift; ++sy) {
            for (int sx = (int)std::floor(coarse.dx) - fine_shift;
                 sx <= (int)std::ceil(coarse.dx) + fine_shift; ++sx) {
                if (std::abs(sx) <= max_shift && std::abs(sy) <= max_shift)
                    consider((f32)sx, (f32)sy, a);
            }
        }
    }

    best.valid = std::isfinite(best.score) && best.score > 0.03f;
    if (!best.valid) best = PrealignTransform{};
    return best;
}

static PrealignPlan build_prealign_plan_from_first(
    int frame_count, const RawFrameLoaderFn& loader, Config& work,
    const Image& first_ref, const ProgressFn& progress) {
    PrealignPlan plan;
    plan.to_first.assign((size_t)frame_count, PrealignTransform{});
    plan.from_reference.assign((size_t)frame_count, PrealignTransform{});
    if (!work.global_prealignment_enabled || frame_count < 2 ||
        !loader || first_ref.h <= 0 || first_ref.w <= 0)
        return plan;

    Image ref_thumb = make_prealign_thumbnail(first_ref, work);
    if (ref_thumb.h <= 0 || ref_thumb.w <= 0)
        return plan;

    plan.to_first[0].valid = true;
    plan.to_first[0].score = 1.f;
    const f32 thumb_to_raw_x = (f32)first_ref.w / (f32)ref_thumb.w;
    const f32 thumb_to_raw_y = (f32)first_ref.h / (f32)ref_thumb.h;
    for (int k = 1; k < frame_count; ++k) {
        if (progress) {
            progress("Pre-aligning frame " + std::to_string(k + 1),
                     0.025f + 0.035f * (float)(k - 1) / std::max(1, frame_count - 1));
        }
        Image comp = loader(k, work, false, first_ref.h, first_ref.w);
        if (comp.h <= 0 || comp.w <= 0) continue;
        Image thumb = make_prealign_thumbnail(comp, work);
        comp = Image();
        if (thumb.h != ref_thumb.h || thumb.w != ref_thumb.w) continue;
        PrealignTransform t = estimate_prealign_transform(ref_thumb, thumb, work);
        if (t.valid) {
            t.dx *= thumb_to_raw_x;
            t.dy *= thumb_to_raw_y;
        }
        plan.to_first[(size_t)k] = t;
    }

    if (work.global_prealignment_choose_reference) {
        std::vector<PrealignTransform> basic((size_t)std::max(0, frame_count - 1));
        bool all_adjacent_valid = true;
        for (int i = 1; i < frame_count; ++i) {
            const PrealignTransform a = plan.to_first[(size_t)i];
            const PrealignTransform b = plan.to_first[(size_t)i - 1u];
            all_adjacent_valid = all_adjacent_valid && a.valid && b.valid;
            if (a.valid && b.valid) {
                basic[(size_t)i - 1u].dx = a.dx - b.dx;
                basic[(size_t)i - 1u].dy = a.dy - b.dy;
                basic[(size_t)i - 1u].angle = a.angle - b.angle;
                basic[(size_t)i - 1u].valid = true;
            }
        }

        if (all_adjacent_valid && frame_count > 1) {
            f32 best_len = std::numeric_limits<f32>::infinity();
            int best_ref = 0;
            for (int i = 0; i < frame_count - 1; ++i) {
                f32 ax = 0.f, ay = 0.f;
                for (int j = 0; j < i; ++j) {
                    ax -= basic[(size_t)j].dx;
                    ay -= basic[(size_t)j].dy;
                }
                f32 bx = 0.f, by = 0.f;
                for (int j = i; j < frame_count - 1; ++j) {
                    bx += basic[(size_t)j].dx;
                    by += basic[(size_t)j].dy;
                }
                const f32 tx = ax + bx;
                const f32 ty = ay + by;
                const f32 len = std::sqrt(tx * tx + ty * ty);
                if (len < best_len) {
                    best_len = len;
                    best_ref = i;
                }
            }
            plan.reference_index = best_ref;
        }
    }

    const PrealignTransform ref_t = plan.to_first[(size_t)plan.reference_index];
    for (int k = 0; k < frame_count; ++k) {
        PrealignTransform rel;
        if (plan.to_first[(size_t)k].valid && ref_t.valid) {
            const PrealignTransform tk = plan.to_first[(size_t)k];
            rel.angle = tk.angle - ref_t.angle;
            const f32 ca = std::cos(rel.angle);
            const f32 sa = std::sin(rel.angle);
            rel.dx = tk.dx - (ca * ref_t.dx - sa * ref_t.dy);
            rel.dy = tk.dy - (sa * ref_t.dx + ca * ref_t.dy);
            rel.score = tk.score;
            rel.valid = true;
        }
        plan.from_reference[(size_t)k] = rel;
    }
    plan.from_reference[(size_t)plan.reference_index] = PrealignTransform{};
    plan.from_reference[(size_t)plan.reference_index].valid = true;
    plan.from_reference[(size_t)plan.reference_index].score = 1.f;
    return plan;
}

struct DebugImageStats {
    f32 min_v = std::numeric_limits<f32>::infinity();
    f32 max_v = -std::numeric_limits<f32>::infinity();
    double mean = 0.0;
    size_t nonfinite = 0;
};

static DebugImageStats image_stats(const Image& im) {
    DebugImageStats s;
    if (im.data.empty()) {
        s.min_v = 0.f;
        s.max_v = 0.f;
        return s;
    }
    double sum = 0.0;
    size_t finite_n = 0;
    for (f32 v : im.data) {
        if (!std::isfinite(v)) {
            ++s.nonfinite;
            continue;
        }
        s.min_v = std::min(s.min_v, v);
        s.max_v = std::max(s.max_v, v);
        sum += v;
        ++finite_n;
    }
    if (finite_n == 0) {
        s.min_v = 0.f;
        s.max_v = 0.f;
    }
    s.mean = finite_n ? sum / (double)finite_n : 0.0;
    return s;
}

struct DebugCovStats {
    f32 min_det = std::numeric_limits<f32>::infinity();
    f32 max_det = -std::numeric_limits<f32>::infinity();
    size_t det_nonfinite = 0;
    size_t det_nonpositive = 0;
    size_t det_tiny = 0;
};

static DebugCovStats cov_stats(const CovField& covs) {
    DebugCovStats s;
    if (covs.cov.empty()) {
        s.min_det = 0.f;
        s.max_det = 0.f;
        return s;
    }
    const size_t n = (size_t)covs.h * (size_t)covs.w;
    for (size_t i = 0; i < n; ++i) {
        const f32* c = &covs.cov[i * 4u];
        f32 det = c[0] * c[3] - c[1] * c[1];
        if (!std::isfinite(det)) {
            ++s.det_nonfinite;
            continue;
        }
        s.min_det = std::min(s.min_det, det);
        s.max_det = std::max(s.max_det, det);
        if (det <= 0.f) ++s.det_nonpositive;
        else if (det < 1e-12f) ++s.det_tiny;
    }
    if (!std::isfinite(s.min_det)) s.min_det = 0.f;
    if (!std::isfinite(s.max_det)) s.max_det = 0.f;
    return s;
}

struct MergeDebugStats {
    size_t pixels = 0;
    size_t channel_zero[3] = {0, 0, 0};
    size_t channel_tiny[3] = {0, 0, 0};
    size_t channel_nonfinite[3] = {0, 0, 0};
    size_t all_zero = 0;
    size_t only_green = 0;
    f32 min_den[3] = {
        std::numeric_limits<f32>::infinity(),
        std::numeric_limits<f32>::infinity(),
        std::numeric_limits<f32>::infinity()
    };
    f32 max_den[3] = {
        -std::numeric_limits<f32>::infinity(),
        -std::numeric_limits<f32>::infinity(),
        -std::numeric_limits<f32>::infinity()
    };
    std::string samples;
    size_t sample_count = 0;
};

static bool analyze_merge_band(const Image& num, const Image& den, int y0,
                               MergeDebugStats& total) {
    constexpr f32 tiny = 1e-12f;
    bool suspicious_band = false;
    const int nch = std::min(3, den.c);
    for (int y = 0; y < den.h; ++y) {
        for (int x = 0; x < den.w; ++x) {
            ++total.pixels;
            f32 d[3] = {0.f, 0.f, 0.f};
            f32 n[3] = {0.f, 0.f, 0.f};
            bool bad_pixel = false;
            for (int ch = 0; ch < nch; ++ch) {
                d[ch] = den.at(y, x, ch);
                n[ch] = num.at(y, x, ch);
                if (std::isfinite(d[ch])) {
                    total.min_den[ch] = std::min(total.min_den[ch], d[ch]);
                    total.max_den[ch] = std::max(total.max_den[ch], d[ch]);
                    if (d[ch] == 0.f) {
                        ++total.channel_zero[ch];
                        bad_pixel = true;
                    } else if (d[ch] > 0.f && d[ch] < tiny) {
                        ++total.channel_tiny[ch];
                        bad_pixel = true;
                    }
                } else {
                    ++total.channel_nonfinite[ch];
                    bad_pixel = true;
                }
            }
            const bool r_bad = nch > 0 && (!std::isfinite(d[0]) || d[0] <= tiny);
            const bool g_ok = nch > 1 && std::isfinite(d[1]) && d[1] > tiny;
            const bool b_bad = nch > 2 && (!std::isfinite(d[2]) || d[2] <= tiny);
            const bool all_bad = (nch == 1)
                ? r_bad
                : (r_bad && (!g_ok) && b_bad);
            const bool only_g = (nch >= 3 && r_bad && g_ok && b_bad);
            if (all_bad) {
                ++total.all_zero;
                bad_pixel = true;
            }
            if (only_g) {
                ++total.only_green;
                bad_pixel = true;
            }
            if (bad_pixel) {
                suspicious_band = true;
                if (total.sample_count < 2000) {
                    char line[320];
                    std::snprintf(line, sizeof(line),
                        "y=%d x=%d den=(%.9g,%.9g,%.9g) num=(%.9g,%.9g,%.9g)\n",
                        y0 + y, x, d[0], d[1], d[2], n[0], n[1], n[2]);
                    total.samples += line;
                    ++total.sample_count;
                }
            }
        }
    }
    return suspicious_band;
}

static void append_image_summary(std::ostringstream& ss, const char* name, const Image& im) {
    DebugImageStats s = image_stats(im);
    ss << name << " shape=" << im.h << "x" << im.w << "x" << im.c
       << " min=" << s.min_v << " max=" << s.max_v
       << " mean=" << s.mean << " nonfinite=" << s.nonfinite << "\n";
}

static void append_cov_summary(std::ostringstream& ss, const char* name, const CovField& covs) {
    DebugCovStats s = cov_stats(covs);
    ss << name << " shape=" << covs.h << "x" << covs.w
       << " det_min=" << s.min_det << " det_max=" << s.max_det
       << " det_nonfinite=" << s.det_nonfinite
       << " det_nonpositive=" << s.det_nonpositive
       << " det_tiny=" << s.det_tiny << "\n";
}

static void append_merge_summary(std::ostringstream& ss, const MergeDebugStats& s) {
    ss << "merge_pixels=" << s.pixels << "\n";
    const char* channel_name[3] = {"R", "G", "B"};
    for (int ch = 0; ch < 3; ++ch) {
        const f32 min_den = std::isfinite(s.min_den[ch]) ? s.min_den[ch] : 0.f;
        const f32 max_den = std::isfinite(s.max_den[ch]) ? s.max_den[ch] : 0.f;
        ss << "den_ch" << ch << "_" << channel_name[ch]
           << " min=" << min_den
           << " max=" << max_den
           << " zero=" << s.channel_zero[ch]
           << " tiny=" << s.channel_tiny[ch]
           << " nonfinite=" << s.channel_nonfinite[ch] << "\n";
    }
    ss << "den_all_zero=" << s.all_zero << "\n";
    ss << "den_only_green=" << s.only_green << "\n";
    ss << "den_sample_count=" << s.sample_count << "\n";
    if (!s.samples.empty()) {
        ss << "den_samples\n";
        ss << s.samples;
    }
}

static void absorb_robustness_sum(Image& acc_rob, const Image& rob, bool& have) {
    if (!have) {
        acc_rob = Image(rob.h, rob.w, 1);
        acc_rob.data = rob.data;
        have = true;
        return;
    }
    for (size_t i = 0; i < rob.data.size(); ++i)
        acc_rob.data[i] += rob.data[i];
}

static void build_robustness_sum(const std::vector<CachedCompFrame>& cached,
                                 const std::vector<CachedCompMeta>& cached_meta,
                                 bool stream_comp_raw,
                                 Image& acc_rob, bool& have) {
    if (stream_comp_raw) {
        for (const CachedCompMeta& meta : cached_meta)
            absorb_robustness_sum(acc_rob, meta.rob, have);
    } else {
        for (const CachedCompFrame& fc : cached)
            absorb_robustness_sum(acc_rob, fc.rob, have);
    }
}

static void encode_band_rows(const Image& num_band, const Image& den_band, int y0, int bh,
                             const Config& work, int nch, Image& preview, float pscale,
                             int ph, int pw, int Ws, std::vector<uint16_t>& row16) {
    // Same num/den → RGB16 math as before; pointer loops + sparse preview only.
    auto to_srgb = [](f32 v) {
        v = clampf(v, 0.f, 1.f);
        return v <= 0.0031308f ? 12.92f * v : 1.055f * std::pow(v, 1.f / 2.4f) - 0.055f;
    };
    const int x_step = std::max(1, (int)std::ceil(1.f / std::max(pscale, 1e-6f)));
    // Preview is UI-only; sample a bit more sparsely (DNG pixels unchanged).
    const int y_step = std::max(1, x_step);
    const f32* nump = num_band.data.data();
    const f32* denp = den_band.data.data();
    const bool bake = work.bake_srgb && nch >= 3;
    const f32* m = work.cam_to_srgb;
    // Pre-whitened RAW already has WB baked in (Python utils_dng order).
    const f32 wb0 = work.raw_prewhitened ? 1.f : work.white_balance[0];
    const f32 wb1 = work.raw_prewhitened ? 1.f : work.white_balance[1];
    const f32 wb2 = work.raw_prewhitened ? 1.f : work.white_balance[2];
    const bool prev_color = !bake && nch >= 3 && work.has_cam_to_srgb;

#if defined(__APPLE__)
    // Dense DNG band on GPU (1:1); sparse preview stays on CPU below.
    const bool gpu_rgb = metal_normalize_band_rgb16(num_band, den_band, work, row16);
#else
    const bool gpu_rgb = false;
#endif
    uint16_t* outp = row16.data();
    if (!gpu_rgb) {
        row16.resize((size_t)bh * (size_t)Ws * 3u);
        outp = row16.data();
    }

    parallel_rows(bh, work.num_threads, [&](int i) {
        const int gy = y0 + i;
        const bool do_prev_row = ((gy % y_step) == 0);
        const int py = std::min(ph - 1, (int)(gy * pscale));
        const size_t row_off = (size_t)i * (size_t)Ws * (size_t)nch;
        for (int x = 0; x < Ws; ++x) {
            const bool need_prev = do_prev_row && (x % x_step) == 0;
            if (gpu_rgb && !need_prev) continue;

            const size_t pi = row_off + (size_t)x * (size_t)nch;
            f32 d0 = denp[pi];
            f32 cn0 = (d0 > 0.f) ? nump[pi] / d0 : 0.f;
            f32 cn1 = 0.f, cn2 = 0.f;
            if (nch >= 2) {
                f32 d1 = denp[pi + 1];
                cn1 = (d1 > 0.f) ? nump[pi + 1] / d1 : 0.f;
            }
            if (nch >= 3) {
                f32 d2 = denp[pi + 2];
                cn2 = (d2 > 0.f) ? nump[pi + 2] / d2 : 0.f;
            }
            f32 lin0, lin1, lin2;
            if (bake) {
                f32 wr = cn0 * wb0, wg = cn1 * wb1, wb = cn2 * wb2;
                lin0 = m[0] * wr + m[1] * wg + m[2] * wb;
                lin1 = m[3] * wr + m[4] * wg + m[5] * wb;
                lin2 = m[6] * wr + m[7] * wg + m[8] * wb;
            } else if (nch >= 3) {
                lin0 = cn0; lin1 = cn1; lin2 = cn2;
            } else {
                lin0 = lin1 = lin2 = cn0;
            }
            if (!gpu_rgb) {
                const f32 v0 = bake ? to_srgb(lin0) : clampf(lin0, 0.f, 1.f);
                const f32 v1 = bake ? to_srgb(lin1) : clampf(lin1, 0.f, 1.f);
                const f32 v2 = bake ? to_srgb(lin2) : clampf(lin2, 0.f, 1.f);
                const size_t base = ((size_t)i * (size_t)Ws + (size_t)x) * 3u;
                outp[base + 0] = (uint16_t)(v0 * 65535.f + 0.5f);
                outp[base + 1] = (uint16_t)(v1 * 65535.f + 0.5f);
                outp[base + 2] = (uint16_t)(v2 * 65535.f + 0.5f);
            }
            if (need_prev) {
                f32 preview_lin[3] = {lin0, lin1, lin2};
                if (prev_color) {
                    f32 wr = cn0 * wb0, wg = cn1 * wb1, wb = cn2 * wb2;
                    preview_lin[0] = m[0] * wr + m[1] * wg + m[2] * wb;
                    preview_lin[1] = m[3] * wr + m[4] * wg + m[5] * wb;
                    preview_lin[2] = m[6] * wr + m[7] * wg + m[8] * wb;
                } else if (!bake && nch >= 3) {
                    preview_lin[0] = cn0 * wb0;
                    preview_lin[1] = cn1 * wb1;
                    preview_lin[2] = cn2 * wb2;
                }
                const int px = std::min(pw - 1, (int)(x * pscale));
                for (int k = 0; k < 3; ++k)
                    preview.at(py, px, k) = to_srgb(clampf(preview_lin[k], 0.f, 1.f));
            }
        }
    });
}

} // namespace

Image process_burst_loader_to_dng(int frame_count, const RawFrameLoaderFn& loader,
                                  const Config& cfg, const std::string& dng_path,
                                  const ProgressFn& progress, int maxPreviewDim) {
    if (frame_count < 2 || !loader) return Image();

    prof_reset();
    const double t_burst = prof_now_ms();

    Config work = cfg;
    auto report = [&](const std::string& s, float f) { if (progress) progress(s, f); };
    const bool debug = debug_dumps_enabled();
    std::ostringstream debug_summary;
    if (debug) debug_summary << "cpp_debug_summary\n";

    report("Loading first frame", 0.02f);
    const double t_first = prof_now_ms();
    Image first_ref = loader(0, work, true, 0, 0);
    prof_add_cpu("ref:decode", prof_now_ms() - t_first);
    if (first_ref.h <= 0 || first_ref.w <= 0) return Image();

    PrealignPlan prealign = build_prealign_plan_from_first(
        frame_count, loader, work, first_ref, progress);
    const int ref_index = std::max(0, std::min(frame_count - 1, prealign.reference_index));
    Image ref;
    if (ref_index == 0) {
        ref = std::move(first_ref);
    } else {
        report("Loading selected reference frame", 0.06f);
        ref = loader(ref_index, work, true, 0, 0);
        first_ref = Image();
    }
    if (ref.h <= 0 || ref.w <= 0) return Image();

    clear_align_ref_ica_cache();
    debug_dump_bin("cpp_raw_ref", ref.data.data(), ref.data.size());
    if (debug) append_image_summary(debug_summary, "raw_ref", ref);
    if (debug) {
        std::string listing = "ref_index=" + std::to_string(ref_index) + "\n";
        for (int i = 0; i < frame_count; ++i) {
            listing += std::to_string(i) + " provider_frame";
            if (i < (int)prealign.from_reference.size()) {
                const PrealignTransform& t = prealign.from_reference[(size_t)i];
                listing += " pre_dx=" + std::to_string(t.dx);
                listing += " pre_dy=" + std::to_string(t.dy);
                listing += " pre_angle_rad=" + std::to_string(t.angle);
                listing += " score=" + std::to_string(t.score);
            }
            listing += "\n";
        }
        debug_dump_text("cpp_burst_paths", listing);
    }
    tune_config_snr(ref, work);

    // Same UI status line as "Frame N: analyze" (CameraModel.statusText).
    {
        f32 sum = 0.f;
        for (f32 v : ref.data) sum += v;
        const f32 brightness = ref.data.empty() ? 0.f : sum / (f32)ref.data.size();
        const f32 sigma = noise_std_at_brightness(brightness, work.alpha, work.beta);
        const f32 snr = (sigma > 1e-8f) ? brightness / sigma : 0.f;
        char buf[288];
        const int t0 = work.bm_tile_sizes.size() > 0 ? work.bm_tile_sizes[0] : -1;
        std::snprintf(buf, sizeof(buf),
            "Noise %s α=%.3g β=%.3g  b=%.2f σ=%.2e SNR=%.1f  T=%d  r_t=%.2f",
            work.has_noise_profile ? "OK" : "FALLBACK",
            work.alpha, work.beta, brightness, sigma, snr, t0, work.r_t);
        report(buf, 0.065f);
    }

    const int ref_h = ref.h, ref_w = ref.w;
    const int n = frame_count;
    const int tile_size = work.bm_tile_sizes.empty() ? 16 : work.bm_tile_sizes[0];
    const int nch = work.bayer_mode ? 3 : 1;
    std::vector<int> comp_indices;
    comp_indices.reserve((size_t)std::max(0, n - 1));
    for (int i = 0; i < n; ++i) {
        if (i != ref_index) comp_indices.push_back(i);
    }

    report("Reference: grey + pyramid", 0.07f);
    prof_mark_memory("ref:start");
    const double t_ref_grey = prof_now_ms();
    // 460-main block matching circular-pads the reference before pyramid construction.
    Image ref_grey = compute_grey(ref, work.bayer_mode, work.grey_method);
    debug_dump_bin("cpp_ref_grey", ref_grey.data.data(), ref_grey.data.size());
    Image ref_grey_padded = pad_image_circular(ref_grey, tile_size);
    Pyramid ref_pyr = build_pyramid(ref_grey_padded, work.bm_factors);
    prof_add_cpu("ref:grey+pyramid", prof_now_ms() - t_ref_grey);
    if (debug && !ref_pyr.levels.empty()) {
        // Match Python py_pyramid_0: first after pyramid[::-1] = coarsest.
        const Image& coarse = ref_pyr.levels.back();
        debug_dump_bin("cpp_pyramid_0", coarse.data.data(), coarse.data.size());
    }
    // Full-res (~12MP Bayer) cannot hold every comparison RAW + dual Metal peaks.
    const bool full_res =
        ((size_t)ref_h * (size_t)ref_w) >= 8ull * 1000ull * 1000ull;

    // Same math; sequential on Apple Metal (rob + kernels both GPU — overlap
    // doubles peak and races shared scratch). Light crops may still overlap.
    RefStats ref_stats;
    CovField ref_covs;
    const double t_ref_analyze = prof_now_ms();
    if (full_res) {
        ref_stats = init_robustness(ref, work);
        ref_covs = estimate_kernels(ref, work);
    } else {
        std::future<CovField> ref_cov_fut =
            std::async(std::launch::async, [&]() { return estimate_kernels(ref, work); });
        ref_stats = init_robustness(ref, work);
        ref_covs = ref_cov_fut.get();
    }
    prof_add_cpu("ref:robustness+kernels", prof_now_ms() - t_ref_analyze);
    prof_mark_memory("ref:analyzed");
    if (debug) append_cov_summary(debug_summary, "cov_ref", ref_covs);
#if defined(__APPLE__)
    // GPU already holds ref means/vars; drop host copies to cut peak RAM.
    metal_release_host_ref_stats(ref_stats);
#endif

    // Keep ref Bayer in RAM for merge (avoids a second LibRaw decode).
    // Peak during analyze ≈ ref + one comparison (+ optional prefetch on 2×).

    fs::path cache = fs::path(dng_path).parent_path() /
                     (fs::path(dng_path).stem().string() + "_cache");
    std::error_code ec;
    fs::remove_all(cache, ec);

    // Stream Bayer on full-res (any burst size) or when comps > 8. Keeps flow/R/cov.
    std::vector<CachedCompFrame> cached;
    std::vector<CachedCompMeta> cached_meta;
    const bool stream_comp_raw = full_res || (n - 1) > 8;
    const bool cache_streamed_comp_raw =
        stream_comp_raw && !work.stream_comp_raw_from_loader;
    auto load_streamed_comp_raw = [&](int frame_index, Image& comp) -> bool {
        if (cache_streamed_comp_raw)
            return load_cached_comp_raw(cache, frame_index, comp);
        comp = loader(frame_index, work, false, ref_h, ref_w);
        return comp.h > 0 && comp.w > 0;
    };
    if (cache_streamed_comp_raw)
        fs::create_directories(cache, ec);

    cached.reserve((size_t)std::max(0, n - 1));
    cached_meta.reserve((size_t)std::max(0, n - 1));

    // Prefetch next LibRaw decode: 2× during whole analyze; 1× only after grey
    // is freed (overlaps rob/kernels, +1 Bayer peak briefly).
    int pref_k = -1;
    std::future<Image> pref_fut;
    std::future<bool> spill_fut;
    bool spill_pending = false;
    int n_comp_ok = 0;
    auto drain_spill = [&]() {
        if (!spill_fut.valid()) return;
        const bool ok = spill_fut.get();
        if (!ok && spill_pending && !cached_meta.empty()) {
            cached_meta.pop_back();
            n_comp_ok = std::max(0, n_comp_ok - 1);
        }
        spill_pending = false;
    };
    for (int pos = 0; pos < (int)comp_indices.size(); ++pos) {
        const int k = comp_indices[(size_t)pos];
        report("Frame " + std::to_string(k + 1) + ": analyze",
               0.08f + 0.35f * (float)pos / std::max(1, n - 1));
        drain_spill();

        Image comp;
        const double t_decode = prof_now_ms();
        if (pref_k == k && pref_fut.valid()) {
            comp = pref_fut.get();
            pref_k = -1;
            // Stall only. Near-zero here means the prefetch fully hid the decode.
            prof_add_cpu("comp:decode-wait(prefetched)", prof_now_ms() - t_decode);
        } else {
            comp = loader(k, work, false, ref_h, ref_w);
            prof_add_cpu("comp:decode(sync)", prof_now_ms() - t_decode);
        }
        if (comp.h <= 0) continue;
        debug_dump_bin("cpp_raw_comp_" + std::to_string(pos),
                       comp.data.data(), comp.data.size());
        if (debug) {
            const std::string raw_name = "raw_comp_" + std::to_string(pos);
            append_image_summary(debug_summary, raw_name.c_str(), comp);
        }

        // 2×: decode next during align. 1×: wait until after grey (lower peak).
        if (!full_res && pos + 1 < (int)comp_indices.size()) {
            const int nk = comp_indices[(size_t)pos + 1u];
            pref_k = nk;
            pref_fut = std::async(std::launch::async, [&, nk]() {
                return loader(nk, work, false, ref_h, ref_w);
            });
        }

        const double t_comp_grey = prof_now_ms();
        Image comp_grey = compute_grey(comp, work.bayer_mode, work.grey_method);
        prof_add_cpu("comp:grey", prof_now_ms() - t_comp_grey);
        debug_dump_bin("cpp_mov_grey_" + std::to_string(pos),
                       comp_grey.data.data(), comp_grey.data.size());
        PrealignTransform init;
        if (k >= 0 && k < (int)prealign.from_reference.size())
            init = prealign.from_reference[(size_t)k];
        const f32 grey_scale_x = ref_w > 0 ? (f32)ref_grey.w / (f32)ref_w : 1.f;
        const f32 grey_scale_y = ref_h > 0 ? (f32)ref_grey.h / (f32)ref_h : 1.f;
        const double t_align = prof_now_ms();
        FlowField flow = align(ref_pyr, ref_grey, comp_grey, work, tile_size,
                               init.dx * grey_scale_x,
                               init.dy * grey_scale_y,
                               init.angle);
        prof_add_cpu("comp:align", prof_now_ms() - t_align);
        debug_dump_bin("cpp_flow_" + std::to_string(pos),
                       flow.flow.data(), flow.flow.size());
        comp_grey = Image(); // free before robustness/kernels peak
        if (flow.ny <= 0 || flow.nx <= 0 || flow.flow.empty()) {
            report("Frame " + std::to_string(k + 1) + ": Metal alignment failed",
                   0.08f + 0.35f * (float)(pos + 1) / std::max(1, n - 1));
            continue;
        }

        // Full-res: decode next while Metal rob/kernels run (grey already freed).
        if (full_res && pos + 1 < (int)comp_indices.size()) {
            const int nk = comp_indices[(size_t)pos + 1u];
            pref_k = nk;
            pref_fut = std::async(std::launch::async, [&, nk]() {
                return loader(nk, work, false, ref_h, ref_w);
            });
        }

        Image rob;
        CovField covs;
        std::vector<uint8_t> rob_rows_nonzero;
        bool rob_has_nonzero = true;
        if (full_res) {
            // Keep rob ∥ kernels serialized — dual Metal peaks jetsam on 1×.
            const double t_rob = prof_now_ms();
            rob = compute_robustness(comp, ref_stats, flow, tile_size, work);
            prof_add_cpu("comp:robustness", prof_now_ms() - t_rob);
            rob_has_nonzero = robustness_row_activity(rob, rob_rows_nonzero);
            if (rob_has_nonzero) {
                const double t_kern = prof_now_ms();
                covs = estimate_kernels(comp, work);
                prof_add_cpu("comp:kernels", prof_now_ms() - t_kern);
            }
        } else {
            // Same math; overlap only when peak RAM is affordable (2× crop).
            std::future<CovField> cov_fut =
                std::async(std::launch::async, [&]() { return estimate_kernels(comp, work); });
            rob = compute_robustness(comp, ref_stats, flow, tile_size, work);
            covs = cov_fut.get();
            rob_has_nonzero = robustness_row_activity(rob, rob_rows_nonzero);
        }
        debug_dump_bin("cpp_mask_" + std::to_string(pos),
                       rob.data.data(), rob.data.size());
        if (debug) {
            const std::string mask_name = "mask_" + std::to_string(pos);
            const std::string cov_name = "cov_comp_" + std::to_string(pos);
            append_image_summary(debug_summary, mask_name.c_str(), rob);
            append_cov_summary(debug_summary, cov_name.c_str(), covs);
        }
        if (stream_comp_raw) {
            // Keep flow/R/cov in RAM. For DNG-file input, spill normalized Bayer
            // async; for direct RAW, reload the original uint16 frame later.
            // Grow-only L2/Alg.5 scratch stays until merge.
            if (cache_streamed_comp_raw && rob_has_nonzero) {
                const int sk = k;
                auto spill_img = std::make_shared<Image>(std::move(comp));
                const fs::path spill_path = cache / ("f" + std::to_string(sk) + ".raw");
                spill_fut = std::async(std::launch::async, [spill_path, spill_img]() {
                    return save_image(spill_path, *spill_img);
                });
                spill_pending = true;
            } else {
                comp = Image();
            }
            CachedCompMeta meta;
            meta.index = k;
            meta.flow = rob_has_nonzero ? std::move(flow) : FlowField();
            meta.rob = std::move(rob);
            meta.covs = rob_has_nonzero ? std::move(covs) : CovField();
            meta.rob_rows_nonzero = std::move(rob_rows_nonzero);
            meta.rob_has_nonzero = rob_has_nonzero;
            cached_meta.push_back(std::move(meta));
            n_comp_ok++;
        } else {
            CachedCompFrame fc;
            fc.index = k;
            fc.flow = rob_has_nonzero ? std::move(flow) : FlowField();
            fc.rob = std::move(rob);
            fc.covs = rob_has_nonzero ? std::move(covs) : CovField();
            fc.comp = rob_has_nonzero ? std::move(comp) : Image();
            fc.rob_rows_nonzero = std::move(rob_rows_nonzero);
            fc.rob_has_nonzero = rob_has_nonzero;
            cached.push_back(std::move(fc));
            n_comp_ok++;
        }
        prof_mark_memory("analyze:frame-end");
    }
    drain_spill();
    if (pref_fut.valid()) (void)pref_fut.get(); // drain unused prefetch

    if (n_comp_ok < 1) {
        if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
        report("Error: could not analyze comparison frames", 1.f);
        return Image();
    }

    // Release reference-side helpers not needed during merge.
    clear_align_ref_ica_cache();
    ref_grey = Image();
    ref_pyr = Pyramid();
    ref_stats = RefStats();

    if (ref.h <= 0) {
        if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
        report("Error: reference frame missing for merge", 1.f);
        return Image();
    }

    const int Hs = (int)std::lround(work.scale * ref.h);
    const int Ws = (int)std::lround(work.scale * ref.w);

    const bool accumulate_r =
        work.accumulated_robustness_denoiser_enabled || work.robustness_save_mask;
    Image acc_rob;
    bool have_acc_rob = false;
    build_robustness_sum(cached, cached_meta, stream_comp_raw, acc_rob, have_acc_rob);
    const Image* acc_rob_ptr = (accumulate_r && have_acc_rob) ? &acc_rob : nullptr;
    if (have_acc_rob) {
        debug_dump_bin("cpp_acc_rob", acc_rob.data.data(), acc_rob.data.size());
        if (debug) append_image_summary(debug_summary, "acc_rob", acc_rob);
    }

    DngStreamWriter writer;
    const std::string& model = work.camera_model.empty() ? std::string("HandheldSR-x2") : work.camera_model;
    const std::string& make = work.camera_make.empty() ? std::string("HandheldSR") : work.camera_make;
    if (!writer.open(dng_path, Ws, Hs, model, work.orientation,
                     work.has_color_matrix ? work.color_matrix : nullptr,
                     work.white_balance,
                     work.bake_srgb, make,
                     work.has_cam_to_srgb ? work.cam_to_srgb : nullptr,
                     work.raw_prewhitened)) {
        if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
        report("Error: cannot open output DNG", 1.f);
        return Image();
    }

    const float pscale = std::min(1.f, (float)maxPreviewDim / (float)std::max(Hs, Ws));
    const int ph = std::max(1, (int)(Hs * pscale));
    const int pw = std::max(1, (int)(Ws * pscale));
    Image preview(ph, pw, 3);

    // Band size: 2× double-buffers GPU+host. Full 1× keeps one GPU acc slot
    // (jetsam-safe) but dual host bands so Deflate can overlap the next GPU band.
    const size_t out_px = (size_t)Hs * (size_t)Ws;
    const bool heavy_1x = out_px >= 28ull * 1000ull * 1000ull; // ~full-res scale-2
#if defined(__APPLE__)
    const size_t band_budget = heavy_1x ? (192u * 1024u * 1024u) : (384u * 1024u * 1024u);
#else
    const size_t band_budget = 64u * 1024u * 1024u;
#endif
    const size_t bytes_per_row = (size_t)Ws * nch * 4 * 2; // num+den float row
    int band_rows = (int)std::max<size_t>(4, band_budget / std::max<size_t>(1, bytes_per_row));
    band_rows = std::min(band_rows, Hs);
    // ~960 rows fits the 192MB single-slot budget at full-res scale-2.
    if (heavy_1x) band_rows = std::min(band_rows, 960);
    std::vector<uint16_t> row16((size_t)band_rows * (size_t)Ws * 3u);

    Image comp_scratch;
#if defined(__APPLE__)
    // Upload comparison frames to GPU before the band loop so merge isn't
    // stalled on the first band's PCIe copies. On full-res, drop host R/cov
    // after each upload so we never hold host+GPU copies of every frame.
    report("Preparing GPU merge", 0.46f);
    const double t_prefetch = prof_now_ms();
    metal_merge_set_single_acc_slot(heavy_1x);
    metal_merge_begin_burst();
    if (stream_comp_raw) {
        for (CachedCompMeta& meta : cached_meta) {
            if (!meta.rob_has_nonzero) continue;
            if (!load_streamed_comp_raw(meta.index, comp_scratch)) continue;
            if (metal_merge_prefetch_frame(comp_scratch, meta.flow, meta.covs, meta.rob,
                                           meta.index)) {
                meta.rob = Image();
                meta.covs = CovField();
                meta.flow = FlowField();
            }
            if (heavy_1x) comp_scratch = Image();
        }
        comp_scratch = Image();
    } else {
        for (CachedCompFrame& fc : cached) {
            if (!fc.rob_has_nonzero) continue;
            if (metal_merge_prefetch_frame(fc.comp, fc.flow, fc.covs, fc.rob, fc.index)) {
                fc.comp = Image();
                if (heavy_1x) {
                    fc.rob = Image();
                    fc.covs = CovField();
                    fc.flow = FlowField();
                }
            }
        }
    }
    // Includes the disk reload / re-decode of every comparison frame on 1×.
    prof_add_cpu("merge:gpu-upload", prof_now_ms() - t_prefetch);
    prof_mark_memory("merge:frames-resident");
#endif

    AccumDiag diag;
    MergeDebugStats merge_debug;
#if defined(__APPLE__)
    // Dual host bands + async Deflate for both 1× and 2×. 1× still uses a
    // single GPU acc slot (ensure_acc waits/readbacks before reuse).
    Image num_bands[2], den_bands[2];
    std::vector<uint16_t> row16_async[2];
    row16_async[0].resize(row16.size());
    row16_async[1].resize(row16.size());
    int cur = 0;
    bool have_ready = false;
    int ready = 0, ready_y0 = 0, ready_bh = 0;
    std::thread encode_thr;
    auto join_encode = [&]() {
        if (encode_thr.joinable()) encode_thr.join();
    };
    report("Merging output", 0.48f);
    for (int y0 = 0; y0 < Hs; y0 += band_rows) {
        const double t_band = prof_now_ms();
        const int bh = std::min(band_rows, Hs - y0);
        cur ^= 1;
        const double t_join = prof_now_ms();
        join_encode();
        prof_add_cpu("merge:wait-dng-encode", prof_now_ms() - t_join);
        Image& num_band = num_bands[cur];
        Image& den_band = den_bands[cur];
        if (num_band.h != bh || num_band.w != Ws || num_band.c != nch) {
            try {
                num_band = Image(bh, Ws, nch);
                den_band = Image(bh, Ws, nch);
            } catch (...) {
                report("Error: out of memory during merge", 1.f);
                join_encode();
                metal_merge_set_single_acc_slot(false);
                writer.close();
                if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
                return Image();
            }
        }

        // On 1× (single acc slot) the previous band's waitUntilCompleted +
        // readback happens inside the first ensure_acc_buffers below, so this
        // bucket is "GPU stall + encode", not encode alone.
        const double t_comp_bands = prof_now_ms();
        if (stream_comp_raw) {
            for (const CachedCompMeta& meta : cached_meta) {
                if (!meta.rob_has_nonzero ||
                    !robustness_band_can_contribute(meta.rob_rows_nonzero, y0, bh,
                                                    work.scale, work.bayer_mode))
                    continue;
                if (metal_merge_has_frame(meta.index)) {
                    Image empty;
                    merge_comp_band(empty, meta.flow, meta.covs, meta.rob, tile_size,
                                    num_band, den_band, y0, work, meta.index);
                    continue;
                }
                if (!load_streamed_comp_raw(meta.index, comp_scratch)) continue;
                merge_comp_band(comp_scratch, meta.flow, meta.covs, meta.rob, tile_size,
                                num_band, den_band, y0, work, meta.index);
            }
        } else {
            for (const CachedCompFrame& fc : cached) {
                if (!fc.rob_has_nonzero ||
                    !robustness_band_can_contribute(fc.rob_rows_nonzero, y0, bh,
                                                    work.scale, work.bayer_mode))
                    continue;
                if (metal_merge_has_frame(fc.index)) {
                    Image empty;
                    merge_comp_band(empty, fc.flow, fc.covs, fc.rob, tile_size,
                                    num_band, den_band, y0, work, fc.index);
                    continue;
                }
                if (fc.comp.h <= 0) continue;
                merge_comp_band(fc.comp, fc.flow, fc.covs, fc.rob, tile_size,
                                num_band, den_band, y0, work, fc.index);
            }
        }

        prof_add_cpu("merge:comp-encode+gpu-stall", prof_now_ms() - t_comp_bands);

        const double t_ref_band = prof_now_ms();
        const bool ref_band_ok =
            merge_ref_band_metal(ref, ref_covs, num_band, den_band, y0, work, acc_rob_ptr);
        prof_add_cpu("merge:ref-encode+commit", prof_now_ms() - t_ref_band);
        if (!ref_band_ok) {
            report("Error: GPU merge failed (memory?)", 1.f);
            join_encode();
            metal_merge_set_single_acc_slot(false);
            writer.close();
            if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
            return Image();
        }

        // Previous band is resident on host once ensure_acc waited (1×) or
        // ping-pong resolved (2×). Encode it while this band's GPU runs.
        if (have_ready) {
            const int er = ready, ey0 = ready_y0, ebh = ready_bh;
            if (debug && analyze_merge_band(num_bands[er], den_bands[er], ey0, merge_debug)) {
                const std::string ys = std::to_string(ey0);
                debug_dump_bin("cpp_num_bad_band_y" + ys,
                               num_bands[er].data.data(), num_bands[er].data.size());
                debug_dump_bin("cpp_den_bad_band_y" + ys,
                               den_bands[er].data.data(), den_bands[er].data.size());
            }
            std::vector<uint16_t>& out16 = row16_async[er];
            if (out16.size() < (size_t)ebh * (size_t)Ws * 3u)
                out16.resize((size_t)ebh * (size_t)Ws * 3u);
            encode_thr = std::thread([&, er, ey0, ebh]() {
                encode_band_rows(num_bands[er], den_bands[er], ey0, ebh, work, nch,
                                 preview, pscale, ph, pw, Ws, out16);
                writer.write_rows(out16.data(), ebh);
            });
            report("Merging output", 0.48f + 0.50f * (float)(ey0 + ebh) / Hs);
        }
        ready = cur;
        ready_y0 = y0;
        ready_bh = bh;
        have_ready = true;
        prof_add_cpu("merge:band(total)", prof_now_ms() - t_band);
        prof_mark_memory("merge:band");
    }
    join_encode();
    if (have_ready && metal_merge_wait_inflight()) {
        Image& rn = num_bands[ready];
        Image& rd = den_bands[ready];
        accumulate_diag(rn, rd, diag);
        if (debug && analyze_merge_band(rn, rd, ready_y0, merge_debug)) {
            const std::string ys = std::to_string(ready_y0);
            debug_dump_bin("cpp_num_bad_band_y" + ys, rn.data.data(), rn.data.size());
            debug_dump_bin("cpp_den_bad_band_y" + ys, rd.data.data(), rd.data.size());
        }
        if (row16.size() < (size_t)ready_bh * (size_t)Ws * 3u)
            row16.resize((size_t)ready_bh * (size_t)Ws * 3u);
        encode_band_rows(rn, rd, ready_y0, ready_bh, work, nch, preview, pscale, ph, pw, Ws, row16);
        writer.write_rows(row16.data(), ready_bh);
        report("Merging output", 0.48f + 0.50f * (float)(ready_y0 + ready_bh) / Hs);
    }
    metal_merge_set_single_acc_slot(false);
#else
    Image num_band, den_band;
    report("Merging output", 0.48f);
    for (int y0 = 0; y0 < Hs; y0 += band_rows) {
        const int bh = std::min(band_rows, Hs - y0);
        if (num_band.h != bh || num_band.w != Ws || num_band.c != nch) {
            num_band = Image(bh, Ws, nch);
            den_band = Image(bh, Ws, nch);
        } else {
            std::fill(num_band.data.begin(), num_band.data.end(), 0.f);
            std::fill(den_band.data.begin(), den_band.data.end(), 0.f);
        }

        if (stream_comp_raw) {
            for (const CachedCompMeta& meta : cached_meta) {
                if (!meta.rob_has_nonzero ||
                    !robustness_band_can_contribute(meta.rob_rows_nonzero, y0, bh,
                                                    work.scale, work.bayer_mode))
                    continue;
                if (!load_streamed_comp_raw(meta.index, comp_scratch)) continue;
                merge_comp_band(comp_scratch, meta.flow, meta.covs, meta.rob, tile_size,
                                num_band, den_band, y0, work, meta.index);
            }
        } else {
            for (const CachedCompFrame& fc : cached) {
                if (!fc.rob_has_nonzero ||
                    !robustness_band_can_contribute(fc.rob_rows_nonzero, y0, bh,
                                                    work.scale, work.bayer_mode))
                    continue;
                merge_comp_band(fc.comp, fc.flow, fc.covs, fc.rob, tile_size,
                                num_band, den_band, y0, work, fc.index);
            }
        }

        merge_ref_band(ref, ref_covs, num_band, den_band, y0, work, acc_rob_ptr);
        if (y0 + bh >= Hs)
            accumulate_diag(num_band, den_band, diag);
        if (debug && analyze_merge_band(num_band, den_band, y0, merge_debug)) {
            const std::string ys = std::to_string(y0);
            debug_dump_bin("cpp_num_bad_band_y" + ys,
                           num_band.data.data(), num_band.data.size());
            debug_dump_bin("cpp_den_bad_band_y" + ys,
                           den_band.data.data(), den_band.data.size());
        }

        encode_band_rows(num_band, den_band, y0, bh, work, nch, preview, pscale, ph, pw, Ws, row16);
        writer.write_rows(row16.data(), bh);
        report("Merging output", 0.48f + 0.50f * (float)(y0 + bh) / Hs);
    }
#endif

    writer.close();
    if (work.robustness_save_mask && have_acc_rob) {
        if (write_robustness_mask_pgm(acc_rob, n - 1, dng_path))
            report("Wrote robustness mask", 0.985f);
    }
    cached.clear();
    cached_meta.clear();
    ref = Image();
    ref_covs = CovField();
    if (cache_streamed_comp_raw) fs::remove_all(cache, ec);
    report(format_accum_diag(diag), 0.99f);
    if (debug) {
        append_merge_summary(debug_summary, merge_debug);
        debug_dump_text("cpp_debug_summary", debug_summary.str());
    }
    prof_mark_memory("burst:end");
    if (prof_enabled()) {
        // Wall time reported separately: the buckets above overlap (async
        // decode/encode threads) and must not be summed against it.
        const double wall = prof_now_ms() - t_burst;
        char hdr[192];
        std::snprintf(hdr, sizeof(hdr),
                      "\nburst wall %.1f ms over %d frames (%.1f ms/frame)\n",
                      wall, n, wall / std::max(1, n));
        const std::string prof = prof_report() + hdr;
        std::printf("%s", prof.c_str());
        std::fflush(stdout);
        debug_dump_text("cpp_profile", prof);
    }
    report("Done", 1.f);
    return preview;
}

Image process_burst_paths_to_dng(const std::vector<std::string>& paths, const Config& cfg,
                                 const std::string& dng_path, const ProgressFn& progress,
                                 int maxPreviewDim) {
    return process_burst_loader_to_dng(
        (int)paths.size(),
        [&](int index, Config& work, bool is_reference, int crop_h, int crop_w) {
            if (index < 0 || index >= (int)paths.size()) return Image();
            return load_raw_frame(paths[(size_t)index], work, is_reference, crop_h, crop_w);
        },
        cfg, dng_path, progress, maxPreviewDim);
}

} // namespace hhsr
