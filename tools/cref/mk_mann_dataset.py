"""Derive mann_dataset.cpp from cref_dataset.cpp (edge slips + mann features + gate plane)."""
s = open("cref_dataset.cpp", encoding="utf8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('// Consensus-refiner dataset harness (built against the app core, core/cref.h features).',
    '// Misalignment-artifact NN dataset harness (core/mann.h features), derived from the cref tool.\n'
    '// Adds EDGE slips (aperture problem): along-edge slides, across-edge shifts (doubled edges) and\n'
    '// ramped offsets along runs of edge tiles (tile stairs). Crop planes: [kMannIn] features, z, gate.')
rep('#include "cref.h"\n', '#include "cref.h"\n#include "mann.h"\n')
rep('    // Ghost size z per raw pixel and frame, for a given set of merged flows.', r'''    // EDGE slips (mann): tiles dominated by one long edge (structure tensor coherence > 0.6, gradient energy
    // > 16x noise). Per frame with p 0.6: each edge tile with p 0.2 slides ALONG its edge by 0.5-4 px (harmless
    // on a straight edge) plus, with p 0.7, ACROSS it by 0.2-1.5 px (log-uniform; a shifted / doubled edge).
    // Per frame with p 0.5: tile stairs -- runs of 3-7 edge tiles walked along the edge direction get a ramp of
    // across-edge offsets (step 0.15-0.7 px per tile, centred), as a slight roll aligned per tile produces.
    size_t n_edge_tiles = 0, n_edge_slips = 0, n_stairs = 0;
    {
        std::vector<float> ex((size_t)ny * nx, 0.f), ey((size_t)ny * nx, 0.f);
        std::vector<uint8_t> is_edge((size_t)ny * nx, 0);
        for (int t = 0; t < ny * nx; ++t) {
            const int ty = t / nx, tx = t % nx;
            double jxx = 0, jyy = 0, jxy = 0, mean = 0; int n = 0;
            for (int y = ty * ts; y < std::min(H - 1, (ty + 1) * ts); ++y)
                for (int x = tx * ts; x < std::min(W - 1, (tx + 1) * ts); ++x) {
                    const double gx = ref_grey.at(y, x + 1) - ref_grey.at(y, x), gy = ref_grey.at(y + 1, x) - ref_grey.at(y, x);
                    jxx += gx * gx; jyy += gy * gy; jxy += gx * gy; mean += ref_grey.at(y, x); ++n;
                }
            if (!n) continue;
            jxx /= n; jyy /= n; jxy /= n; mean /= n;
            const double tr = jxx + jyy, dt = std::sqrt(std::max(0.0, (jxx - jyy) * (jxx - jyy) + 4 * jxy * jxy));
            const double l1 = 0.5 * (tr + dt), l2 = 0.5 * (tr - dt);
            const double nv = hf[0] * std::max(0.0, mean) + hf[1];
            if (tr < 8.0 * 2.0 * nv || (l1 - l2) / std::max(1e-12, l1 + l2) < 0.6) continue;
            // normal = eigenvector of l1; edge direction = perpendicular
            double nxv = jxy, nyv = l1 - jxx;
            if (std::fabs(nxv) + std::fabs(nyv) < 1e-12) { nxv = 1; nyv = 0; if (jyy > jxx) { nxv = 0; nyv = 1; } }
            const double nn = std::hypot(nxv, nyv);
            ex[t] = (float)(-nyv / nn); ey[t] = (float)(nxv / nn);   // along the edge
            is_edge[t] = 1; ++n_edge_tiles;
        }
        auto logu = [&](float lo, float hi) { return std::exp(std::log(lo) + U(rng) * (std::log(hi) - std::log(lo))); };
        for (int k = 1; k < N; ++k) {
            FlowField& f = flows[k];
            if (U(rng) < 0.6f)
                for (int t = 0; t < ny * nx; ++t) {
                    if (!is_edge[t] || U(rng) >= 0.2f) continue;
                    const float a = (U(rng) < 0.5f ? -1.f : 1.f) * (0.5f + 3.5f * U(rng));
                    const float b = U(rng) < 0.7f ? (U(rng) < 0.5f ? -1.f : 1.f) * logu(0.2f, 1.5f) : 0.f;
                    // along = (ex, ey); across = the normal (ey, -ex)
                    f.flow[2 * t] += a * ex[t] + b * ey[t];
                    f.flow[2 * t + 1] += a * ey[t] - b * ex[t];
                    ++n_edge_slips;
                }
            if (U(rng) < 0.5f && n_edge_tiles) {
                const int runs = 1 + (int)(U(rng) * std::max<size_t>(1, n_edge_tiles / 12));
                for (int rr = 0; rr < runs; ++rr) {
                    int t = (int)(U(rng) * ny * nx) % (ny * nx);
                    for (int tries = 0; tries < 50 && !is_edge[t]; ++tries) t = (int)(U(rng) * ny * nx) % (ny * nx);
                    if (!is_edge[t]) break;
                    const int len = 3 + (int)(U(rng) * 5);
                    std::vector<int> run{t};
                    float py = (float)(t / nx), px = (float)(t % nx);
                    for (int s2 = 1; s2 < len; ++s2) {
                        py += ey[t]; px += ex[t];
                        const int qy = (int)std::lround(py), qx = (int)std::lround(px);
                        if (qy < 0 || qy >= ny || qx < 0 || qx >= nx) break;
                        const int q = qy * nx + qx;
                        if (q != run.back()) run.push_back(q);
                    }
                    if (run.size() < 3) continue;
                    const float step = (U(rng) < 0.5f ? -1.f : 1.f) * (0.15f + 0.55f * U(rng));
                    const float c0 = 0.5f * (float)(run.size() - 1);
                    for (size_t i = 0; i < run.size(); ++i) {
                        const int q = run[i];
                        const float b = step * ((float)i - c0);
                        f.flow[2 * q] += b * ey[t];
                        f.flow[2 * q + 1] += -b * ex[t];
                    }
                    ++n_stairs;
                }
            }
        }
    }

    // Ghost size z per raw pixel and frame, for a given set of merged flows.''')
rep('    std::vector<uint16_t> buf((size_t)(kCrefIn + 1) * S * S);',
    '    constexpr int kPl = kMannIn + 2;   // features, z, gate\n    std::vector<uint16_t> buf((size_t)kPl * S * S);')
rep('''        std::vector<Image> aligned(N);
        Image cons(H, W, 1);
        for (int k = 1; k < N; ++k) {
            aligned[k] = warp_grey(greys[k], fv[k], ts);
            for (size_t i = 0; i < cons.data.size(); ++i) cons.data[i] += aligned[k].data[i];
        }
        for (float& v : cons.data) v /= (float)(N - 1);''',
    '''        std::vector<Image> aligned(N);
        for (int k = 1; k < N; ++k) aligned[k] = warp_grey(greys[k], fv[k], ts);''')
rep('pattern tiles %zu period slips %zu\\n",\n                     argv[1], variant, N, W, H, ts, err_sum / std::max<size_t>(1, err_n), n_pattern_tiles, n_period_slips);',
    'pattern tiles %zu period slips %zu edge tiles %zu edge slips %zu stairs %zu\\n",\n'
    '                     argv[1], variant, N, W, H, ts, err_sum / std::max<size_t>(1, err_n), n_pattern_tiles, n_period_slips,\n'
    '                     n_edge_tiles, n_edge_slips, n_stairs);')
rep('''            std::vector<float> F;
            const std::vector<float> tdis = cref_flow_disagreement(f);
            cref_features(ref_grey, aligned[k], cons, N - 1, Rg, tdis, f.ny, f.nx, ts, cv, 0, H, 0, F);
            const size_t plane = (size_t)H * W;
            // Crops: half uniform, half centred on harmful (z > 2), detailed or pattern pixels.
            std::vector<size_t> hot;
            for (size_t i = 0; i < plane; i += 7)
                if (Z[k][i] > 2.f || std::fabs(F[4 * plane + i]) > 0.25f || F[7 * plane + i] > 0.5f) hot.push_back(i);
            for (int c = 0; c < per_frame; ++c) {
                int y0, x0;
                if ((c & 1) && !hot.empty()) {''',
    '''            std::vector<float> F, G;
            const std::vector<float> tdis = cref_flow_disagreement(f);
            const std::vector<float> M = mann_tile_irregularity(f);
            mann_features(ref_grey, aligned[k], Rg, f, M, tdis, ts, cv, 0, H, 0, W, F, G);
            const size_t plane = (size_t)H * W;
            // Crops: 1/4 uniform, 3/4 centred on gated pixels (g > 0.5), preferring harmful or edge ones.
            std::vector<size_t> hot, gated;
            for (size_t i = 0; i < plane; i += 7)
                if (G[i] > 0.5f) {
                    gated.push_back(i);
                    if (Z[k][i] > 2.f || std::fabs(F[2 * plane + i]) > 0.25f) hot.push_back(i);
                }
            if (hot.empty()) hot = gated;
            for (int c = 0; c < per_frame; ++c) {
                int y0, x0;
                if ((c & 3) && !hot.empty()) {''')
rep('''                for (int ch = 0; ch <= kCrefIn; ++ch)
                    for (int y = 0; y < S; ++y)
                        for (int x = 0; x < S; ++x) {
                            const size_t q = (size_t)(y0 + y) * W + (x0 + x);
                            const float v = ch < kCrefIn ? F[(size_t)ch * plane + q] : Z[k][q];''',
    '''                for (int ch = 0; ch < kPl; ++ch)
                    for (int y = 0; y < S; ++y)
                        for (int x = 0; x < S; ++x) {
                            const size_t q = (size_t)(y0 + y) * W + (x0 + x);
                            const float v = ch < kMannIn ? F[(size_t)ch * plane + q] : (ch == kMannIn ? Z[k][q] : G[q]);''')
open("mann_dataset.cpp", "w", encoding="utf8").write(s)
print("written")
