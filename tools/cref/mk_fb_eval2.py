"""fb_eval2.cpp: like fb_eval (mk_fb_eval.py) but compares FB 4x4 (across / +along), burst consensus (across /
+along) and FB+consensus combined, on identical pixels with ground-truth ghost labels. NOSLIP=1 -> natural errors."""
import re
s = open("fb_eval.cpp", encoding="utf8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('#include "fb_confidence.h"\n', '#include "fb_confidence.h"\n#include "burst_consensus.h"\n')
rep("        RefStats rs = init_robustness(ref, cv);\n", """        RefStats rs = init_robustness(ref, cv);
        // Burst consensus over all comparison frames of this variant (times = frame index, ref = 0).
        std::vector<std::vector<float>> cons_w, cons_a;
        {
            std::vector<std::vector<float>> fl; std::vector<int> tm;
            for (int k = 1; k < N; ++k) { fl.push_back(fv[k].flow); tm.push_back(k); }
            const std::vector<float> tsd = consensus_tile_structure(ref_grey, H, W, fv[1].ny, fv[1].nx, ts, cv);
            Config ca = cv; ca.fb_conf_along_edges = false;
            cons_w = burst_consensus_error(fl, tm, 0, fv[1].ny, fv[1].nx, ts, tsd, ca);
            ca.fb_conf_along_edges = true;
            cons_a = burst_consensus_error(fl, tm, 0, fv[1].ny, fv[1].nx, ts, tsd, ca);
        }
""")
a = s[s.index("                struct Cfg { const char* name; int sub; bool full; };"):s.index("                        FILE* fs = std::fopen(argv[2], \"ab\");")]
b = """                // mode: 0 FB, 1 consensus, 2 FB + consensus (errors combined in quadrature = product of C's)
                struct Cfg { const char* name; int mode; bool along; };
                const Cfg cfgs[] = {{"FB4x4", 0, false}, {"FB4x4+along", 0, true}, {"CONS", 1, false},
                                    {"CONS+along", 1, true}, {"FB+CONS", 2, false}};
                const float taus[2] = {0.75f, 0.2f};
                Config c4 = cv; c4.fb_conf_subdiv = 4; c4.fb_conf_along_edges = false;
                const std::vector<float> efb_w = fb_tile_error(ref_grey, greys[k], f, H, W, ts, c4);
                c4.fb_conf_along_edges = true;
                const std::vector<float> efb_a = fb_tile_error(ref_grey, greys[k], f, H, W, ts, c4);
                const std::vector<float>& ecw = cons_w[(size_t)(k - 1)];
                const std::vector<float>& eca = cons_a[(size_t)(k - 1)];
                for (const Cfg& q : cfgs) {
                    const std::vector<float>& efb = q.along ? efb_a : efb_w;
                    const std::vector<float>& ec = q.along ? eca : ecw;
                    const int pnx = f.nx * 4; const float ps = (float)ts / 4;
                    for (int ti = 0; ti < 2; ++ti) {
                        const float ethr = taus[ti] * 1.17741f;   // C < 0.5  <=>  e > tau sqrt(2 ln 2)
                        double nb = 0, nok = 0, cb = 0, cf = 0, nbk = 0, nokk = 0, cbk = 0, cfk = 0;
                        for (int y = 2; y < H - 2; ++y)
                            for (int x = 2; x < W - 2; ++x) {
                                const float z = Z[k][(size_t)y * W + x];
                                if (z >= 50.f) continue;
                                float bx = 0; for (int i = -1; i <= 1; ++i) for (int j = -1; j <= 1; ++j) bx += ref_grey.at(y + i, x + j);
                                const float hpv = std::fabs(ref_grey.at(y, x) - bx / 9.f);
                                const float nvp = a * std::max(0.f, ref_grey.at(y, x)) + b;
                                const bool det = hpv > 2.f * std::sqrt(nvp);
                                const bool bad = z > 4.f, ok = z < 1.f && det;
                                if (!bad && !ok) continue;
                                const int py = std::min(f.ny * 4 - 1, (int)(y / ps)), px = std::min(pnx - 1, (int)(x / ps));
                                const int ty = std::min(f.ny - 1, y / ts), tx = std::min(f.nx - 1, x / ts);
                                const float e_fb = efb.empty() ? 0.f : efb[(size_t)py * pnx + px];
                                const float e_c = ec.empty() ? 0.f : ec[(size_t)ty * f.nx + tx];
                                const float e = q.mode == 0 ? e_fb : (q.mode == 1 ? e_c : std::sqrt(e_fb * e_fb + e_c * e_c));
                                const bool flag = e > ethr;
                                const bool kept = Rg.at(y, x) >= 0.5f;
                                if (bad) { nb++; cb += flag; if (kept) { nbk++; cbk += flag; } }
                                else { nok++; cf += flag; if (kept) { nokk++; cfk += flag; } }
                            }
"""
s = s.replace(a, b)
open("fb_eval2.cpp", "w", encoding="utf8").write(s)
print("written")
