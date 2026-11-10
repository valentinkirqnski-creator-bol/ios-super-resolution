"""Derive fb_eval.cpp from mann_dataset.cpp: ground-truth evaluation of the forward-backward confidence.
NOSLIP=1 restores the aligner's own flows after the slip injection (natural errors only). Per comparison frame and
mask variant: harmful (ghost z > 4) and harmless detailed (z < 1, |hp ref| > 2 sigma) pixels; for each FB config,
caught = harmful with C < 0.5, false = harmless with C < 0.5; also only where the mask KEPT the pixel (R >= 0.5),
i.e. what the check adds on top of the mask. Appends one line per config to the stats file argv[2]."""
s = open("mann_dataset.cpp", encoding="utf8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:70], s.count(a))
    s = s.replace(a, b)


rep('#include "mann.h"\n', '#include "mann.h"\n#include "fb_confidence.h"\n')
rep('    if (std::getenv("CREF_CONV")) {   // convention check',
    '    const std::vector<FlowField> flows_clean = flows;\n    if (std::getenv("CREF_CONV")) {   // convention check')
rep('    // Ghost size z per raw pixel and frame, for a given set of merged flows.',
    '    if (std::getenv("NOSLIP")) flows = flows_clean;\n    // Ghost size z per raw pixel and frame, for a given set of merged flows.')
rep('''            std::vector<float> F, G;
            const std::vector<float> tdis = cref_flow_disagreement(f);''',
'''            {   // ---- FB evaluation ----
                struct Cfg { const char* name; int sub; bool full; };
                const Cfg cfgs[] = {{"local 1x1", 1, false}, {"local 2x2", 2, false}, {"local 4x4", 4, false},
                                    {"FULL  4x4", 4, true}};
                const float taus[2] = {0.75f, 0.2f};
                for (const Cfg& q : cfgs) {
                    Config cq = cv; cq.fb_conf_subdiv = q.sub; cq.fb_conf_full = q.full;
                    const std::vector<float> e = fb_tile_error(ref_grey, greys[k], f, H, W, ts, cq);
                    const int pnx = f.nx * q.sub; const float ps = (float)ts / q.sub;
                    for (int ti = 0; ti < 2; ++ti) {
                        const float ethr = taus[ti] * 1.17741f;   // C < 0.5  <=>  e > tau sqrt(2 ln 2)
                        double nb = 0, nok = 0, cb = 0, cf = 0, nbk = 0, nokk = 0, cbk = 0, cfk = 0;
                        for (int y = 2; y < H - 2; ++y)
                            for (int x = 2; x < W - 2; ++x) {
                                const float z = Z[k][(size_t)y * W + x];
                                if (z >= 50.f) continue;   // occluded / not visible: skip
                                float bx = 0; for (int i = -1; i <= 1; ++i) for (int j = -1; j <= 1; ++j) bx += ref_grey.at(y + i, x + j);
                                const float hpv = std::fabs(ref_grey.at(y, x) - bx / 9.f);
                                const float nvp = a * std::max(0.f, ref_grey.at(y, x)) + b;
                                const bool det = hpv > 2.f * std::sqrt(nvp);
                                const bool bad = z > 4.f, ok = z < 1.f && det;
                                if (!bad && !ok) continue;
                                const int py = std::min(f.ny * q.sub - 1, (int)(y / ps)), px = std::min(pnx - 1, (int)(x / ps));
                                const bool flag = e[(size_t)py * pnx + px] > ethr;
                                const bool kept = Rg.at(y, x) >= 0.5f;
                                if (bad) { nb++; cb += flag; if (kept) { nbk++; cbk += flag; } }
                                else { nok++; cf += flag; if (kept) { nokk++; cfk += flag; } }
                            }
                        FILE* fs = std::fopen(argv[2], "ab");
                        std::fprintf(fs, "%d %s %.2f %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f\\n", variant, q.name, taus[ti],
                                     nb, cb, nok, cf, nbk, cbk, nokk, cfk);
                        std::fclose(fs);
                    }
                }
                continue;
            }
            std::vector<float> F, G;
            const std::vector<float> tdis = cref_flow_disagreement(f);''')
rep('    FILE* fo = std::fopen(argv[2], "wb");\n    if (!fo) return 4;', '    FILE* fo = nullptr;')
rep('    std::fclose(fo);\n', '')
open("fb_eval.cpp", "w", encoding="utf8").write(s)
print("written")
