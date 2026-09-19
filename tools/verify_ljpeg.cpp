// Round-trip and bitstream conformance check for core/ljpeg.cpp -- the lossless
// JPEG (ITU-T T.81 Annex H, SOF3) encoder the DNG writer uses for Compression=7.
//
//   g++ -std=c++17 -O2 -I core tools/verify_ljpeg.cpp core/ljpeg.cpp -o verify_ljpeg
//   ./verify_ljpeg            (exit 0 iff every case round-trips bit-exactly)
//
// Two things are checked per case: that decode(encode(x)) == x exactly, and that
// the bitstream is well formed -- SOI, a SOF3 whose precision/width/rows/ncomp
// match what was asked for, an SOS declaring predictor 1, entropy data in which
// the only 0xFF is a stuffed 0xFF 0x00, and a terminating EOI with nothing after
// it. A decoder of our own agreeing with an encoder of our own would not catch a
// misunderstanding they share, which is what the conformance half is for.

#include "ljpeg.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <string>

using namespace hhsr;

static int g_fail = 0;
static int g_pass = 0;

static void check(bool ok, const std::string& what) {
    if (ok) { ++g_pass; }
    else { ++g_fail; std::printf("  FAIL: %s\n", what.c_str()); }
}

// Every 0xFF in entropy-coded data must be followed by 0x00 (stuffing) or be a
// marker in the header/EOI. Verify structure: SOI, markers, then scan data where
// the only legal 0xFF xx is 0xFF 0x00, terminated by 0xFF 0xD9.
static bool check_bitstream(const std::vector<uint8_t>& s, size_t len,
                            int expect_prec, int expect_ncomp,
                            int expect_w, int expect_rows, std::string& err) {
    if (len < 4) { err = "too short"; return false; }
    if (s[0] != 0xFF || s[1] != 0xD8) { err = "no SOI"; return false; }
    size_t pos = 2;
    bool saw_sof = false, saw_sos = false;
    while (pos + 1 < len) {
        if (s[pos] != 0xFF) { err = "expected marker at " + std::to_string(pos); return false; }
        uint8_t m = s[pos + 1];
        pos += 2;
        if (m == 0xD9) { err = "EOI before SOS"; return false; }
        if (pos + 1 >= len) { err = "truncated segment"; return false; }
        size_t seglen = ((size_t)s[pos] << 8) | s[pos + 1];
        if (seglen < 2 || pos + seglen > len) { err = "bad segment length"; return false; }
        if (m == 0xC3) {                       // SOF3
            saw_sof = true;
            int prec = s[pos + 2];
            int rows = (s[pos + 3] << 8) | s[pos + 4];
            int w    = (s[pos + 5] << 8) | s[pos + 6];
            int nc   = s[pos + 7];
            if (prec != expect_prec) { err = "SOF3 precision " + std::to_string(prec); return false; }
            if (nc != expect_ncomp) { err = "SOF3 ncomp " + std::to_string(nc); return false; }
            if (w != expect_w) { err = "SOF3 width " + std::to_string(w); return false; }
            if (rows != expect_rows) { err = "SOF3 rows " + std::to_string(rows); return false; }
        } else if (m == 0xDA) {                // SOS -> entropy data follows
            saw_sos = true;
            int ns = s[pos + 2];
            int predictor = s[pos + 3 + 2 * ns];
            if (predictor != 1) { err = "predictor " + std::to_string(predictor); return false; }
            pos += seglen;
            break;
        }
        pos += seglen;
    }
    if (!saw_sof || !saw_sos) { err = "missing SOF3/SOS"; return false; }
    // Scan: only 0xFF 0x00 permitted until the final 0xFF 0xD9.
    while (pos + 1 < len) {
        if (s[pos] == 0xFF) {
            uint8_t nxt = s[pos + 1];
            if (nxt == 0x00) { pos += 2; continue; }
            if (nxt == 0xD9) {
                if (pos + 2 != len) { err = "trailing bytes after EOI"; return false; }
                return true;
            }
            err = "unstuffed 0xFF" + std::to_string(nxt) + " at " + std::to_string(pos);
            return false;
        }
        ++pos;
    }
    err = "no EOI";
    return false;
}

static void roundtrip(const char* name, const std::vector<uint16_t>& img,
                      int W, int rows, int ncomp) {
    std::vector<uint8_t> buf;
    size_t len = 0;
    bool enc = ljpeg_encode(img.data(), W, rows, ncomp, buf, len);
    check(enc, std::string(name) + ": encode returned true");
    if (!enc) return;

    std::string err;
    bool form = check_bitstream(buf, len, 16, ncomp, W, rows, err);
    check(form, std::string(name) + ": bitstream conformance (" + err + ")");

    std::vector<uint16_t> back(img.size(), 0xDEAD);
    bool dec = ljpeg_decode(buf.data(), len, back.data(), W, rows, ncomp);
    check(dec, std::string(name) + ": decode returned true");
    if (!dec) return;

    size_t bad = 0, first_bad = 0;
    for (size_t i = 0; i < img.size(); ++i) {
        if (img[i] != back[i]) { if (!bad) first_bad = i; ++bad; }
    }
    if (bad) {
        std::printf("  FAIL: %s: %zu/%zu samples differ, first at %zu (%u vs %u)\n",
                    name, bad, img.size(), first_bad, img[first_bad], back[first_bad]);
        ++g_fail;
    } else {
        ++g_pass;
    }
    const double raw = (double)img.size() * 2.0;
    std::printf("  %-34s %7d x %4d x %d  %9.0f -> %9zu bytes (%.2fx)\n",
                name, W, rows, ncomp, raw, len, raw / (double)len);
}

int main() {
    std::mt19937 rng(12345);

    // 1. Smooth gradient -- the case real image data resembles; must compress.
    {
        int W = 640, rows = 64, nc = 3;
        std::vector<uint16_t> img((size_t)W * rows * nc);
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < nc; ++c)
                    img[((size_t)y * W + x) * nc + c] =
                        (uint16_t)((x * 37 + y * 11 + c * 5) & 0xFFFF);
        roundtrip("smooth gradient", img, W, rows, nc);
    }

    // 2. Uniform random -- incompressible; exercises every magnitude category
    //    and produces plenty of 0xFF bytes to stuff.
    {
        int W = 333, rows = 17, nc = 3;
        std::vector<uint16_t> img((size_t)W * rows * nc);
        std::uniform_int_distribution<int> d(0, 65535);
        for (auto& v : img) v = (uint16_t)d(rng);
        roundtrip("uniform random", img, W, rows, nc);
    }

    // 3. Extremes only -- 0 and 0xFFFF alternating: maximum-magnitude diffs.
    {
        int W = 128, rows = 8, nc = 3;
        std::vector<uint16_t> img((size_t)W * rows * nc);
        for (size_t i = 0; i < img.size(); ++i) img[i] = (i & 1) ? 0xFFFF : 0x0000;
        roundtrip("alternating 0 / 65535", img, W, rows, nc);
    }

    // 4. All zeros and all 0xFFFF -- degenerate, and 0xFFFF stresses stuffing.
    {
        int W = 100, rows = 5, nc = 3;
        std::vector<uint16_t> z((size_t)W * rows * nc, 0);
        roundtrip("all zero", z, W, rows, nc);
        std::vector<uint16_t> f((size_t)W * rows * nc, 0xFFFF);
        roundtrip("all 65535", f, W, rows, nc);
    }

    // 5. Component counts 1..4.
    for (int nc = 1; nc <= 4; ++nc) {
        int W = 61, rows = 7;
        std::vector<uint16_t> img((size_t)W * rows * nc);
        std::uniform_int_distribution<int> d(0, 65535);
        for (auto& v : img) v = (uint16_t)d(rng);
        roundtrip((std::string("ncomp=") + std::to_string(nc)).c_str(), img, W, rows, nc);
    }

    // 6. Degenerate geometry -- single pixel, single row, single column.
    {
        std::vector<uint16_t> one(3, 0x1234);
        roundtrip("1x1x3", one, 1, 1, 3);
        int W = 257;
        std::vector<uint16_t> row((size_t)W * 3);
        std::uniform_int_distribution<int> d(0, 65535);
        for (auto& v : row) v = (uint16_t)d(rng);
        roundtrip("257x1x3 (single row)", row, W, 1, 3);
        int R = 129;
        std::vector<uint16_t> col((size_t)R * 3);
        for (auto& v : col) v = (uint16_t)d(rng);
        roundtrip("1x129x3 (single column)", col, 1, R, 3);
    }

    // 7. A realistic 48MP-shaped strip: one band of the real output geometry.
    {
        int W = 8064, rows = 8, nc = 3;
        std::vector<uint16_t> img((size_t)W * rows * nc);
        // Bayer-ish smooth content with noise, the mix a real merge produces.
        std::normal_distribution<double> n(0.0, 120.0);
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < nc; ++c) {
                    double base = 20000.0 + 8000.0 * (double)x / W + 3000.0 * c;
                    double v = base + n(rng);
                    if (v < 0) v = 0;
                    if (v > 65535) v = 65535;
                    img[((size_t)y * W + x) * nc + c] = (uint16_t)(v + 0.5);
                }
        roundtrip("48MP-shaped strip (noisy)", img, W, rows, nc);
    }

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
