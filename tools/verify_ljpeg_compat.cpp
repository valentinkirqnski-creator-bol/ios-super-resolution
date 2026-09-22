// Does our lossless-JPEG stream stay decodable by SOMEONE ELSE'S decoder?
//
//   g++ -std=c++17 -O2 -I core tools/verify_ljpeg_compat.cpp core/ljpeg.cpp
//       -o verify_ljpeg_compat
//
// A round-trip through our own decoder cannot answer that, and this is not
// hypothetical: the encoder used to emit one Huffman table PER COMPONENT (Th =
// 0,1,2 with matching Td in the SOS). That is legal T.81 -- an independent
// decoder written from Annex H reconstructs it exactly -- but no other DNG
// writer does it, so it is the one arrangement real decoders are never exercised
// against. Apple's rejected it: Photos showed the embedded JPEG preview, then
// replaced it with its own render of the main image and cached the failure, so a
// shot went black and stayed black. Compression=1 and Compression=8 were fine,
// which is what isolated the stream.
//
// So this checks the header choices a foreign decoder depends on, not the pixels:
//
//   1. exactly ONE DHT, with Tc = 0 and Th = 0. dcraw-derived decoders fill
//      unset tables by copying the previous one (LibRaw ljpeg_start:
//      "huff[c+1] = huff[c]"), so table 0 is what every component resolves to,
//      and a decoder that ignores Td entirely still lands on the right table.
//   2. every SOS component selects Td = 0, so a decoder that DOES read Td agrees
//      with one that does not.
//   3. Ss = 1 (predictor 1) and Al = 0, the only combination in use here.
//   4. no subsampling: Hi = Vi = 1 for every component.
//   5. SOF3 geometry matches what was handed in, and P = 16.
//   6. the scan is byte-stuffed -- no 0xFF followed by anything but 0x00 -- or a
//      conformant decoder would read a marker mid-scan and stop.
//   7. SOI first, EOI last.
//
// Pixels are still checked through our own decoder at the end, so a change that
// satisfies all of the above but breaks the coding is still caught.
#include "ljpeg.h"

#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;

static void chk(bool ok, const char* what) {
    if (!ok) ++g_fail;
    std::printf("    %-52s %s\n", what, ok ? "ok" : "FAIL");
}

struct Parsed {
    int prec = 0, rows = 0, w = 0, ncomp = 0;
    int ndht = 0;
    int first_dht_byte = -1;
    int predictor = -1, al = -1;
    bool all_td_zero = true;
    bool all_no_subsample = true;
    bool stuffing_ok = true;
    bool saw_eoi = false;
    bool structural_ok = true;
};

// Walks the markers only; it does not decode. Deliberately independent of
// ljpeg_decode so a change to one does not silently validate the other.
static Parsed parse(const std::vector<uint8_t>& d) {
    Parsed p;
    if (d.size() < 4 || d[0] != 0xFF || d[1] != 0xD8) { p.structural_ok = false; return p; }
    size_t i = 2;
    while (i + 4 <= d.size()) {
        if (d[i] != 0xFF) { p.structural_ok = false; return p; }
        const uint8_t m = d[i + 1];
        const size_t seg = ((size_t)d[i + 2] << 8) | d[i + 3];
        if (seg < 2 || i + 2 + seg > d.size()) { p.structural_ok = false; return p; }
        const uint8_t* s = d.data() + i + 4;
        const size_t n = seg - 2;

        if (m == 0xC3) {                                  // SOF3
            if (n < 6) { p.structural_ok = false; return p; }
            p.prec = s[0];
            p.rows = ((int)s[1] << 8) | s[2];
            p.w    = ((int)s[3] << 8) | s[4];
            p.ncomp = s[5];
            for (int c = 0; c < p.ncomp; ++c)
                if (s[6 + 3 * c + 1] != 0x11) p.all_no_subsample = false;
        } else if (m == 0xC4) {                           // DHT
            ++p.ndht;
            if (p.first_dht_byte < 0 && n >= 1) p.first_dht_byte = s[0];
        } else if (m == 0xDA) {                           // SOS
            if (n < 1) { p.structural_ok = false; return p; }
            const int ns = s[0];
            if (n < (size_t)(1 + 2 * ns + 3)) { p.structural_ok = false; return p; }
            for (int c = 0; c < ns; ++c)
                if ((s[1 + 2 * c + 1] >> 4) != 0) p.all_td_zero = false;
            p.predictor = s[1 + 2 * ns];
            p.al = s[3 + 2 * ns] & 15;
            // The entropy-coded scan runs from here to EOI.
            size_t k = i + 2 + seg;
            while (k + 1 < d.size()) {
                if (d[k] == 0xFF) {
                    const uint8_t nx = d[k + 1];
                    if (nx == 0x00) { k += 2; continue; }
                    if (nx == 0xD9) { p.saw_eoi = (k + 2 == d.size()); break; }
                    p.stuffing_ok = false;                // a marker inside the scan
                    break;
                }
                ++k;
            }
            return p;
        }
        i += 2 + seg;
    }
    p.structural_ok = false;
    return p;
}

static void case_(int W, int rows, int nc, const char* note) {
    std::printf("  %s  (%dx%d, %d component%s)\n", note, W, rows, nc, nc == 1 ? "" : "s");
    std::vector<uint16_t> in((size_t)W * rows * nc);
    for (size_t k = 0; k < in.size(); ++k) {
        uint32_t h = (uint32_t)(k * 2654435761u);
        h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
        // Mixed content: smooth ramps plus noise, so the histogram spans many
        // categories and long codes actually occur.
        in[k] = (uint16_t)((k % 5 == 0) ? (h & 0xFFFFu) : ((k * 37u) & 0xFFFFu));
    }

    std::vector<uint8_t> enc;
    if (!hhsr::ljpeg_encode(in.data(), W, rows, nc, enc)) {
        chk(false, "encode succeeded");
        return;
    }
    const Parsed p = parse(enc);
    chk(p.structural_ok,              "markers parse, SOI present");
    chk(p.saw_eoi,                    "EOI is the last two bytes");
    chk(p.ndht == 1,                  "exactly ONE DHT segment");
    chk(p.first_dht_byte == 0,        "that DHT is Tc=0, Th=0");
    chk(p.all_td_zero,                "every SOS component selects Td=0");
    chk(p.predictor == 1,             "Ss = 1 (predictor 1, Ra)");
    chk(p.al == 0,                    "Al = 0 (no point transform)");
    chk(p.all_no_subsample,           "Hi = Vi = 1 for every component");
    chk(p.prec == 16,                 "P = 16");
    chk(p.rows == rows && p.w == W,   "SOF3 geometry matches the input");
    chk(p.ncomp == nc,                "SOF3 Nf matches the input");
    chk(p.stuffing_ok,                "no unescaped 0xFF inside the scan");

    std::vector<uint16_t> out(in.size(), 0xDEAD);
    const bool dec = hhsr::ljpeg_decode(enc.data(), enc.size(), out.data(), W, rows, nc);
    chk(dec, "our decoder accepts it");
    if (dec) chk(std::memcmp(in.data(), out.data(), in.size() * 2) == 0,
                 "pixels survive exactly");
}

int main() {
    std::printf("lossless JPEG: header choices a FOREIGN decoder depends on\n");
    std::printf("(a round-trip through our own decoder cannot check these)\n\n");

    case_(8064, 64, 3, "48MP strip");
    case_(4032, 64, 3, "12MP strip");
    case_(8064, 32, 3, "short final strip");
    case_(100, 20, 1, "single component");
    case_(100, 20, 2, "two components");
    case_(100, 20, 4, "four components");
    case_(1, 1, 3, "1x1");
    case_(1, 64, 3, "single column");
    case_(64, 1, 3, "single row");

    std::printf("\n%d failed\n", g_fail);
    return g_fail ? 1 : 0;
}
