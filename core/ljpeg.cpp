#include "ljpeg.h"
#include <cstring>
#include <algorithm>

namespace hhsr {
namespace {

// Symbols are the JPEG "SSSS" difference categories, 0..16.
constexpr int kNSym = 17;
constexpr int kMaxClen = 32;   // longest code the optimality pass may produce

// SSSS for a difference already reduced to [-32768, 32767].
// 0 means "no difference"; 16 is the reserved code for -32768, which carries no
// extra bits (T.81 H.1.2.2, and what every DNG >= 1.1 decoder expects).
inline int category(int32_t d) {
    if (d == 0) return 0;
    if (d == -32768) return 16;
    uint32_t m = (uint32_t)(d < 0 ? -d : d);
#if defined(__GNUC__) || defined(__clang__)
    return 32 - __builtin_clz(m);           // 1..15
#elif defined(_MSC_VER)
    unsigned long idx;
    _BitScanReverse(&idx, m);
    return (int)idx + 1;
#else
    int s = 0;
    while (m) { ++s; m >>= 1; }
    return s;
#endif
}

// Reduce a - b into [-32768, 32767]. Differences are modulo 2^16 in lossless
// JPEG; the decoder wraps in uint16 when it adds the prediction back, so this
// is exact rather than approximate.
inline int32_t wrap_diff(uint32_t a, uint32_t b) {
    uint32_t u = (a - b) & 0xFFFFu;
    return (u >= 32768u) ? (int32_t)u - 65536 : (int32_t)u;
}

// ---------------------------------------------------------------- Huffman ---

struct HuffEnc {
    uint8_t  bits[17] = {0};   // bits[len] = number of codes of that length
    uint8_t  vals[kNSym] = {0};
    int      nvals = 0;
    uint16_t code[kNSym] = {0};
    uint8_t  size[kNSym] = {0};
};

// T.81 Annex K.2: build the minimum-redundancy code, cap it at 16 bits, and
// keep one code out of circulation so no symbol ever gets the all-ones code
// (which JPEG reserves). Slot kNSym is that reserved symbol and is dropped
// before the table is emitted.
bool build_huff_enc(const uint64_t freq_in[kNSym], HuffEnc& h) {
    uint64_t freq[kNSym + 1];
    int codesize[kNSym + 1] = {0};
    int others[kNSym + 1];
    for (int i = 0; i < kNSym; ++i) freq[i] = freq_in[i];
    freq[kNSym] = 1;                       // the reserved symbol
    for (int i = 0; i <= kNSym; ++i) others[i] = -1;

    for (;;) {
        // Least and second-least non-zero frequency. Scanning upward with <=
        // breaks ties toward the higher index, which is what sinks the reserved
        // symbol to the longest code.
        int v1 = -1, v2 = -1;
        for (int i = 0; i <= kNSym; ++i)
            if (freq[i] && (v1 < 0 || freq[i] <= freq[v1])) v1 = i;
        for (int i = 0; i <= kNSym; ++i)
            if (freq[i] && i != v1 && (v2 < 0 || freq[i] <= freq[v2])) v2 = i;
        if (v2 < 0) break;

        freq[v1] += freq[v2];
        freq[v2] = 0;
        ++codesize[v1];
        while (others[v1] >= 0) { v1 = others[v1]; ++codesize[v1]; }
        others[v1] = v2;
        ++codesize[v2];
        while (others[v2] >= 0) { v2 = others[v2]; ++codesize[v2]; }
    }

    int bitcount[kMaxClen + 2] = {0};
    for (int i = 0; i <= kNSym; ++i) {
        if (!codesize[i]) continue;
        if (codesize[i] > kMaxClen) return false;
        ++bitcount[codesize[i]];
    }

    // Figure K.3: fold everything longer than 16 bits back in, preserving the
    // Kraft sum so the code stays prefix-free and complete.
    for (int i = kMaxClen; i > 16; ) {
        if (bitcount[i] == 0) { --i; continue; }
        int j = i - 1;
        do { --j; } while (j > 0 && bitcount[j] == 0);
        if (j <= 0) return false;
        bitcount[i] -= 2;
        bitcount[i - 1] += 1;
        bitcount[j + 1] += 2;
        bitcount[j] -= 1;
    }

    // Drop the reserved symbol: it always holds a code of the longest length.
    int last = 16;
    while (last > 0 && bitcount[last] == 0) --last;
    if (last == 0) return false;           // no symbols at all
    --bitcount[last];

    // huffval is ordered by the *pre-capping* code length (as libjpeg does):
    // the capping pass moves codes between lengths but never reorders symbols.
    h.nvals = 0;
    for (int len = 1; len <= kMaxClen; ++len)
        for (int sym = 0; sym < kNSym; ++sym)
            if (codesize[sym] == len) h.vals[h.nvals++] = (uint8_t)sym;

    int total = 0;
    for (int len = 1; len <= 16; ++len) {
        h.bits[len] = (uint8_t)bitcount[len];
        total += bitcount[len];
    }
    if (total != h.nvals) return false;

    // Canonical code assignment, shortest length first.
    uint32_t code = 0;
    int k = 0;
    for (int len = 1; len <= 16; ++len) {
        for (int n = 0; n < h.bits[len]; ++n, ++k) {
            h.code[h.vals[k]] = (uint16_t)code;
            h.size[h.vals[k]] = (uint8_t)len;
            ++code;
        }
        code <<= 1;
    }
    return true;
}

// MSB-first bit sink with the 0xFF -> 0xFF 0x00 stuffing JPEG entropy data
// requires, so a payload byte can never be mistaken for a marker.
//
// Writes through a raw cursor rather than push_back: at three samples per pixel
// this is the hottest loop in the encoder, and a capacity check per *byte* was
// costing more than the Huffman lookup. The caller sizes `out` so `p` always
// has kSlack bytes of headroom, and reserve_more() is the only place that can
// invalidate the cursor.
struct BitWriter {
    // Headroom for one whole pixel: ncomp <= 4 samples x (16-bit code + 15-bit
    // value) = 124 bits, plus up to 7 already in the accumulator, is 16 output
    // bytes -- and byte stuffing can double every one of them. 64 leaves room
    // for flush() on top.
    static constexpr size_t kSlack = 64;
    std::vector<uint8_t>& out;   // the caller's scratch buffer
    uint8_t* p = nullptr;
    uint8_t* limit = nullptr;
    uint64_t acc = 0;
    int nbits = 0;

    BitWriter(std::vector<uint8_t>& o, size_t start) : out(o), used_(start) { rebind(); }

    void rebind() {
        p = out.data() + used_;
        limit = out.data() + out.size();
    }
    // Called with the cursor's position already folded back into used_.
    void reserve_more() {
        used_ = (size_t)(p - out.data());
        out.resize(out.size() * 2 + 4096);
        rebind();
    }
    inline void ensure() {
        if ((size_t)(limit - p) < kSlack) reserve_more();
    }

    inline void put(uint32_t v, int n) {
        acc = (acc << n) | (uint64_t)(v & ((1u << n) - 1u));
        nbits += n;
        while (nbits >= 8) {
            nbits -= 8;
            const uint8_t b = (uint8_t)((acc >> nbits) & 0xFFu);
            *p++ = b;
            if (b == 0xFFu) *p++ = 0x00u;   // byte stuffing
        }
    }
    // JPEG pads the final partial byte with 1-bits.
    void flush() {
        ensure();
        if (nbits > 0) {
            const int pad = 8 - nbits;
            const uint8_t b = (uint8_t)(((acc << pad) | ((1u << pad) - 1u)) & 0xFFu);
            *p++ = b;
            if (b == 0xFFu) *p++ = 0x00u;
            nbits = 0;
        }
        acc = 0;
        used_ = (size_t)(p - out.data());
    }
    size_t used() const { return used_; }
    void set_used(size_t u) { used_ = u; rebind(); }

private:
    size_t used_ = 0;
};

struct HuffDec {
    int32_t mincode[17] = {0};
    int32_t maxcode[17];
    int     valptr[17] = {0};
    uint8_t vals[kNSym] = {0};
    int     nvals = 0;
    bool    present = false;
};

bool build_huff_dec(const uint8_t bits[17], const uint8_t* vals, int nvals, HuffDec& d) {
    int total = 0;
    for (int len = 1; len <= 16; ++len) total += bits[len];
    if (total != nvals || nvals <= 0 || nvals > kNSym) return false;
    uint32_t code = 0;
    int k = 0;
    for (int len = 1; len <= 16; ++len) {
        d.valptr[len] = k;
        d.mincode[len] = (int32_t)code;
        if (bits[len]) {
            code += bits[len];
            k += bits[len];
            d.maxcode[len] = (int32_t)code - 1;
        } else {
            d.maxcode[len] = -1;           // no code of this length
        }
        code <<= 1;
    }
    std::memcpy(d.vals, vals, (size_t)nvals);
    d.nvals = nvals;
    d.present = true;
    return true;
}

struct BitReader {
    const uint8_t* p;
    const uint8_t* end;
    uint32_t acc = 0;
    int nbits = 0;
    bool bad = false;

    BitReader(const uint8_t* d, size_t n) : p(d), end(d + n) {}

    inline int bit() {
        if (nbits == 0) {
            if (p >= end) { bad = true; return 0; }
            uint8_t b = *p++;
            if (b == 0xFFu) {
                // Stuffed zero belongs to the data; any other marker ends it.
                if (p < end && *p == 0x00u) ++p;
                else { bad = true; return 0; }
            }
            acc = b;
            nbits = 8;
        }
        --nbits;
        return (int)((acc >> nbits) & 1u);
    }
    inline uint32_t bits(int n) {
        uint32_t v = 0;
        for (int i = 0; i < n; ++i) v = (v << 1) | (uint32_t)bit();
        return v;
    }
};

inline int decode_sym(BitReader& br, const HuffDec& d) {
    int32_t code = br.bit();
    int len = 1;
    while (len <= 16 && (d.maxcode[len] < 0 || code > d.maxcode[len])) {
        code = (code << 1) | br.bit();
        ++len;
    }
    if (len > 16) return -1;
    const int idx = d.valptr[len] + (int)(code - d.mincode[len]);
    if (idx < 0 || idx >= d.nvals) return -1;
    return d.vals[idx];
}

// Header writer over the scratch buffer. `n` always has room: the buffer is
// sized for the whole scan before anything is written into it.
struct HdrWriter {
    uint8_t* p;
    inline void put8(uint8_t v) { *p++ = v; }
    inline void put16(uint32_t v) { *p++ = (uint8_t)((v >> 8) & 0xFFu); *p++ = (uint8_t)(v & 0xFFu); }
};

} // namespace

bool ljpeg_encode(const uint16_t* src, int W, int rows, int ncomp,
                  std::vector<uint8_t>& buf, size_t& out_len) {
    out_len = 0;
    if (!src || W <= 0 || rows <= 0 || ncomp < 1 || ncomp > 4) return false;
    if (W > 65535 || rows > 65535) return false;

    const size_t nsamp = (size_t)rows * (size_t)W * (size_t)ncomp;
    // Category is kept beside the difference rather than recomputed in pass 2:
    // it is needed once for the histogram and once for the code, and the bit
    // scan is not free at three samples per pixel.
    std::vector<int16_t> diff(nsamp);
    std::vector<uint8_t> cat(nsamp);
    uint64_t freq[4][kNSym];
    std::memset(freq, 0, sizeof freq);

    // Prediction, exactly as T.81 H.1.2.1 defines it for predictor 1:
    //   first sample of the frame -> 2^(P-1) = 32768
    //   first sample of a line    -> the sample directly above it
    //   every other sample        -> the sample to its left, same component
    for (int y = 0; y < rows; ++y) {
        const size_t row0 = (size_t)y * (size_t)W * (size_t)ncomp;
        const uint16_t* r = src + row0;
        const uint16_t* up = r - (size_t)W * (size_t)ncomp;
        int16_t* d = diff.data() + row0;
        uint8_t* q = cat.data() + row0;
        for (int c = 0; c < ncomp; ++c) {
            const uint32_t pred = (y == 0) ? 32768u : (uint32_t)up[c];
            const int32_t v = wrap_diff(r[c], pred);
            const int s = category(v);
            d[c] = (int16_t)v;
            q[c] = (uint8_t)s;
            ++freq[c][s];
        }
        for (size_t i = (size_t)ncomp; i < (size_t)W * (size_t)ncomp; i += (size_t)ncomp) {
            for (int c = 0; c < ncomp; ++c) {
                const int32_t v = wrap_diff(r[i + c], r[i + c - ncomp]);
                const int s = category(v);
                d[i + c] = (int16_t)v;
                q[i + c] = (uint8_t)s;
                ++freq[c][s];
            }
        }
    }

    HuffEnc h[4];
    for (int c = 0; c < ncomp; ++c)
        if (!build_huff_enc(freq[c], h[c])) return false;

    // Size once for header + worst realistic scan, then write through a raw
    // cursor. Never shrunk: the caller reuses this buffer for the next strip,
    // so after the first one there is no allocation and no zero-fill at all.
    // ~1.2 bytes/sample on photographic data; 1.5 covers the noisy strips, and
    // BitWriter doubles if a pathological one needs more.
    const size_t need = nsamp + nsamp / 2 + 1024;
    if (buf.size() < need) buf.resize(need);

    HdrWriter hw{buf.data()};
    hw.put8(0xFF); hw.put8(0xD8);                        // SOI

    hw.put8(0xFF); hw.put8(0xC3);                        // SOF3: lossless, Huffman
    hw.put16((uint32_t)(8 + 3 * ncomp));
    hw.put8(16);                                         // P: 16-bit samples
    hw.put16((uint32_t)rows);                            // Y
    hw.put16((uint32_t)W);                               // X
    hw.put8((uint8_t)ncomp);                             // Nf
    for (int c = 0; c < ncomp; ++c) {
        hw.put8((uint8_t)(c + 1));                       // Ci
        hw.put8(0x11);                                   // Hi = Vi = 1
        hw.put8(0);                                      // Tq unused when lossless
    }

    // One table per component: R, G and B have visibly different difference
    // statistics, and dcraw-derived decoders index tables by component number,
    // so component c must use table c.
    for (int c = 0; c < ncomp; ++c) {
        hw.put8(0xFF); hw.put8(0xC4);                    // DHT
        hw.put16((uint32_t)(2 + 1 + 16 + h[c].nvals));
        hw.put8((uint8_t)c);                             // Tc = 0 (lossless), Th = c
        for (int i = 1; i <= 16; ++i) hw.put8(h[c].bits[i]);
        for (int i = 0; i < h[c].nvals; ++i) hw.put8(h[c].vals[i]);
    }

    hw.put8(0xFF); hw.put8(0xDA);                        // SOS
    hw.put16((uint32_t)(6 + 2 * ncomp));
    hw.put8((uint8_t)ncomp);                             // Ns
    for (int c = 0; c < ncomp; ++c) {
        hw.put8((uint8_t)(c + 1));                       // Cs
        hw.put8((uint8_t)(c << 4));                      // Td = c, Ta = 0
    }
    hw.put8(1);                                          // Ss = predictor 1 (Ra)
    hw.put8(0);                                          // Se, unused
    hw.put8(0);                                          // Ah/Al: no point transform

    const size_t scan_start = (size_t)(hw.p - buf.data());
    BitWriter bw(buf, scan_start);
    const int16_t* dp = diff.data();
    const uint8_t* qp = cat.data();
    for (size_t i = 0; i < nsamp; i += (size_t)ncomp) {
        bw.ensure();
        for (int c = 0; c < ncomp; ++c) {
            const int s = qp[i + c];
            const HuffEnc& hc = h[c];
            bw.put(hc.code[s], hc.size[s]);
            if (s > 0 && s < 16) {
                // Positive values code as themselves; negative ones as
                // v + 2^s - 1, which leaves the top bit clear and marks them
                // negative.
                const int32_t v = dp[i + c];
                const uint32_t mag = (uint32_t)((v > 0) ? v : (v + (int32_t)((1u << s) - 1u)));
                bw.put(mag, s);
            }
        }
    }
    bw.flush();
    size_t n = bw.used();
    if (buf.size() < n + 2) buf.resize(n + 2);
    buf[n++] = 0xFF; buf[n++] = 0xD9;                    // EOI
    out_len = n;
    return true;
}

bool ljpeg_encode(const uint16_t* src, int W, int rows, int ncomp,
                  std::vector<uint8_t>& out) {
    size_t n = 0;
    if (!ljpeg_encode(src, W, rows, ncomp, out, n)) return false;
    out.resize(n);
    out.shrink_to_fit();
    return true;
}

bool ljpeg_decode(const uint8_t* data, size_t len, uint16_t* dst,
                  int W, int rows, int ncomp) {
    if (!data || !dst || len < 4 || W <= 0 || rows <= 0 || ncomp < 1 || ncomp > 4)
        return false;
    if (data[0] != 0xFFu || data[1] != 0xD8u) return false;

    HuffDec dec[4];
    int f_prec = 0, f_rows = 0, f_w = 0, f_ncomp = 0, predictor = -1, point_xform = 0;
    int td[4] = {0, 1, 2, 3};
    size_t pos = 2;
    bool saw_sos = false;

    while (pos + 4 <= len) {
        if (data[pos] != 0xFFu) return false;
        const uint8_t marker = data[pos + 1];
        if (marker == 0xD9u) return false;                // EOI before any scan
        const uint32_t seg = ((uint32_t)data[pos + 2] << 8) | data[pos + 3];
        if (seg < 2 || pos + 2 + seg > len) return false;
        const uint8_t* p = data + pos + 4;
        const uint32_t n = seg - 2;

        if (marker == 0xC3u) {                            // SOF3
            if (n < 6) return false;
            f_prec = p[0];
            f_rows = ((int)p[1] << 8) | p[2];
            f_w    = ((int)p[3] << 8) | p[4];
            f_ncomp = p[5];
            if (f_ncomp < 1 || f_ncomp > 4 || n < 6u + 3u * (uint32_t)f_ncomp) return false;
            for (int c = 0; c < f_ncomp; ++c)
                if (p[6 + 3 * c + 1] != 0x11u) return false;   // no subsampling
        } else if (marker == 0xC4u) {                     // DHT
            uint32_t q = 0;
            while (q + 17 <= n) {
                const uint8_t tc_th = p[q];
                const int th = tc_th & 0x0F;
                if ((tc_th >> 4) != 0 || th > 3) return false; // lossless: DC class only
                uint8_t bits[17] = {0};
                int nv = 0;
                for (int i = 1; i <= 16; ++i) { bits[i] = p[q + i]; nv += bits[i]; }
                if (nv <= 0 || nv > kNSym || q + 17u + (uint32_t)nv > n) return false;
                if (!build_huff_dec(bits, p + q + 17, nv, dec[th])) return false;
                q += 17u + (uint32_t)nv;
            }
        } else if (marker == 0xDAu) {                     // SOS
            if (n < 1) return false;
            const int ns = p[0];
            if (ns != f_ncomp || n < 1u + 2u * (uint32_t)ns + 3u) return false;
            for (int c = 0; c < ns; ++c) td[c] = p[1 + 2 * c + 1] >> 4;
            predictor   = p[1 + 2 * ns];
            point_xform = p[1 + 2 * ns + 2] & 0x0F;
            pos = pos + 2 + seg;
            saw_sos = true;
            break;
        } else if (marker == 0xDDu) {                     // DRI
            // Restart intervals reset the predictor mid-strip. Nothing this
            // writer produces uses them, and honouring them silently would
            // decode garbage, so refuse instead of guessing.
            if (n >= 2 && (((uint32_t)p[0] << 8) | p[1]) != 0) return false;
        }
        pos = pos + 2 + seg;
    }

    if (!saw_sos) return false;
    if (predictor != 1 || point_xform != 0 || f_prec != 16) return false;
    if (f_w != W || f_rows != rows || f_ncomp != ncomp) return false;
    for (int c = 0; c < ncomp; ++c)
        if (td[c] < 0 || td[c] > 3 || !dec[td[c]].present) return false;

    BitReader br(data + pos, len - pos);
    for (int y = 0; y < rows; ++y) {
        uint16_t* r = dst + (size_t)y * (size_t)W * (size_t)ncomp;
        const uint16_t* up = r - (size_t)W * (size_t)ncomp;
        for (int x = 0; x < W; ++x) {
            const size_t base = (size_t)x * (size_t)ncomp;
            for (int c = 0; c < ncomp; ++c) {
                const int s = decode_sym(br, dec[td[c]]);
                if (s < 0 || s > 16 || br.bad) return false;
                int32_t v;
                if (s == 0) {
                    v = 0;
                } else if (s == 16) {
                    v = -32768;
                } else {
                    const uint32_t b = br.bits(s);
                    v = (int32_t)b;
                    if ((b & (1u << (s - 1))) == 0) v -= (int32_t)((1u << s) - 1u);
                }
                uint32_t pred;
                if (x > 0)      pred = r[base + c - ncomp];
                else if (y > 0) pred = up[c];
                else            pred = 32768u;
                r[base + c] = (uint16_t)((pred + (uint32_t)v) & 0xFFFFu);
            }
        }
    }
    return !br.bad;
}

} // namespace hhsr
