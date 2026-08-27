#include "lj92_enc.h"
#include <cstring>
#include <algorithm>

namespace hhsr {

namespace {

// ---------------------------------------------------------------- bit sink
struct BitSink {
    std::vector<uint8_t>& out;
    uint32_t acc = 0;
    int nbits = 0;
    explicit BitSink(std::vector<uint8_t>& o) : out(o) {}
    inline void put(uint32_t bits, int n) {
        // n <= 16; accumulate MSB-first.
        acc = (acc << n) | (bits & ((n == 32) ? 0xFFFFFFFFu : ((1u << n) - 1u)));
        nbits += n;
        while (nbits >= 8) {
            uint8_t b = (uint8_t)(acc >> (nbits - 8));
            out.push_back(b);
            if (b == 0xFF) out.push_back(0x00);  // byte stuffing
            nbits -= 8;
        }
    }
    inline void flush() {
        if (nbits > 0) {
            // Pad the final partial byte with 1s, per T.81 F.1.2.3.
            uint8_t b = (uint8_t)((acc << (8 - nbits)) | ((1u << (8 - nbits)) - 1u));
            out.push_back(b);
            if (b == 0xFF) out.push_back(0x00);
            nbits = 0;
        }
        acc = 0;
    }
};

// SSSS category of a difference in [-32767, 32768]. 32768 maps to 16 (no
// extra bits). Everything else: number of bits of |d|.
inline int ssss_of(int32_t d) {
    if (d == 32768) return 16;
    uint32_t a = (uint32_t)(d < 0 ? -d : d);
    int n = 0;
    while (a) { ++n; a >>= 1; }
    return n;
}

// ---------------------------------------------------- optimal Huffman table
// jpeg_gen_optimal_table's algorithm (libjpeg): merge the two least-frequent
// "trees" repeatedly, tracking code lengths via the codesize/others arrays.
// A pseudo-symbol with count 1 reserves the all-ones code word, as the spec
// convention requires; lengths are then limited to 16 by node adoption.
struct HuffTable {
    uint8_t bits[17] = {0};    // bits[k] = #codes of length k (1-indexed)
    uint8_t huffval[17] = {0}; // symbols in canonical order (max 17 real)
    int nsyms = 0;
    uint16_t code[17] = {0};   // per SYMBOL (ssss 0..16): code word
    uint8_t len[17] = {0};     // per SYMBOL: code length
};

void build_huffman(const uint64_t counts_in[17], HuffTable& t) {
    // 18 slots: 17 real symbols + the reserved pseudo-symbol (index 17).
    int64_t freq[18];
    int codesize[18];
    int others[18];
    for (int i = 0; i < 18; ++i) {
        freq[i] = (i < 17) ? (int64_t)counts_in[i] : 1;  // pseudo gets 1
        codesize[i] = 0;
        others[i] = -1;
    }

    for (;;) {
        // Two smallest nonzero frequencies; ties prefer the LARGER index so
        // the pseudo-symbol sinks to the deepest leaf (the all-ones code).
        int c1 = -1, c2 = -1;
        int64_t v = INT64_MAX;
        for (int i = 0; i <= 17; ++i)
            if (freq[i] && freq[i] <= v) { v = freq[i]; c1 = i; }
        v = INT64_MAX;
        for (int i = 0; i <= 17; ++i)
            if (freq[i] && freq[i] <= v && i != c1) { v = freq[i]; c2 = i; }
        if (c2 < 0) break;  // one tree left -- done

        freq[c1] += freq[c2];
        freq[c2] = 0;
        codesize[c1]++;
        while (others[c1] >= 0) { c1 = others[c1]; codesize[c1]++; }
        others[c1] = c2;
        codesize[c2]++;
        while (others[c2] >= 0) { c2 = others[c2]; codesize[c2]++; }
    }

    int bits[34] = {0};  // enough headroom for intermediate lengths
    for (int i = 0; i <= 17; ++i)
        if (codesize[i]) bits[std::min(codesize[i], 33)]++;

    // Length-limit to 16 (libjpeg adjust_bits): move a pair of leaves up.
    for (int i = 33; i > 16; --i) {
        while (bits[i] > 0) {
            int j = i - 2;
            while (bits[j] == 0) --j;
            bits[i] -= 2;
            bits[i - 1]++;
            bits[j + 1] += 2;
            bits[j]--;
        }
    }
    // Drop the largest code (the pseudo-symbol's slot).
    int i = 16;
    while (i > 0 && bits[i] == 0) --i;
    if (i > 0) bits[i]--;

    t.nsyms = 0;
    for (int k = 1; k <= 16; ++k) t.bits[k] = (uint8_t)bits[k];
    // Symbols sorted by (codesize, symbol value) -- canonical order.
    for (int size = 1; size <= 32; ++size)
        for (int s = 0; s < 17; ++s)
            if (codesize[s] == size) t.huffval[t.nsyms++] = (uint8_t)s;

    // Canonical code assignment.
    uint16_t codeacc = 0;
    int k = 0;
    for (int size2 = 1; size2 <= 16; ++size2) {
        for (int n = 0; n < t.bits[size2]; ++n) {
            const uint8_t sym = t.huffval[k++];
            t.code[sym] = codeacc;
            t.len[sym] = (uint8_t)size2;
            ++codeacc;
        }
        codeacc <<= 1;
    }
}

inline void put_marker(std::vector<uint8_t>& out, uint8_t m) {
    out.push_back(0xFF);
    out.push_back(m);
}

} // namespace

bool lj92_encode_strip(const uint16_t* rows, int w, int h, int ncomp,
                       std::vector<uint8_t>& out) {
    if (!rows || w <= 0 || h <= 0 || ncomp < 1 || ncomp > 3) return false;

    // Pass 1: differences (cached for pass 2) + SSSS histogram, one shared
    // table for all components. Diffs stored in scan order (pixel-interleaved
    // by component), so pass 2 is a straight walk.
    uint64_t counts[17] = {0};
    std::vector<int32_t> diffs((size_t)w * (size_t)h * (size_t)ncomp);
    {
        int32_t predc[3];
        for (int y = 0; y < h; ++y) {
            const uint16_t* row = rows + (size_t)y * (size_t)w * (size_t)ncomp;
            const uint16_t* prev = (y > 0)
                ? rows + (size_t)(y - 1) * (size_t)w * (size_t)ncomp : nullptr;
            int32_t* drow = diffs.data() + (size_t)y * (size_t)w * (size_t)ncomp;
            for (int c = 0; c < ncomp; ++c)
                predc[c] = prev ? (int32_t)prev[c] : 32768;
            for (int x = 0; x < w; ++x) {
                for (int c = 0; c < ncomp; ++c) {
                    const int32_t s = (int32_t)row[(size_t)x * ncomp + c];
                    int32_t d = s - predc[c];
                    // 16-bit modulo arithmetic, mapped to [-32767, 32768].
                    d &= 0xFFFF;
                    if (d > 32768) d -= 65536;
                    counts[ssss_of(d)]++;
                    drow[(size_t)x * ncomp + c] = d;
                    predc[c] = s;
                }
            }
        }
    }

    HuffTable t;
    build_huffman(counts, t);

    // ---- headers
    put_marker(out, 0xD8);  // SOI

    put_marker(out, 0xC4);  // DHT
    {
        int nv = 0;
        for (int k = 1; k <= 16; ++k) nv += t.bits[k];
        const int L = 2 + 1 + 16 + nv;
        out.push_back((uint8_t)(L >> 8));
        out.push_back((uint8_t)(L & 0xFF));
        out.push_back(0x00);  // class 0 (DC/lossless), table 0
        for (int k = 1; k <= 16; ++k) out.push_back(t.bits[k]);
        for (int k = 0; k < nv; ++k) out.push_back(t.huffval[k]);
    }

    put_marker(out, 0xC3);  // SOF3 (lossless, Huffman)
    {
        const int L = 8 + 3 * ncomp;
        out.push_back((uint8_t)(L >> 8));
        out.push_back((uint8_t)(L & 0xFF));
        out.push_back(16);  // precision
        out.push_back((uint8_t)(h >> 8));
        out.push_back((uint8_t)(h & 0xFF));
        out.push_back((uint8_t)(w >> 8));
        out.push_back((uint8_t)(w & 0xFF));
        out.push_back((uint8_t)ncomp);
        for (int c = 0; c < ncomp; ++c) {
            out.push_back((uint8_t)c);  // component id
            out.push_back(0x11);        // H=V=1
            out.push_back(0x00);        // Tq (unused in lossless)
        }
    }

    put_marker(out, 0xDA);  // SOS
    {
        const int L = 6 + 2 * ncomp;
        out.push_back((uint8_t)(L >> 8));
        out.push_back((uint8_t)(L & 0xFF));
        out.push_back((uint8_t)ncomp);
        for (int c = 0; c < ncomp; ++c) {
            out.push_back((uint8_t)c);
            out.push_back(0x00);  // DC table 0
        }
        out.push_back(0x01);  // Ss: predictor 1 (Ra)
        out.push_back(0x00);  // Se
        out.push_back(0x00);  // Ah/Al: no point transform
    }

    // ---- entropy-coded data. MCU = one sample per component, interleaved;
    // diffs already in scan order from pass 1.
    {
        BitSink bs(out);
        out.reserve(out.size() + diffs.size());  // ~1 byte/sample typical
        for (size_t i = 0; i < diffs.size(); ++i) {
            const int32_t d = diffs[i];
            const int ss = ssss_of(d);
            bs.put(t.code[ss], t.len[ss]);
            if (ss > 0 && ss < 16) {
                int32_t v = d;
                if (v < 0) v += (1 << ss) - 1;
                bs.put((uint32_t)v, ss);
            }
        }
        bs.flush();
    }

    put_marker(out, 0xD9);  // EOI
    return true;
}

// =========================================================== decoder
namespace {

struct BitSource {
    const uint8_t* p;
    const uint8_t* end;
    uint32_t acc = 0;
    int nbits = 0;
    bool bad = false;
    BitSource(const uint8_t* d, size_t n) : p(d), end(d + n) {}
    inline int bit() {
        if (nbits == 0) {
            if (p >= end) { bad = true; return 0; }
            uint8_t b = *p++;
            if (b == 0xFF) {
                // Stuffed byte or a marker. EOI mid-scan = truncated.
                if (p >= end) { bad = true; return 0; }
                if (*p == 0x00) ++p;
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

struct DecTable {
    // T.81 F.2.2.3 canonical decode arrays, lengths 1..16.
    int32_t mincode[17] = {0};
    int32_t maxcode[17] = {0};
    int valptr[17] = {0};
    uint8_t huffval[17] = {0};
    bool valid = false;
};

inline void build_dec_table(const uint8_t bits[17], const uint8_t* vals,
                            int nvals, DecTable& t) {
    for (int i = 0; i < nvals && i < 17; ++i) t.huffval[i] = vals[i];
    int32_t code = 0;
    int k = 0;
    for (int l = 1; l <= 16; ++l) {
        t.valptr[l] = k;
        t.mincode[l] = code;
        code += bits[l];
        k += bits[l];
        t.maxcode[l] = code - 1;
        if (bits[l] == 0) t.maxcode[l] = -1;
        code <<= 1;
    }
    t.valid = true;
}

inline int decode_sym(BitSource& bs, const DecTable& t) {
    int32_t code = bs.bit();
    int l = 1;
    while (l < 16 && (t.maxcode[l] < 0 || code > t.maxcode[l])) {
        code = (code << 1) | bs.bit();
        ++l;
    }
    if (bs.bad || t.maxcode[l] < 0 || code > t.maxcode[l]) return -1;
    return t.huffval[t.valptr[l] + (code - t.mincode[l])];
}

} // namespace

bool lj92_decode_strip(const uint8_t* data, size_t len, int w, int h,
                       int ncomp, uint16_t* out) {
    if (!data || len < 8 || !out || w <= 0 || h <= 0 || ncomp < 1 || ncomp > 3)
        return false;
    size_t pos = 0;
    auto rd8 = [&]() -> int { return pos < len ? data[pos++] : -1; };
    auto rd16 = [&]() -> int {
        if (pos + 2 > len) return -1;
        int v = (data[pos] << 8) | data[pos + 1];
        pos += 2;
        return v;
    };
    if (rd8() != 0xFF || rd8() != 0xD8) return false;  // SOI

    DecTable tables[4];
    int comp_td[3] = {0, 0, 0};
    int sof_ncomp = 0;
    bool have_sof = false;

    for (;;) {
        int m = rd8();
        if (m < 0) return false;
        if (m != 0xFF) continue;  // resync (fill bytes allowed)
        int marker = rd8();
        while (marker == 0xFF) marker = rd8();
        if (marker < 0) return false;
        if (marker == 0xDA) break;  // SOS handled below
        if (marker == 0xD9) return false;  // EOI before any scan
        const int L = rd16();
        if (L < 2 || pos + (size_t)(L - 2) > len) return false;
        const size_t seg_end = pos + (size_t)(L - 2);
        if (marker == 0xC4) {  // DHT (may hold several tables)
            while (pos < seg_end) {
                const int tc_th = rd8();
                if (tc_th < 0) return false;
                const int th = tc_th & 0x0F;
                uint8_t bits[17] = {0};
                int nv = 0;
                for (int k = 1; k <= 16; ++k) {
                    const int b = rd8();
                    if (b < 0) return false;
                    bits[k] = (uint8_t)b;
                    nv += b;
                }
                if (nv > 17 || pos + (size_t)nv > seg_end) return false;
                if (th < 4) build_dec_table(bits, data + pos, nv, tables[th]);
                pos += (size_t)nv;
            }
        } else if (marker == 0xC3) {  // SOF3
            const int prec = rd8();
            const int fh = rd16(), fw = rd16();
            const int nf = rd8();
            if (prec != 16 || fh != h || fw != w || nf != ncomp) return false;
            for (int c = 0; c < nf; ++c) {
                (void)rd8();               // component id
                const int hv = rd8();
                (void)rd8();               // Tq
                if (hv != 0x11) return false;  // only H=V=1
            }
            have_sof = true;
            sof_ncomp = nf;
        } else {
            pos = seg_end;  // skip unknown segment
        }
    }
    if (!have_sof || sof_ncomp != ncomp) return false;

    // SOS
    {
        const int L = rd16();
        if (L < 2) return false;
        const int ns = rd8();
        if (ns != ncomp) return false;
        for (int c = 0; c < ns; ++c) {
            (void)rd8();  // component selector (assume SOF order)
            const int td = rd8();
            if (td < 0) return false;
            comp_td[c] = (td >> 4) & 0x0F;
            if (comp_td[c] > 3 || !tables[comp_td[c]].valid) return false;
        }
        const int ss = rd8();
        (void)rd8();  // Se
        const int ahal = rd8();
        if (ss != 1 || (ahal & 0x0F) != 0) return false;  // predictor 1, Pt 0
    }

    BitSource bs(data + pos, len - pos);
    int32_t predc[3];
    for (int y = 0; y < h; ++y) {
        uint16_t* row = out + (size_t)y * (size_t)w * (size_t)ncomp;
        const uint16_t* prev = (y > 0)
            ? out + (size_t)(y - 1) * (size_t)w * (size_t)ncomp : nullptr;
        for (int c = 0; c < ncomp; ++c)
            predc[c] = prev ? (int32_t)prev[c] : 32768;
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < ncomp; ++c) {
                const int ss = decode_sym(bs, tables[comp_td[c]]);
                if (ss < 0 || ss > 16 || bs.bad) return false;
                int32_t d;
                if (ss == 0) d = 0;
                else if (ss == 16) d = 32768;
                else {
                    int32_t v = (int32_t)bs.bits(ss);
                    // EXTEND (T.81 F.2.2.1): high bit clear = negative branch.
                    if (v < (1 << (ss - 1))) v += -(1 << ss) + 1;
                    d = v;
                }
                const int32_t s = (predc[c] + d) & 0xFFFF;
                row[(size_t)x * ncomp + c] = (uint16_t)s;
                predc[c] = s;
            }
        }
    }
    return !bs.bad;
}

} // namespace hhsr
