#include "JpegStreamEncoder.h"

#include <cmath>
#include <cstring>

// Baseline JPEG encoder, streaming. Tables and the AAN DCT / Huffman routines
// follow the public-domain jo_write_jpg (Jon Olick), refactored to consume rows
// incrementally. 4:4:4, standard Annex-K Huffman tables.

namespace hhsr {
namespace {

const int kZigZag[64] = {
    0,1,5,6,14,15,27,28,2,4,7,13,16,26,29,42,3,8,12,17,25,30,41,43,
    9,11,18,24,31,40,44,53,10,19,23,32,39,45,52,54,20,22,33,38,46,51,55,60,
    21,34,37,47,50,56,59,61,35,36,48,49,57,58,62,63 };

const int kYQT[64] = {16,11,10,16,24,40,51,61,12,12,14,19,26,58,60,55,14,13,16,24,40,57,69,56,
    14,17,22,29,51,87,80,62,18,22,37,56,68,109,103,77,24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101,72,92,95,98,112,100,103,99};
const int kUVQT[64] = {17,18,24,47,99,99,99,99,18,21,26,66,99,99,99,99,24,26,56,99,99,99,99,99,
    47,66,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,
    99,99,99,99,99,99,99,99,99,99,99,99,99,99,99,99};
const float kAASF[8] = {2.828427125f, 3.923141359f, 3.695518131f, 3.325878449f,
    2.828427125f, 2.222280420f, 1.530733729f, 0.780361287f};

const uint8_t kDcLumCodes[17] = {0,0,1,5,1,1,1,1,1,1,0,0,0,0,0,0,0};
const uint8_t kDcLumVals[12]  = {0,1,2,3,4,5,6,7,8,9,10,11};
const uint8_t kAcLumCodes[17] = {0,0,2,1,3,3,2,4,3,5,5,4,4,0,0,1,0x7d};
const uint8_t kAcLumVals[162] = {
 0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,
 0x23,0x42,0xb1,0xc1,0x15,0x52,0xd1,0xf0,0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,0x26,0x27,0x28,
 0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,
 0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,0x87,0x88,0x89,
 0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,
 0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,0xe1,0xe2,
 0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa};
const uint8_t kDcChrCodes[17] = {0,0,3,1,1,1,1,1,1,1,1,1,0,0,0,0,0};
const uint8_t kDcChrVals[12]  = {0,1,2,3,4,5,6,7,8,9,10,11};
const uint8_t kAcChrCodes[17] = {0,0,2,1,2,4,4,3,4,7,5,4,4,0,1,2,0x77};
const uint8_t kAcChrVals[162] = {
 0x00,0x01,0x02,0x03,0x11,0x04,0x05,0x21,0x31,0x06,0x12,0x41,0x51,0x07,0x61,0x71,0x13,0x22,0x32,0x81,0x08,0x14,0x42,0x91,
 0xa1,0xb1,0xc1,0x09,0x23,0x33,0x52,0xf0,0x15,0x62,0x72,0xd1,0x0a,0x16,0x24,0x34,0xe1,0x25,0xf1,0x17,0x18,0x19,0x1a,0x26,
 0x27,0x28,0x29,0x2a,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,
 0x59,0x5a,0x63,0x64,0x65,0x66,0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x82,0x83,0x84,0x85,0x86,0x87,
 0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,
 0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,0xda,
 0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf2,0xf3,0xf4,0xf5,0xf6,0xf7,0xf8,0xf9,0xfa};

// Build the canonical (code,length)[256] Huffman table from the count/value
// spec: codes are assigned in order of increasing length, left-shifting by one
// bit each time the length increases.
void build_ht_canonical(uint16_t HT[256][2], const uint8_t* nrcodes, const uint8_t* values) {
    std::memset(HT, 0, sizeof(uint16_t) * 256 * 2);
    int code = 0, k = 0;
    for (int len = 1; len <= 16; ++len) {
        for (int j = 0; j < nrcodes[len]; ++j) {
            HT[values[k]][0] = (uint16_t)code;
            HT[values[k]][1] = (uint16_t)len;
            ++code; ++k;
        }
        code <<= 1;
    }
}

// AAN 1-D DCT in place on 8 samples spaced by `stride`. (jo_write_jpg)
void jo_dct(float& d0, float& d1, float& d2, float& d3, float& d4, float& d5, float& d6, float& d7) {
    float tmp0 = d0 + d7, tmp7 = d0 - d7;
    float tmp1 = d1 + d6, tmp6 = d1 - d6;
    float tmp2 = d2 + d5, tmp5 = d2 - d5;
    float tmp3 = d3 + d4, tmp4 = d3 - d4;
    float tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3;
    float tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;
    d0 = tmp10 + tmp11; d4 = tmp10 - tmp11;
    float z1 = (tmp12 + tmp13) * 0.707106781f;
    d2 = tmp13 + z1; d6 = tmp13 - z1;
    tmp10 = tmp4 + tmp5; tmp11 = tmp5 + tmp6; tmp12 = tmp6 + tmp7;
    float z5 = (tmp10 - tmp12) * 0.382683433f;
    float z2 = tmp10 * 0.541196100f + z5;
    float z4 = tmp12 * 1.306562965f + z5;
    float z3 = tmp11 * 0.707106781f;
    float z11 = tmp7 + z3, z13 = tmp7 - z3;
    d5 = z13 + z2; d3 = z13 - z2; d1 = z11 + z4; d7 = z11 - z4;
}

int calc_bits(int v, uint16_t bits[2]) {
    int mag = v < 0 ? -v : v;          // magnitude decides the category (nbits)
    int val = v < 0 ? v - 1 : v;       // negative: additional bits from (v-1), masked
    int nb = 0; while (mag) { mag >>= 1; ++nb; }
    bits[0] = (uint16_t)(val & ((1 << nb) - 1));
    bits[1] = (uint16_t)nb;
    return nb;
}

}  // namespace

// Static Huffman tables shared by all encoders (built once).
static uint16_t g_YDC[256][2], g_YAC[256][2], g_UDC[256][2], g_UAC[256][2];
static bool g_ht_ready = false;
static void ensure_ht() {
    if (g_ht_ready) return;
    build_ht_canonical(g_YDC, kDcLumCodes, kDcLumVals);
    build_ht_canonical(g_YAC, kAcLumCodes, kAcLumVals);
    build_ht_canonical(g_UDC, kDcChrCodes, kDcChrVals);
    build_ht_canonical(g_UAC, kAcChrCodes, kAcChrVals);
    g_ht_ready = true;
}

void JpegStreamEncoder::put_byte(uint8_t b) { if (std::fputc(b, out_) == EOF) ok_ = false; }
void JpegStreamEncoder::put_word(uint16_t w) { put_byte((uint8_t)(w >> 8)); put_byte((uint8_t)(w & 0xff)); }

void JpegStreamEncoder::write_bits(const uint16_t bs[2]) {
    bitCnt_ += bs[1];
    bitBuf_ |= bs[0] << (24 - bitCnt_);
    while (bitCnt_ >= 8) {
        uint8_t c = (uint8_t)((bitBuf_ >> 16) & 0xFF);
        put_byte(c);
        if (c == 0xFF) put_byte(0);      // byte stuffing
        bitBuf_ <<= 8; bitCnt_ -= 8;
    }
}

JpegStreamEncoder::JpegStreamEncoder(FILE* out, int width, int height, int quality)
    : out_(out), W_(width), H_(height) {
    ensure_ht();
    stripe_.assign((size_t)8 * W_ * 3, 0);
    write_header(quality);
}

void JpegStreamEncoder::write_header(int quality) {
    quality = quality < 1 ? 1 : (quality > 100 ? 100 : quality);
    const int q = quality < 50 ? 5000 / quality : 200 - quality * 2;
    uint8_t YTable[64], UVTable[64];
    for (int i = 0; i < 64; ++i) {
        int y = (kYQT[i] * q + 50) / 100; y = y < 1 ? 1 : (y > 255 ? 255 : y);
        int u = (kUVQT[i] * q + 50) / 100; u = u < 1 ? 1 : (u > 255 ? 255 : u);
        YTable[kZigZag[i]] = (uint8_t)y;
        UVTable[kZigZag[i]] = (uint8_t)u;
    }
    for (int r = 0; r < 8; ++r)
        for (int c = 0; c < 8; ++c) {
            const int i = r * 8 + c;
            fdtbl_Y_[i]  = 1.f / (YTable[kZigZag[i]]  * kAASF[r] * kAASF[c]);
            fdtbl_UV_[i] = 1.f / (UVTable[kZigZag[i]] * kAASF[r] * kAASF[c]);
        }
    // SOI, APP0(JFIF), DQT, SOF0, DHT, SOS
    static const uint8_t head[] = {0xFF,0xD8, 0xFF,0xE0,0,0x10,'J','F','I','F',0,1,1,0,0,1,0,1,0,0};
    for (uint8_t b : head) put_byte(b);
    put_byte(0xFF); put_byte(0xDB); put_word(0x84);
    put_byte(0); for (int i=0;i<64;++i) put_byte(YTable[i]);
    put_byte(1); for (int i=0;i<64;++i) put_byte(UVTable[i]);
    // SOF0
    static const uint8_t sof[] = {0xFF,0xC0,0,0x11,8};
    for (uint8_t b : sof) put_byte(b);
    put_word((uint16_t)H_); put_word((uint16_t)W_);
    put_byte(3);
    put_byte(1); put_byte(0x11); put_byte(0);
    put_byte(2); put_byte(0x11); put_byte(1);
    put_byte(3); put_byte(0x11); put_byte(1);
    // DHT
    auto put_dht = [&](uint8_t id, const uint8_t* codes, int nval, const uint8_t* vals) {
        put_byte(id);
        for (int i = 1; i <= 16; ++i) put_byte(codes[i]);
        for (int i = 0; i < nval; ++i) put_byte(vals[i]);
    };
    put_byte(0xFF); put_byte(0xC4); put_word(0x01A2);
    put_dht(0x00, kDcLumCodes, 12, kDcLumVals);
    put_dht(0x10, kAcLumCodes, 162, kAcLumVals);
    put_dht(0x01, kDcChrCodes, 12, kDcChrVals);
    put_dht(0x11, kAcChrCodes, 162, kAcChrVals);
    // SOS
    static const uint8_t sos[] = {0xFF,0xDA,0,0x0C,3,1,0,2,0x11,3,0x11,0,0x3F,0};
    for (uint8_t b : sos) put_byte(b);
}

void JpegStreamEncoder::process_block(const float* px, const float* fdtbl, int& dc,
                                      const uint16_t (*htdc)[2], const uint16_t (*htac)[2]) {
    float du[64];
    for (int i = 0; i < 64; ++i) du[i] = px[i] - 128.f;
    for (int i = 0; i < 8; ++i)   // rows
        jo_dct(du[i*8+0],du[i*8+1],du[i*8+2],du[i*8+3],du[i*8+4],du[i*8+5],du[i*8+6],du[i*8+7]);
    for (int i = 0; i < 8; ++i)   // cols
        jo_dct(du[i+0],du[i+8],du[i+16],du[i+24],du[i+32],du[i+40],du[i+48],du[i+56]);
    int DU[64];
    for (int i = 0; i < 64; ++i) {
        float v = du[i] * fdtbl[i];
        DU[kZigZag[i]] = (int)(v < 0 ? v - 0.5f : v + 0.5f);
    }
    // DC
    int diff = DU[0] - dc; dc = DU[0];
    if (diff == 0) write_bits(htdc[0]);
    else { uint16_t bits[2]; int nb = calc_bits(diff, bits); write_bits(htdc[nb]); write_bits(bits); }
    // AC
    int end = 63; while (end > 0 && DU[end] == 0) --end;
    if (end == 0) { write_bits(htac[0x00]); return; }
    for (int i = 1; i <= end; ) {
        int run = 0;
        while (i <= end && DU[i] == 0) { ++run; ++i; }
        while (run > 15) { write_bits(htac[0xF0]); run -= 16; }
        uint16_t bits[2]; int nb = calc_bits(DU[i], bits);
        write_bits(htac[(run << 4) | nb]);
        write_bits(bits);
        ++i;
    }
    if (end != 63) write_bits(htac[0x00]);
}

void JpegStreamEncoder::encode_mcu_row() {
    // 8 rows in stripe_ (partial last rows already edge-padded). One MCU = 8x8.
    float Y[64], U[64], V[64];
    const int nbx = (W_ + 7) / 8;
    for (int bx = 0; bx < nbx; ++bx) {
        for (int yy = 0; yy < 8; ++yy)
            for (int xx = 0; xx < 8; ++xx) {
                int x = bx * 8 + xx; if (x >= W_) x = W_ - 1;   // edge pad columns
                const uint8_t* p = &stripe_[((size_t)yy * W_ + x) * 3];
                const float r = p[0], g = p[1], b = p[2];
                const int i = yy * 8 + xx;
                Y[i] = 0.299f*r + 0.587f*g + 0.114f*b;
                U[i] = -0.168736f*r - 0.331264f*g + 0.5f*b + 128.f;
                V[i] = 0.5f*r - 0.418688f*g - 0.081312f*b + 128.f;
            }
        process_block(Y, fdtbl_Y_, dcY_, g_YDC, g_YAC);
        process_block(U, fdtbl_UV_, dcU_, g_UDC, g_UAC);
        process_block(V, fdtbl_UV_, dcV_, g_UDC, g_UAC);
    }
}

void JpegStreamEncoder::write_rows(const uint8_t* rgb, int nrows) {
    if (!ok_ || finished_) return;
    for (int r = 0; r < nrows && rows_in_ < H_; ++r, ++rows_in_) {
        std::memcpy(&stripe_[(size_t)stripe_fill_ * W_ * 3], rgb + (size_t)r * W_ * 3,
                    (size_t)W_ * 3);
        if (++stripe_fill_ == 8) { encode_mcu_row(); stripe_fill_ = 0; }
    }
}

bool JpegStreamEncoder::finish() {
    if (finished_) return ok_;
    finished_ = true;
    if (stripe_fill_ > 0) {
        // edge-pad the remaining rows down to 8 by replicating the last row
        for (int y = stripe_fill_; y < 8; ++y)
            std::memcpy(&stripe_[(size_t)y * W_ * 3],
                        &stripe_[(size_t)(stripe_fill_ - 1) * W_ * 3], (size_t)W_ * 3);
        encode_mcu_row();
        stripe_fill_ = 0;
    }
    // flush bit buffer (pad with 1s), then EOI
    const uint16_t fill[2] = {0x7F, 7};
    write_bits(fill);
    put_byte(0xFF); put_byte(0xD9);
    return ok_;
}

}  // namespace hhsr
