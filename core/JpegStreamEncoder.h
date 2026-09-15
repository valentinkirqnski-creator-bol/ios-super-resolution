#pragma once
//
// Minimal baseline (sequential DCT, Huffman) JPEG encoder that accepts pixel
// rows incrementally and writes the file as 8-row MCU stripes complete, so the
// whole image never has to be resident. Used by the HDR+ finish path to stay
// within a bounded memory budget at 48 MP (Apple ImageIO would need the entire
// output image in memory to encode).
//
// 4:4:4 (no chroma subsampling), standard Annex-K Huffman tables, quality-scaled
// quantisation. Not byte-identical to OpenCV/libjpeg output -- a valid baseline
// JPEG with the same pixels. Derived from the public-domain jo_write_jpg design,
// refactored to stream rows.
//
#include <cstdint>
#include <cstdio>
#include <vector>

namespace hhsr {

class JpegStreamEncoder {
public:
    // Writes the JFIF header immediately. quality in [1,100].
    JpegStreamEncoder(FILE* out, int width, int height, int quality);

    // Append `nrows` rows of interleaved RGB (8-bit). Rows are buffered into
    // 8-high MCU stripes and encoded as each stripe fills. Call repeatedly,
    // top to bottom, until `height` rows total have been supplied.
    void write_rows(const uint8_t* rgb, int nrows);

    // Flush the final (possibly padded) stripe and the bit buffer, write EOI.
    // Safe to call once after all rows are supplied. Returns false on I/O error.
    bool finish();

private:
    void write_header(int quality);
    void encode_mcu_row();                 // encodes the 8 buffered rows
    void put_byte(uint8_t b);
    void put_word(uint16_t w);
    void write_bits(const uint16_t bs[2]); // bs[0]=value, bs[1]=nbits
    void process_block(const float* px, const float* fdtbl, int& dc,
                       const uint16_t (*htdc)[2], const uint16_t (*htac)[2]);

    FILE* out_ = nullptr;
    int W_ = 0, H_ = 0;
    int rows_in_ = 0;                      // rows supplied so far
    int stripe_fill_ = 0;                  // rows currently in the stripe buffer
    std::vector<uint8_t> stripe_;          // 8 * W * 3
    float fdtbl_Y_[64], fdtbl_UV_[64];
    int dcY_ = 0, dcU_ = 0, dcV_ = 0;
    int bitBuf_ = 0, bitCnt_ = 0;
    bool ok_ = true;
    bool finished_ = false;
};

}  // namespace hhsr
