#pragma once
//
// Lossless JPEG (ITU-T T.81 Annex H, SOF3) -- the entropy coder DNG selects
// with Compression=7, and the one Adobe's own "Compressed (lossless)" DNGs use.
//
// Despite the name it shares nothing with the lossy JPEG in the preview SubIFD:
// no DCT, no 8x8 blocks, no quantization table, no quality setting. Each sample
// is predicted from its left neighbour in the same component and the difference
// is Huffman-coded, so decoding returns the exact uint16 that was handed in.
//
// Measured on a 48MP merge: 292.6MB uncompressed -> 172.1MB, versus 256.8MB for
// Deflate at zlib-6 and 294.6MB (larger than raw) at the zlib-1 the writer used
// to run. Unlike a Deflate stream, which is inherently serial, every strip here
// is a self-contained SOI..EOI bitstream, so strips encode and decode on all
// cores.
//
#include <cstddef>
#include <cstdint>
#include <vector>

namespace hhsr {

// Encode `rows` lines of `W` pixels x `ncomp` interleaved 16-bit samples.
// Predictor 1 (Ra), 16-bit precision, one Huffman table per component.
// ncomp is 1..4.
//
// `buf` is scratch the caller is expected to hand back on the next call: it is
// grown as needed and never shrunk, and `out_len` says how much of it holds the
// bitstream. Reusing it across strips is what keeps the encoder out of the
// allocator -- a fresh vector per strip spent more time zero-filling on resize
// and reallocating on shrink than the Huffman coding itself took.
bool ljpeg_encode(const uint16_t* src, int W, int rows, int ncomp,
                  std::vector<uint8_t>& buf, size_t& out_len);

// Convenience form that sizes `out` to exactly the bitstream. Allocates; use
// the scratch form on any hot path.
bool ljpeg_encode(const uint16_t* src, int W, int rows, int ncomp,
                  std::vector<uint8_t>& out);

// Decode one such stream into `dst` (rows*W*ncomp samples). The frame geometry
// carried in the stream must match W/rows/ncomp, otherwise this fails rather
// than writing a differently-shaped image into the caller's buffer.
bool ljpeg_decode(const uint8_t* data, size_t len, uint16_t* dst,
                  int W, int rows, int ncomp);

} // namespace hhsr
