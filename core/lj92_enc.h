#pragma once
//
// Lossless JPEG (ITU T.81 process 14, "LJ92") encoder for DNG Compression=7
// strips. This is the codec every DNG reader ships (dcraw, LibRaw, ImageIO,
// Lightroom -- Apple ProRAW's own LinearRaw DNGs use it), so unlike the
// Deflate experiment (kDngCompress) the output stays universally readable.
//
// Scope: exactly what the writer needs -- 16-bit precision, 1..3 interleaved
// components (H=V=1), predictor 1 (left neighbour; row 0 predicts from
// 1<<15, later rows' first sample from the sample above), one scan, no
// restart markers. Each call produces a complete SOI..EOI stream, which is
// what a DNG strip must contain.
//
// Compression on 16-bit linear scene data is typically 2-3x and depends on
// content (entropy coding: smooth sky is cheap, noisy foliage is not), which
// is why the file size varies with the scene. The Huffman table is built
// per strip from the actual difference histogram (optimal for 17 symbols,
// length-limited to 16 with the all-ones code reserved per the spec).
//
#include <cstdint>
#include <vector>

namespace hhsr {

// rows: interleaved samples, ncomp per pixel, w pixels per row, h rows.
// Appends a complete lossless-JPEG stream to out (out is NOT cleared).
// Returns false only on invalid arguments.
bool lj92_encode_strip(const uint16_t* rows, int w, int h, int ncomp,
                       std::vector<uint8_t>& out);

// Decode a complete SOI..EOI stream produced by the encoder above (P=16,
// interleaved components, predictor 1; liberal in table ids). Dimensions and
// component count must match the stream's SOF3. out receives w*h*ncomp
// interleaved samples. Used by the DNG loader for Compression=7 tiles.
bool lj92_decode_strip(const uint8_t* data, size_t len, int w, int h,
                       int ncomp, uint16_t* out);

} // namespace hhsr
