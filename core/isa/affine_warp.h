#pragma once
// CPU port of the affine-warp machinery PreAlignment.cs uses for its
// rotation search: PEFStudioDX/Matrix.cs's Matrix3x3 (Rotation,
// RotAroundCenter, ShiftAffine) plus NPP's WarpAffine(..., Cubic).
#include <vector>
#include <array>

namespace isacpu {

// Row-major 3x3, matching Matrix3x3's [row,col] indexing exactly.
using Mat3 = std::array<float, 9>;

Mat3 mat3_identity();
Mat3 mat3_mul(const Mat3& a, const Mat3& b);  // a * b, same order as Matrix3x3's operator*
Mat3 mat3_shift(float x, float y);
Mat3 mat3_rotation_deg(float angle_deg);       // matches Matrix3x3.Rotation
Mat3 mat3_rot_around_center(float angle_deg, float width, float height);  // matches RotAroundCenter
Mat3 mat3_invert(const Mat3& m);               // full 3x3 inverse (bottom row is [0,0,1])

// Bicubic convolution (Keys 1981, a=-0.5 -- NPP's own Cubic interpolation
// convention). Returns false (value left at 0) if any of the 4x4 taps fall
// outside the source image -- matching ScanAngles' own
// `imgToTrackRotated.Set(0)` before each WarpAffine call, so out-of-bounds
// destination pixels stay background instead of extrapolating.
bool bicubic_sample(const std::vector<float>& img, int width, int height, float x, float y, float& out);

// Warps `src` into `dst` (dst pre-zeroed) using the FORWARD src->dst affine
// map `m` (matching Matrix3x3/NPP's own convention) -- internally inverts
// `m` and inverse-maps each destination pixel back into source space,
// bicubic-sampling there.
void warp_affine(const std::vector<float>& src, int width, int height, const Mat3& m,
                 std::vector<float>& dst);

}  // namespace isacpu
