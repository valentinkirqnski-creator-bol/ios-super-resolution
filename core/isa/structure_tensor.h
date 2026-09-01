#pragma once
// CPU port of ImageStackAlignator's structure-tensor / merge-kernel-covariance
// estimation stage: Kernels/opticalFlow.cu:ComputeDerivatives2Kernel,
// Kernels/kernel.cu:ComputeStructureTensor + ComputeKernelParam, and the
// intervening Gaussian blur (ImageStackAlignatorController.cs:1811,1855,
// gaussian_filter_1D + FilterGaussBorder).
#include <vector>

namespace isacpu {

struct Vec3f { float x = 0, y = 0, z = 0; };

// 5-tap central-difference derivatives, ((-1,8,0,-8,1)/12), with
// ComputeDerivatives2Kernel's CUAddressMode.Mirror boundary (out-of-bounds
// samples reflect symmetrically about the image edge). SIGN NOTE: the CUDA
// kernel's stencil (opticalFlow.cu:168-172) computes the NEGATED
// derivative; this port emits the standard-sign one. Provably equivalent
// for every consumer in this pipeline -- the structure tensor uses only
// dx*dx, dy*dy, dx*dy, all invariant under joint negation.
void compute_derivatives(const std::vector<float>& img, int width, int height,
                         std::vector<float>& dx, std::vector<float>& dy);

// Per-pixel {dx*dx, dy*dy, dx*dy}, matching ComputeStructureTensor.
void compute_structure_tensor_raw(const std::vector<float>& dx, const std::vector<float>& dy,
                                  int width, int height, std::vector<Vec3f>& out);

// Same tap generation as ImageStackAlignatorController.gaussian_filter_1D:
// radius = (int)(sigma/0.6f - 0.4f), size = 2*radius+1+2, capped at 99,
// normalized to sum 1. sigma<=0 returns a size-9 identity (delta) kernel.
std::vector<float> gaussian_filter_1d(float sigma);

// Separable blur (row pass then column pass) with replicate (clamp) border,
// matching NPPImage.FilterGaussBorder(..., NppiBorderType.Replicate).
void gaussian_blur_tensor(const std::vector<Vec3f>& in, int width, int height,
                          const std::vector<float>& kernel1d, std::vector<Vec3f>& out);

// Structure tensor -> anisotropic merge-kernel covariance, matching
// ComputeKernelParam verbatim. In place: `tensor` holds the blurred
// {dx*dx,dy*dy,dx*dy} on entry, {b22/det, b11/det, -b12/det} on exit.
void compute_kernel_param(std::vector<Vec3f>& tensor, int width, int height,
                          float Dth, float Dtr, float kDetail, float kDenoise,
                          float kStretch, float kShrink);

}  // namespace isacpu
