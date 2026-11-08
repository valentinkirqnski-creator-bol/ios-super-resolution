#pragma once
// Handheld-Multi-Frame-Super-Resolution-1.4 robustness noise LUT (monte_carlo.py +
// noise_lut.py), ported exactly: the same Monte Carlo simulation, Numba's own RNG
// (MT19937 seeded with numba_rnd_init, standard_normal = polar Box-Muller that returns
// f*x2 and caches f*x1), the same block / chunk partition and seeds, float64 sums in the
// same order, the same finalisation (bin means, SEM, np.interp over empty bins) and the
// float32 curves robustness reads. Deterministic for a given (alpha, beta, bins, trials,
// seed, blocks_per_chunk): 1.4 uses blocks_per_chunk = 4 x Numba's thread count, so a LUT
// made on a 2-thread machine (Colab) is reproduced with blocks_per_chunk = 8.
#include <cstdint>
#include <string>
#include <vector>

namespace hhsr {

struct NoiseLut14 {
    int bins = 0;
    int64_t trials = 0;
    int seed = 0;
    int blocks_per_chunk = 0;
    double alpha[4] = {0, 0, 0, 0};   // R, G1, B, G2
    double beta[4] = {0, 0, 0, 0};
    std::vector<float> sigma_noise_sq;     // E[sigma^2 | measured brightness], float32 as 1.4
    std::vector<float> d_noise_sq;         // E[d^2 | measured brightness]
    std::vector<int64_t> bin_counts;
    std::vector<float> sigma_noise_sq_sem;
    std::vector<float> d_noise_sq_sem;
    bool valid() const { return bins >= 2 && (int)sigma_noise_sq.size() == bins; }
};

// generate_noise_lut(alpha, beta, bins=1001, trials=10_000_000, seed=0,
// chunk_size=250_000) with work_blocks = max(1, min(blocks_per_chunk, chunk trials)).
// num_threads: worker threads for THIS machine (0 = all); the result does not depend on it.
NoiseLut14 generate_noise_lut14(const double alpha[4], const double beta[4],
                                int bins = 1001, int64_t trials = 10000000, int seed = 0,
                                int blocks_per_chunk = 8, int chunk_size = 250000,
                                int num_threads = 0);

// expand_noise_profile_to_rgbg: 1 -> x4, 3 -> (a0, a1, a2, a1), 4 -> as is.
void expand_noise_profile_to_rgbg14(const double* in, int n, double out[4]);

// Generate once per (alpha, beta, bins, trials, seed, blocks) and reuse: an in-memory
// cache, plus an on-disk copy in `cache_dir` when non-empty (binary, versioned).
const NoiseLut14& noise_lut14_cached(const double alpha[4], const double beta[4],
                                     const std::string& cache_dir,
                                     int bins = 1001, int64_t trials = 10000000,
                                     int seed = 0, int blocks_per_chunk = 8);

bool save_noise_lut14(const std::string& path, const NoiseLut14& lut);
bool load_noise_lut14(const std::string& path, NoiseLut14& lut);

}  // namespace hhsr
