// See noise_lut14.h. Exact port of 1.4 monte_carlo.py (+ Numba's RNG). No FMA contraction:
// Numba compiles without fast-math, so every multiply-add below must round twice.
#if defined(__clang__)
#pragma clang fp contract(off)
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif
#include "noise_lut14.h"
#include "parallel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <tuple>

namespace hhsr {

namespace {

// ---- Numba's thread-local RNG state (numba/_random.c), verbatim ---------------------
constexpr int MT_N = 624, MT_M = 397;
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfU, MT_UPPER = 0x80000000U, MT_LOWER = 0x7fffffffU;

struct RndState {
    int index = MT_N;
    uint32_t mt[MT_N];
    int has_gauss = 0;
    double gauss = 0.0;
};

void rnd_shuffle(RndState& s) {
    int i;
    uint32_t y;
    for (i = 0; i < MT_N - MT_M; i++) {
        y = (s.mt[i] & MT_UPPER) | (s.mt[i + 1] & MT_LOWER);
        s.mt[i] = s.mt[i + MT_M] ^ (y >> 1) ^ (uint32_t)(-(int32_t)(y & 1) & MT_MATRIX_A);
    }
    for (; i < MT_N - 1; i++) {
        y = (s.mt[i] & MT_UPPER) | (s.mt[i + 1] & MT_LOWER);
        s.mt[i] = s.mt[i + (MT_M - MT_N)] ^ (y >> 1) ^ (uint32_t)(-(int32_t)(y & 1) & MT_MATRIX_A);
    }
    y = (s.mt[MT_N - 1] & MT_UPPER) | (s.mt[0] & MT_LOWER);
    s.mt[MT_N - 1] = s.mt[MT_M - 1] ^ (y >> 1) ^ (uint32_t)(-(int32_t)(y & 1) & MT_MATRIX_A);
}

// numba_rnd_init (np.random.seed(int) -> uint32 seed)
void rnd_init(RndState& s, uint32_t seed) {
    for (uint32_t pos = 0; pos < (uint32_t)MT_N; pos++) {
        s.mt[pos] = seed;
        seed = (1812433253U * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffU;
    }
    s.index = MT_N;
    s.has_gauss = 0;
    s.gauss = 0.0;
}

inline uint32_t next_int32(RndState& s) {
    if (s.index == MT_N) { rnd_shuffle(s); s.index = 0; }
    uint32_t y = s.mt[s.index++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680U;
    y ^= (y << 15) & 0xefc60000U;
    y ^= (y >> 18);
    return y;
}

inline double next_double(RndState& s) {           // get_next_double
    const double a = (double)(next_int32(s) >> 5);
    const double b = (double)(next_int32(s) >> 6);
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

// np.random.standard_normal() == np.random.normal(0.0, 1.0) (numba legacy): has_gauss cache;
// otherwise the polar pair, store f*x1, return f*x2; result = 0.0 + 1.0 * ret.
inline double standard_normal(RndState& s) {
    double ret;
    if (s.has_gauss) {
        ret = s.gauss;
        s.has_gauss = 0;
    } else {
        double x1, x2, r2;
        while (true) {
            x1 = 2.0 * next_double(s) - 1.0;
            x2 = 2.0 * next_double(s) - 1.0;
            r2 = x1 * x1 + x2 * x2;
            if (r2 < 1.0 && r2 != 0.0) break;
        }
        const double f = std::sqrt(-2.0 * std::log(r2) / r2);
        s.gauss = f * x1;
        ret = f * x2;
        s.has_gauss = 1;
    }
    return 0.0 + 1.0 * ret;
}

// ---- monte_carlo.py ----------------------------------------------------------------
inline double noisy_raw_value(RndState& s, double brightness, double alpha, double beta) {
    double variance = alpha * brightness + beta;
    if (variance < 0.0) variance = 0.0;
    const double value = brightness + std::sqrt(variance) * standard_normal(s);
    if (value < 0.0) return 0.0;
    if (value > 1.0) return 1.0;
    return value;
}

inline void channel_patch_stats(RndState& s, double b, double a, double be,
                                double& mean_out, double& var_out) {
    double mean = 0.0, moment2 = 0.0;
    for (int sample = 0; sample < 9; ++sample) {
        const double value = std::sqrt(noisy_raw_value(s, b, a, be));
        const double delta = value - mean;
        mean += delta / (double)(sample + 1);
        moment2 += delta * (value - mean);
    }
    mean_out = mean;
    var_out = std::max(moment2 / 9.0, 0.0);
}

inline void green_patch_stats(RndState& s, double b, double ag1, double bg1, double ag2,
                              double bg2, double& mean_out, double& var_out) {
    double mean = 0.0, moment2 = 0.0;
    for (int sample = 0; sample < 9; ++sample) {
        const double green1 = noisy_raw_value(s, b, ag1, bg1);
        const double green2 = noisy_raw_value(s, b, ag2, bg2);
        const double value = std::sqrt(0.5 * (green1 + green2));
        const double delta = value - mean;
        mean += delta / (double)(sample + 1);
        moment2 += delta * (value - mean);
    }
    mean_out = mean;
    var_out = std::max(moment2 / 9.0, 0.0);
}

struct BlockAcc {
    std::vector<int64_t> counts;
    std::vector<double> sigma_sum, sigma_sum_sq, d_sum, d_sum_sq;
    explicit BlockAcc(int bins)
        : counts(bins, 0), sigma_sum(bins, 0.0), sigma_sum_sq(bins, 0.0), d_sum(bins, 0.0), d_sum_sq(bins, 0.0) {}
};

// One prange block of _simulate_chunk.
void simulate_block(const double* alpha, const double* beta, int bins, int64_t start, int64_t stop,
                    int64_t total_trials, int64_t block_seed, BlockAcc& acc) {
    RndState st;
    rnd_init(st, (uint32_t)block_seed);
    for (int64_t trial = start; trial < stop; ++trial) {
        const double latent = ((double)trial + 0.5) / (double)total_trials;
        double ref_r, ref_var_r, ref_g, ref_var_g, ref_b, ref_var_b, mov_r, mov_g, mov_b, unused;
        channel_patch_stats(st, latent, alpha[0], beta[0], ref_r, ref_var_r);
        green_patch_stats(st, latent, alpha[1], beta[1], alpha[3], beta[3], ref_g, ref_var_g);
        channel_patch_stats(st, latent, alpha[2], beta[2], ref_b, ref_var_b);
        channel_patch_stats(st, latent, alpha[0], beta[0], mov_r, unused);
        green_patch_stats(st, latent, alpha[1], beta[1], alpha[3], beta[3], mov_g, unused);
        channel_patch_stats(st, latent, alpha[2], beta[2], mov_b, unused);

        const double measured = (ref_r + ref_g + ref_b) / 3.0;
        int64_t index = (int64_t)std::floor(measured * (double)(bins - 1) + 0.5);
        if (index < 0) index = 0;
        else if (index >= bins) index = bins - 1;

        const double sigma_sq = ref_var_r + ref_var_g + ref_var_b;
        const double dr = ref_r - mov_r, dg = ref_g - mov_g, db = ref_b - mov_b;
        const double d_sq = dr * dr + dg * dg + db * db;
        acc.counts[index] += 1;
        acc.sigma_sum[index] += sigma_sq;
        acc.sigma_sum_sq[index] += sigma_sq * sigma_sq;
        acc.d_sum[index] += d_sq;
        acc.d_sum_sq[index] += d_sq * d_sq;
    }
}

// np.interp(indices, indices[populated], values[populated]) on an integer grid.
std::vector<double> interpolate_missing(const std::vector<double>& v, const std::vector<int64_t>& counts) {
    const int n = (int)v.size();
    std::vector<int> xp;
    for (int i = 0; i < n; ++i) if (counts[i] > 0) xp.push_back(i);
    std::vector<double> out(n, 0.0);
    if (xp.empty()) return out;
    for (int i = 0; i < n; ++i) {
        if (i <= xp.front()) { out[i] = v[xp.front()]; continue; }
        if (i >= xp.back()) { out[i] = v[xp.back()]; continue; }
        const int j = (int)(std::upper_bound(xp.begin(), xp.end(), i) - xp.begin()) - 1;  // xp[j] <= i < xp[j+1]
        const int x0 = xp[j], x1 = xp[j + 1];
        if (x0 == i) { out[i] = v[x0]; continue; }
        // numpy interp: slope = (fp1 - fp0)/(xp1 - xp0); y = slope*(x - xp0) + fp0
        const double slope = (v[x1] - v[x0]) / (double)(x1 - x0);
        out[i] = slope * (double)(i - x0) + v[x0];
    }
    return out;
}

void finalize(const std::vector<int64_t>& counts, const std::vector<double>& sums,
              const std::vector<double>& sums_sq, std::vector<double>& mean_o, std::vector<double>& sem_o) {
    const int n = (int)counts.size();
    std::vector<double> means(n, 0.0), sem(n, 0.0);
    for (int i = 0; i < n; ++i) {
        if (counts[i] <= 0) continue;
        means[i] = sums[i] / (double)counts[i];
        const double var = std::max(sums_sq[i] / (double)counts[i] - means[i] * means[i], 0.0);
        sem[i] = std::sqrt(var / (double)counts[i]);
    }
    mean_o = interpolate_missing(means, counts);
    sem_o = interpolate_missing(sem, counts);
}

}  // namespace

void expand_noise_profile_to_rgbg14(const double* in, int n, double out[4]) {
    if (n == 1) { out[0] = out[1] = out[2] = out[3] = in[0]; return; }
    if (n == 3) { out[0] = in[0]; out[1] = in[1]; out[2] = in[2]; out[3] = in[1]; return; }
    for (int i = 0; i < 4; ++i) out[i] = (i < n) ? in[i] : in[n - 1];
}

NoiseLut14 generate_noise_lut14(const double alpha_in[4], const double beta_in[4], int bins,
                                int64_t trials, int seed, int blocks_per_chunk, int chunk_size,
                                int num_threads) {
    NoiseLut14 lut;
    if (bins < 2 || trials < 1 || seed < 0 || chunk_size < 1 || blocks_per_chunk < 1) return lut;
    double alpha[4], beta[4];
    for (int i = 0; i < 4; ++i) { alpha[i] = alpha_in[i]; beta[i] = beta_in[i]; }

    // The chunk/block partition (and every block seed) as generate_noise_lut builds it.
    struct Task { int64_t start, stop, seed; size_t chunk; };
    std::vector<Task> tasks;
    std::vector<size_t> chunk_first, chunk_n;
    for (int64_t completed = 0; completed < trials;) {
        const int64_t stop = std::min<int64_t>(completed + chunk_size, trials);
        const int64_t chunk_trials = stop - completed;
        const int64_t work_blocks = std::max<int64_t>(1, std::min<int64_t>(blocks_per_chunk, chunk_trials));
        chunk_first.push_back(tasks.size());
        chunk_n.push_back((size_t)work_blocks);
        for (int64_t block = 0; block < work_blocks; ++block) {
            const int64_t start = completed + (chunk_trials * block) / work_blocks;
            const int64_t bstop = completed + (chunk_trials * (block + 1)) / work_blocks;
            const int64_t bseed = ((int64_t)seed + 104729LL * (block + 1) + 1000003LL * completed) & 0x7fffffffLL;
            tasks.push_back({start, bstop, bseed, chunk_first.size() - 1});
        }
        completed = stop;
    }
    std::vector<BlockAcc> accs;
    accs.reserve(tasks.size());
    for (size_t i = 0; i < tasks.size(); ++i) accs.emplace_back(bins);
    parallel_rows((int)tasks.size(), num_threads, [&](int t) {
        const Task& k = tasks[(size_t)t];
        simulate_block(alpha, beta, bins, k.start, k.stop, trials, k.seed, accs[(size_t)t]);
    });

    // counts += np.sum(partial, axis=0) per chunk, chunks in order.
    std::vector<int64_t> counts(bins, 0);
    std::vector<double> ss(bins, 0.0), ss2(bins, 0.0), ds(bins, 0.0), ds2(bins, 0.0);
    for (size_t c = 0; c < chunk_first.size(); ++c) {
        std::vector<int64_t> cc(bins, 0);
        std::vector<double> a(bins), b(bins), d(bins), e(bins);
        for (size_t k = 0; k < chunk_n[c]; ++k) {
            const BlockAcc& B = accs[chunk_first[c] + k];
            for (int i = 0; i < bins; ++i) {
                cc[i] += B.counts[i];
                if (k == 0) { a[i] = B.sigma_sum[i]; b[i] = B.sigma_sum_sq[i]; d[i] = B.d_sum[i]; e[i] = B.d_sum_sq[i]; }
                else { a[i] += B.sigma_sum[i]; b[i] += B.sigma_sum_sq[i]; d[i] += B.d_sum[i]; e[i] += B.d_sum_sq[i]; }
            }
        }
        for (int i = 0; i < bins; ++i) {
            counts[i] += cc[i]; ss[i] += a[i]; ss2[i] += b[i]; ds[i] += d[i]; ds2[i] += e[i];
        }
    }
    std::vector<double> sigma_curve, sigma_sem, d_curve, d_sem;
    finalize(counts, ss, ss2, sigma_curve, sigma_sem);
    finalize(counts, ds, ds2, d_curve, d_sem);

    lut.bins = bins; lut.trials = trials; lut.seed = seed; lut.blocks_per_chunk = blocks_per_chunk;
    for (int i = 0; i < 4; ++i) { lut.alpha[i] = alpha[i]; lut.beta[i] = beta[i]; }
    lut.bin_counts = counts;
    lut.sigma_noise_sq.resize(bins); lut.d_noise_sq.resize(bins);
    lut.sigma_noise_sq_sem.resize(bins); lut.d_noise_sq_sem.resize(bins);
    for (int i = 0; i < bins; ++i) {
        lut.sigma_noise_sq[i] = (float)sigma_curve[i];
        lut.d_noise_sq[i] = (float)d_curve[i];
        lut.sigma_noise_sq_sem[i] = (float)sigma_sem[i];
        lut.d_noise_sq_sem[i] = (float)d_sem[i];
    }
    return lut;
}

// ---- cache --------------------------------------------------------------------------
static const uint32_t kLutMagic = 0x3134544cU;   // "LT41"
static const uint32_t kLutVersion = 1;

bool save_noise_lut14(const std::string& path, const NoiseLut14& lut) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t hdr[2] = {kLutMagic, kLutVersion};
    std::fwrite(hdr, 4, 2, f);
    const int32_t ints[3] = {lut.bins, lut.seed, lut.blocks_per_chunk};
    std::fwrite(ints, 4, 3, f);
    std::fwrite(&lut.trials, 8, 1, f);
    std::fwrite(lut.alpha, 8, 4, f); std::fwrite(lut.beta, 8, 4, f);
    std::fwrite(lut.sigma_noise_sq.data(), 4, lut.bins, f);
    std::fwrite(lut.d_noise_sq.data(), 4, lut.bins, f);
    std::fwrite(lut.bin_counts.data(), 8, lut.bins, f);
    std::fwrite(lut.sigma_noise_sq_sem.data(), 4, lut.bins, f);
    std::fwrite(lut.d_noise_sq_sem.data(), 4, lut.bins, f);
    const bool ok = std::ferror(f) == 0;
    std::fclose(f);
    return ok;
}

bool load_noise_lut14(const std::string& path, NoiseLut14& lut) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t hdr[2] = {0, 0};
    int32_t ints[3] = {0, 0, 0};
    bool ok = std::fread(hdr, 4, 2, f) == 2 && hdr[0] == kLutMagic && hdr[1] == kLutVersion &&
              std::fread(ints, 4, 3, f) == 3 && ints[0] >= 2 && ints[0] < 1000000;
    if (ok) {
        lut.bins = ints[0]; lut.seed = ints[1]; lut.blocks_per_chunk = ints[2];
        lut.sigma_noise_sq.resize(lut.bins); lut.d_noise_sq.resize(lut.bins); lut.bin_counts.resize(lut.bins);
        lut.sigma_noise_sq_sem.resize(lut.bins); lut.d_noise_sq_sem.resize(lut.bins);
        ok = std::fread(&lut.trials, 8, 1, f) == 1 && std::fread(lut.alpha, 8, 4, f) == 4 &&
             std::fread(lut.beta, 8, 4, f) == 4 &&
             std::fread(lut.sigma_noise_sq.data(), 4, lut.bins, f) == (size_t)lut.bins &&
             std::fread(lut.d_noise_sq.data(), 4, lut.bins, f) == (size_t)lut.bins &&
             std::fread(lut.bin_counts.data(), 8, lut.bins, f) == (size_t)lut.bins &&
             std::fread(lut.sigma_noise_sq_sem.data(), 4, lut.bins, f) == (size_t)lut.bins &&
             std::fread(lut.d_noise_sq_sem.data(), 4, lut.bins, f) == (size_t)lut.bins;
    }
    std::fclose(f);
    if (!ok) lut = NoiseLut14{};
    return ok;
}

const NoiseLut14& noise_lut14_cached(const double alpha[4], const double beta[4],
                                     const std::string& cache_dir, int bins, int64_t trials,
                                     int seed, int blocks_per_chunk) {
    using Key = std::tuple<double, double, double, double, double, double, double, double, int, int64_t, int, int>;
    static std::mutex mtx;
    static std::map<Key, NoiseLut14> cache;
    const Key key{alpha[0], alpha[1], alpha[2], alpha[3], beta[0], beta[1], beta[2], beta[3],
                  bins, trials, seed, blocks_per_chunk};
    std::lock_guard<std::mutex> lock(mtx);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    NoiseLut14 lut;
    std::string path;
    if (!cache_dir.empty()) {
        char name[256];
        std::snprintf(name, sizeof(name), "/noise_lut14_%.9g_%.9g_%.9g_%.9g_%.9g_%.9g_%.9g_%.9g_%d_%lld_%d_%d.bin",
                      alpha[0], alpha[1], alpha[2], alpha[3], beta[0], beta[1], beta[2], beta[3],
                      bins, (long long)trials, seed, blocks_per_chunk);
        path = cache_dir + name;
        NoiseLut14 disk;
        if (load_noise_lut14(path, disk) && disk.bins == bins && disk.trials == trials &&
            std::memcmp(disk.alpha, alpha, sizeof(disk.alpha)) == 0 &&
            std::memcmp(disk.beta, beta, sizeof(disk.beta)) == 0)
            lut = std::move(disk);
    }
    if (!lut.valid()) {
        lut = generate_noise_lut14(alpha, beta, bins, trials, seed, blocks_per_chunk);
        if (!path.empty() && lut.valid()) {
            std::error_code ec;
            std::filesystem::create_directories(cache_dir, ec);
            (void)save_noise_lut14(path, lut);
        }
    }
    return cache.emplace(key, std::move(lut)).first->second;
}

}  // namespace hhsr
