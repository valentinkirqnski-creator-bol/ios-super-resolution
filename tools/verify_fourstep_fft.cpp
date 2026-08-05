// Host verification of the four-step FFT implemented by fft_small_batched in
// HHSRKernels.metal.
//
//   N = N1*N2,  n = n1*N2 + n2,  k = k2*N1 + k1
//   pass A: N2 transforms of size N1 over n1, then twiddle by W_N^(n2*k1)
//   pass B: N1 transforms of size N2 over n2, written to k2*N1 + k1
//
// The Metal kernel transcribes this index math, so a pass here means the
// decomposition and gather/scatter strides the GPU uses are correct.
//
// Build: g++ -std=c++17 -O2 tools/verify_fourstep_fft.cpp -o verify_fourstep
// Run:   ./verify_fourstep      (exit 0 iff every length matches a naive DFT)
#include <complex>
#include <vector>
#include <cstdio>
#include <cmath>
#include <algorithm>
using cff = std::complex<float>;
using cf  = std::complex<double>;
static const double PI = 3.14159265358979323846;

static std::vector<int> factorize(int n) {
    std::vector<int> f;
    for (int r : {7, 5, 4, 3, 2}) while (n % r == 0) { f.push_back(r); n /= r; }
    return (n == 1) ? f : std::vector<int>{};
}

// Same Stockham the GPU sub-FFT runs, but on a threadgroup-sized buffer.
static void stockham_small(cff* a, int n, bool inv) {
    std::vector<int> rad = factorize(n);
    std::vector<cff> buf((size_t)n);
    cff* x = a; cff* y = buf.data();
    int Ns = 1;
    for (int R : rad) {
        const int NR = n / R;
        for (int t = 0; t < NR; ++t) {
            const int j = t % Ns;
            const double ang = (inv ? 2.0 : -2.0) * PI * (double)j / (double)(Ns * R);
            cff v[8], tv[8];
            for (int r = 0; r < R; ++r) {
                double b = ang * r;
                v[r] = x[t + r * NR] * cff((float)std::cos(b), (float)std::sin(b));
            }
            for (int i = 0; i < R; ++i) tv[i] = v[i];
            for (int k = 0; k < R; ++k) {
                cff s(0.f, 0.f);
                for (int m = 0; m < R; ++m) {
                    double b = (inv ? 2.0 : -2.0) * PI * (double)((k * m) % R) / (double)R;
                    s += tv[m] * cff((float)std::cos(b), (float)std::sin(b));
                }
                v[k] = s;
            }
            const int base = (t / Ns) * Ns * R + j;
            for (int r = 0; r < R; ++r) y[base + r * Ns] = v[r];
        }
        std::swap(x, y);
        Ns *= R;
    }
    if (x != a) for (int i = 0; i < n; ++i) a[i] = x[i];
}

// Split N into N1*N2 with both factorable and both within the on-chip budget.
static bool split(int N, int& N1, int& N2, int maxsub) {
    int best = -1;
    for (int d = 2; d * d <= N; ++d) {
        if (N % d) continue;
        int a = d, b = N / d;
        if (a > maxsub || b > maxsub) continue;
        if (factorize(a).empty() || factorize(b).empty()) continue;
        if (best < 0 || std::abs(a - b) < std::abs(N1 - N2)) { N1 = a; N2 = b; best = d; }
    }
    return best > 0;
}

static void four_step(std::vector<cff>& x, int N, bool inv, int N1, int N2) {
    std::vector<cff> a((size_t)N);
    std::vector<cff> t((size_t)std::max(N1, N2));
    const double sgn = inv ? 2.0 : -2.0;
    // Pass A: for each n2, FFT over n1 of x[n1*N2 + n2]; twiddle; store a[k1*N2+n2].
    for (int n2 = 0; n2 < N2; ++n2) {
        for (int n1 = 0; n1 < N1; ++n1) t[n1] = x[(size_t)n1 * N2 + n2];
        stockham_small(t.data(), N1, inv);
        for (int k1 = 0; k1 < N1; ++k1) {
            double ang = sgn * PI * (double)((long long)n2 * k1 % N) / (double)N;
            a[(size_t)k1 * N2 + n2] = t[k1] * cff((float)std::cos(ang), (float)std::sin(ang));
        }
    }
    // Pass B: for each k1, FFT over n2 of a[k1*N2 + n2]; store X[k2*N1 + k1].
    for (int k1 = 0; k1 < N1; ++k1) {
        for (int n2 = 0; n2 < N2; ++n2) t[n2] = a[(size_t)k1 * N2 + n2];
        stockham_small(t.data(), N2, inv);
        for (int k2 = 0; k2 < N2; ++k2) x[(size_t)k2 * N1 + k1] = t[k2];
    }
    if (inv) for (int i = 0; i < N; ++i) x[i] /= (float)N;
}

static void naive(const std::vector<cff>& in, std::vector<cf>& out, int N, bool inv) {
    out.assign(N, cf(0, 0));
    for (int k = 0; k < N; ++k) {
        cf s(0, 0);
        for (int n = 0; n < N; ++n) {
            double ang = (inv ? 2.0 : -2.0) * PI * (double)n * (double)k / (double)N;
            s += cf(in[n].real(), in[n].imag()) * cf(std::cos(ang), std::sin(ang));
        }
        out[k] = inv ? s / (double)N : s;
    }
}

static bool check(int N, bool inv, int maxsub) {
    int N1 = 0, N2 = 0;
    if (!split(N, N1, N2, maxsub)) { std::printf("  N=%-6d no split\n", N); return false; }
    std::vector<cff> a((size_t)N);
    unsigned s = 991u + (unsigned)N;
    auto rnd = [&]() { s = s * 1103515245u + 12345u; return (float)((s >> 16) & 0x7FFF) / 32767.f - 0.5f; };
    for (int i = 0; i < N; ++i) a[i] = cff(rnd(), rnd());
    std::vector<cf> ref; naive(a, ref, N, inv);
    std::vector<cff> got = a; four_step(got, N, inv, N1, N2);
    double me = 0, mg = 0;
    for (int i = 0; i < N; ++i) {
        me = std::max(me, std::abs(cf(got[i].real(), got[i].imag()) - ref[i]));
        mg = std::max(mg, std::abs(ref[i]));
    }
    double rel = mg > 0 ? me / mg : me;
    std::printf("  N=%-6d %-7s = %3d x %-3d  max_rel_err=%.3e %s\n",
                N, inv ? "inverse" : "forward", N1, N2, rel, rel < 2e-5 ? "OK" : "*** FAIL ***");
    return rel < 2e-5;
}

int main() {
    std::printf("Four-step FFT vs naive DFT (sub-transforms <= 512 for threadgroup memory)\n");
    bool ok = true;
    for (int N : {12, 36, 63, 64, 105, 144, 252, 504, 1008, 2016}) { ok &= check(N, false, 512); ok &= check(N, true, 512); }
    std::printf("\nPipeline dimensions:\n");
    for (int N : {3024, 4032}) { ok &= check(N, false, 512); ok &= check(N, true, 512); }
    std::printf("\n%s\n", ok ? "ALL PASS" : "FAILURES PRESENT");
    return ok ? 0 : 1;
}
