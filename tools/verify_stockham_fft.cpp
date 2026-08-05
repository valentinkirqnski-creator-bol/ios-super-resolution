// Host-side verification of the mixed-radix Stockham FFT implemented by
// fft_stockham_stage in HHSRKernels.metal.
//
// The Metal kernel is a direct transcription of the index math below, so this
// checks the algorithm the GPU actually runs: stage twiddle, radix-R DFT,
// input gather x[t + r*NR], and output scatter y[(t/Ns)*Ns*R + j + r*Ns].
//
// Build: g++ -std=c++17 -O2 tools/verify_stockham_fft.cpp -o verify_stockham
// Run:   ./verify_stockham        (exit 0 iff every length matches a naive DFT)
#include <complex>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstdlib>
using cf = std::complex<double>;
using cff = std::complex<float>;
static const double PI = 3.14159265358979323846;

// Factor n into radices from {7,5,4,3,2}; empty if any other prime remains.
static std::vector<int> factorize(int n) {
    std::vector<int> f;
    for (int r : {7, 5, 4, 3, 2}) while (n % r == 0) { f.push_back(r); n /= r; }
    if (n != 1) return {};
    return f;
}

static void dft_r(cff* v, int R, bool inv) {
    cff t[8];
    for (int i = 0; i < R; ++i) t[i] = v[i];
    for (int k = 0; k < R; ++k) {
        cff s(0.f, 0.f);
        for (int m = 0; m < R; ++m) {
            double a = (inv ? 2.0 : -2.0) * PI * (double)((k * m) % R) / (double)R;
            s += t[m] * cff((float)std::cos(a), (float)std::sin(a));
        }
        v[k] = s;
    }
}

// Stockham autosort: no digit reversal, ping-pong between two buffers.
static void stockham(std::vector<cff>& data, int N, bool inv) {
    std::vector<int> rad = factorize(N);
    if (rad.empty()) { std::printf("  (unfactorable)\n"); return; }
    std::vector<cff> tmp(N);
    cff* x = data.data();
    cff* y = tmp.data();
    int Ns = 1;
    for (int R : rad) {
        const int NR = N / R;
        for (int t = 0; t < NR; ++t) {
            const int j = t % Ns;
            const double ang = (inv ? 2.0 : -2.0) * PI * (double)j / (double)(Ns * R);
            cff v[8];
            for (int r = 0; r < R; ++r) {
                double a = ang * r;
                v[r] = x[t + r * NR] * cff((float)std::cos(a), (float)std::sin(a));
            }
            dft_r(v, R, inv);
            const int base = (t / Ns) * Ns * R + j;
            for (int r = 0; r < R; ++r) y[base + r * Ns] = v[r];
        }
        std::swap(x, y);
        Ns *= R;
    }
    if (x != data.data()) for (int i = 0; i < N; ++i) data[i] = x[i];
    if (inv) for (int i = 0; i < N; ++i) data[i] /= (float)N;
}

static void naive_dft(const std::vector<cff>& in, std::vector<cf>& out, int N, bool inv) {
    out.assign(N, cf(0, 0));
    for (int k = 0; k < N; ++k) {
        cf s(0, 0);
        for (int n = 0; n < N; ++n) {
            double a = (inv ? 2.0 : -2.0) * PI * (double)n * (double)k / (double)N;
            s += cf((double)in[n].real(), (double)in[n].imag()) * cf(std::cos(a), std::sin(a));
        }
        out[k] = inv ? s / (double)N : s;
    }
}

static bool check(int N, bool inv) {
    std::vector<cff> a((size_t)N);
    unsigned seed = 12345u + (unsigned)N;
    auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return (float)((seed >> 16) & 0x7FFF) / 32767.f - 0.5f; };
    for (int i = 0; i < N; ++i) a[i] = cff(rnd(), rnd());
    std::vector<cf> ref;
    naive_dft(a, ref, N, inv);
    std::vector<cff> got = a;
    stockham(got, N, inv);
    double maxe = 0, mag = 0;
    for (int i = 0; i < N; ++i) {
        maxe = std::max(maxe, std::abs(cf(got[i].real(), got[i].imag()) - ref[i]));
        mag = std::max(mag, std::abs(ref[i]));
    }
    double rel = mag > 0 ? maxe / mag : maxe;
    std::printf("  N=%-6d %-7s radices=", N, inv ? "inverse" : "forward");
    for (int r : factorize(N)) std::printf("%d ", r);
    std::printf(" max_rel_err=%.3e %s\n", rel, rel < 2e-5 ? "OK" : "*** FAIL ***");
    return rel < 2e-5;
}

int main() {
    std::printf("Mixed-radix Stockham vs naive DFT (float32 transform, float64 reference)\n");
    bool ok = true;
    for (int N : {2, 3, 4, 5, 7, 8, 9, 12, 16, 21, 35, 63, 105, 120, 189, 252, 504, 1024}) {
        ok &= check(N, false);
        ok &= check(N, true);
    }
    std::printf("\nActual pipeline dimensions:\n");
    for (int N : {3024, 4032}) { ok &= check(N, false); ok &= check(N, true); }
    std::printf("\n%s\n", ok ? "ALL PASS" : "FAILURES PRESENT");
    return ok ? 0 : 1;
}
