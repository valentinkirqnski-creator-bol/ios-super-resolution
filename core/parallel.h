#pragma once
//
// Minimal std::thread parallel-for used to fan the per-pixel kernels across
// the mobile CPU cores. This replaces the CUDA grid launches of the reference
// implementation. A Vulkan compute backend would replace these loops with
// shader dispatches behind the same call sites.
//
#include <thread>
#include <vector>
#include <functional>
#include <algorithm>

namespace hhsr {

inline int resolve_threads(int requested) {
    if (requested > 0) return requested;
    unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 4 : (int)hc;
}

// Splits the row range [0, rows) across worker threads; body(y) runs per row.
inline void parallel_rows(int rows, int num_threads, const std::function<void(int)>& body) {
    int nt = resolve_threads(num_threads);
    if (nt <= 1 || rows <= 1) {
        for (int y = 0; y < rows; ++y) body(y);
        return;
    }
    nt = std::min(nt, rows);
    std::vector<std::thread> pool;
    pool.reserve(nt);
    int chunk = (rows + nt - 1) / nt;
    for (int t = 0; t < nt; ++t) {
        int y0 = t * chunk;
        int y1 = std::min(y0 + chunk, rows);
        if (y0 >= y1) break;
        pool.emplace_back([y0, y1, &body]() {
            for (int y = y0; y < y1; ++y) body(y);
        });
    }
    for (auto& th : pool) th.join();
}

} // namespace hhsr
