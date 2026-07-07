#pragma once
//
// Row-parallel for loops. On Apple platforms uses GCD (reuses a worker pool);
// std::thread per call was spawning hundreds of threads during band merge.
//
#include <functional>
#include <algorithm>
#include <thread>
#include <vector>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

namespace hhsr {

inline int resolve_threads(int requested) {
    if (requested > 0) return requested;
    unsigned hc = std::thread::hardware_concurrency();
    return hc == 0 ? 4 : (int)hc;
}

inline void parallel_rows(int rows, int num_threads, const std::function<void(int)>& body) {
    if (rows <= 0) return;
#ifdef __APPLE__
    (void)num_threads;
    dispatch_apply((size_t)rows,
                   dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                   ^(size_t y) { body((int)y); });
#else
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
#endif
}

} // namespace hhsr
