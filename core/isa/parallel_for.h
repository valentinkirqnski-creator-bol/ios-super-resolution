#pragma once
// Trivial data-parallel helper used to spread the embarrassingly-parallel
// per-tile/per-pixel/per-frame loops across all CPU cores -- this project
// has no GPU to lean on, so plain std::thread parallelism is the
// difference between a batch run measured in minutes vs. tens of minutes.
#include <thread>
#include <vector>
#include <algorithm>

namespace isacpu {

// Runs fn(i) for i in [0,n), split into std::thread::hardware_concurrency()
// contiguous chunks across worker threads. Falls back to a plain serial
// loop for small n or a single-core machine.
template <typename Fn>
void parallel_for(int n, Fn fn) {
    if (n <= 0) return;
    unsigned hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 1;
    int num_threads = (int)std::min<unsigned>(hw, (unsigned)n);
    if (num_threads <= 1) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    int chunk = (n + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk;
        int end = std::min(n, start + chunk);
        if (start >= end) continue;
        threads.emplace_back([start, end, &fn]() {
            for (int i = start; i < end; ++i) fn(i);
        });
    }
    for (auto& th : threads) th.join();
}

}  // namespace isacpu
