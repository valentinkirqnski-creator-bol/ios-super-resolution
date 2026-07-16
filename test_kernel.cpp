#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>

int main() {
    float factor = 2.0f;
    float sigma = factor * 0.5f; // 1.0
    int radius = (int)(4.0f * sigma + 0.5f); // 4
    int ksize = 2 * radius + 1;

    std::vector<float> kernel(ksize);
    float ksum = 0.f;
    for (int i = -radius; i <= radius; ++i) {
        float v = std::exp(-0.5f * (float)(i * i) / (sigma * sigma));
        kernel[i + radius] = v;
        ksum += v;
    }
    for (auto& v : kernel) v /= ksum;

    std::cout << "C++ Kernel:\n";
    std::cout << std::fixed << std::setprecision(8);
    float final_sum = 0.f;
    for (auto v : kernel) {
        std::cout << v << "\n";
        final_sum += v;
    }
    std::cout << "Sum: " << final_sum << "\n";
    return 0;
}
