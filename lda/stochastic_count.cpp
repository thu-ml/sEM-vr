#include "stochastic_count.h"
#include <iostream>
#include <cmath>
#include <algorithm>

StochasticCount::StochasticCount(int R, int C, size_t T)
    : R(R), C(C), data(R*C, 0), sum(C, 0), s(1), T(T)
{
}

void StochasticCount::Decay(double step_size) {
    if (step_size == 1) 
        s = 1;
    else
        s *= (1 - step_size);
}

void StochasticCount::Add(int r, int c, double ratio) {
    data[r*C+c] += ratio / s;
    sum[c]      += ratio / s;
}

double StochasticCount::Get(int r, int c) const {
    return data[r*C+c] * s;
}

double StochasticCount::GetSum(int c) const {
    return sum[c] * s;
}

void StochasticCount::Compress() {
    for (auto &d: data)
        d *= s;
    for (auto &d: sum)
        d *= s;
    s = 1;

//    double total_data = 0;
//    for (auto d: data)
//        total_data += d;
//    double total_sum = 0;
//    for (auto s: sum)
//        total_sum += s;
//    std::cout << total_data << ' ' << total_sum << std::endl;
//    if (fabs(total_data - T) > 1e-5)
//        std::cout << "Error data " << total_data << std::endl;
//    if (fabs(total_sum - T) > 1e-5)
//        std::cout << "Error sum "  << total_sum  << std::endl;
}

void StochasticCount::Fill(double v) {
    fill(data.begin(), data.end(), v);
    fill(sum.begin(), sum.end(), v*R);
}

std::ostream& operator << (std::ostream &out, const StochasticCount &m) {
    for (int r = 0; r < m.R; r++) {
        for (int c = 0; c < m.C; c++)
            out << m.Get(r, c) << ' ';
        out << '\n';
    }
    return out;
}