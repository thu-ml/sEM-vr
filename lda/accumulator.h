//
// Created by jchen on 19/03/18.
//

#ifndef DSLDA_ACCUMULATOR_H
#define DSLDA_ACCUMULATOR_H

#include <vector>
#include <omp.h>

template <class TData>
struct Accumulator {
    int T, N;
    std::vector<TData> data;

    Accumulator(int Length) : T(omp_get_max_threads()), N(Length+64), data(T*N) {}

    TData* Get() { return data.data() + N * omp_get_thread_num(); }

    std::vector<TData> Sum() {
        for (int t = 1; t < T; t++)
            for (int n = 0; n < N; n++)
                data[n] += data[t*N+n];
        return std::vector<TData>(data.begin(), data.begin()+N);
    }
};

template <class TData>
struct ScalarAccumulator {
    int T;
    std::vector<TData> data;

    ScalarAccumulator() : T(omp_get_max_threads()), data(T * 64) {}

    void Inc(TData d) { data[omp_get_thread_num() * 64] += d; }

    TData Sum() {
        TData result = 0;
        for (auto d: data) result += d;
        return result;
    }
};

#endif //DSLDA_ACCUMULATOR_H
