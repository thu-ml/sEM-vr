//
// Created by jianfei on 2018/1/24.
//

#ifndef DSLDA_UTILS_H
#define DSLDA_UTILS_H

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <memory>
#include <omp.h>
#include "xorshift.h"

extern std::uniform_real_distribution<double> u01;
extern thread_local std::mt19937 generator;

double sqr(double x);

template <class T>
void partially_shuffle(int N, std::vector<T> &data) {
    if (N > data.size())
        throw std::runtime_error("Requested shuffle size is too large " + std::to_string(N) + " " + std::to_string(data.size()));
    for (int i = 0; i < N; i++) {
        // Randomly swap i and [i, N)
        int t = generator() % (data.size() - i) + i;
        std::swap(data[i], data[t]);
    }
}

struct Shuffler {
    std::vector<std::unique_ptr<xorshift> > generators;

    Shuffler() {
        for (int i = 0; i < omp_get_max_threads(); i++)
            generators.emplace_back(new xorshift());
        for (int i = 0; i < omp_get_max_threads(); i++)
            generators[i]->seed(generator(), generator());
    }

    template <class T>
    void Sample(const std::vector<T> &arr, std::vector<T> &result, int num_batches, int batch_size) {
        if (arr.size() > 2147483647)
            throw std::runtime_error("Shuffler: dataset is too large!");

        int M = arr.size();
        size_t N = (size_t)num_batches * batch_size;
        result.resize(N);
#pragma omp parallel
        {
            auto *gen = generators[omp_get_thread_num()].get();
#pragma omp for schedule(static, 100000)
            for (size_t i = 0; i < N; i++) {
                auto idx = (*gen)() % M;
                result[i] = arr[idx];
            }
        };
    }
};

template<class T>
void parallel_copy(std::vector<T> &dest, std::vector<T> &src) {
    dest.resize(src.size());
#pragma omp parallel for
    for (size_t i = 0; i < src.size(); i++) dest[i] = src[i];
}

template<class T>
void parallel_zero(std::vector<T> &dest) {
#pragma omp parallel for
    for (size_t i = 0; i < dest.size(); i++) dest[i] = 0;
}

double digamma(double x);

template<class T>
void print(T* arr, int N) {
    for (int i = 0; i < N; i++)
        std::cout << arr[i] << ' ';
    std::cout << std::endl;
}

template <class T>
void SaveMatrix(const std::vector<T> &mat, int R, int C, const std::string &str) {
    std::ofstream fout(str);
    for (int r = 0; r < R; r++) {
        auto *row = mat.data() + r * C;
        for (int c = 0; c < C; c++)
            fout << row[c] << ' ';
        fout << std::endl;
    }
}

#endif //DSLDA_UTILS_H
