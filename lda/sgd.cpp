//
// Created by 陈键飞 on 2018/4/17.
//

#include "sgd.h"
#include "corpus.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <exception>
#include <iostream>
#include <queue>
#include <atomic>
#include <omp.h>
#include "gflags/gflags.h"
#include "utils.h"
#include "flags.h"
#include "clock.h"
#include "accumulator.h"
using namespace std;

SGD::SGD(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
             int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          theta(corpus.D * K), phi(corpus.V * K)
{
    for (int d = 0; d < corpus.D; d++)
        for (auto v: corpus.w[d]) {
            int k = generator() % K;

            theta[d * K + k]++;
            phi[v * K + k]++;
        }
    for (int d = 0; d < corpus.D; d++) {
        double sum = 0;
        double *th = theta.data() + d * K;
        for (int k = 0; k < K; k++)
            sum += th[k] += 1;
        sum = 1.0 / sum;
        for (int k = 0; k < K; k++)
            th[k] *= sum;
    }
    for (int k = 0; k < K; k++) {
        double sum = 0;
        for (int v = 0; v < corpus.V; v++)
            sum += phi[v*K+k] += 1;
        sum = 1.0 / sum;
        for (int v = 0; v < corpus.V; v++)
            phi[v*K+k] *= sum;
    }

    batch_size = corpus.D / FLAGS_num_batches + 1;
    cout << "Initialization Finished " << BatchPerplexity() << endl;
}

double SGD::InnerProduct(double *th, double *ph) {
    double result = 0;
    for (int k = 0; k < K; k++) result += th[k] * ph[k];
    return result;
}

void SGD::Estimate()
{
    step_size = FLAGS_max_step;
    D_step_size = FLAGS_max_step;
    int iter_cnt = 0;

    for (int iter = 0; iter < FLAGS_num_iters; iter++) {
        step_size_seq.clear();
        this->iter = iter;

        Clock clk;

        clk.tic();
        // Stochastic Inference++
        Clock clk2;
        double sample1_time = 0;
        double sample2_time = 0;
        double init_time = 0;
        double next_time = 0;
        double thetaphi_time = 0;
        double shuffle_time = 0;
        ScalarAccumulator<size_t> num_samples_acc;

        std::vector<int> perm(corpus.D);
        iota(perm.begin(), perm.end(), 0);
        shuffle(perm.begin(), perm.end(), generator);

#pragma omp parallel for
        for (int d = 0; d < corpus.D; d++)
            shuffle(corpus.w[d].begin(), corpus.w[d].end(), generator);

        double cv_time = 0;

        for (int iiter = 0; iiter < FLAGS_num_batches; iiter++) {
            clk2.tic();

            iter_cnt++;

            if (FLAGS_vr) {

            } else {
                step_size = D_step_size = GetSchedule(iter_cnt);
            }

            int d_start = iiter * batch_size;
            int d_end   = min((iiter + 1) * batch_size, corpus.D);

            double T_scale = (double)corpus.D / (d_end - d_start);

            // Initialize buffer
            std::vector<int> offsets(d_end - d_start);
            v_indices.resize(corpus.V);
            for (auto &l: v_indices) l.clear();
            init_time += clk2.toc(); clk2.tic();

            // TODO Here we need a parallel radix sort for the (v, t) pairs
            int sum = 0;
            for (int id = d_start; id < d_end; id++) {
                offsets[id - d_start] = sum;
                int d = perm[id];
                sum += corpus.w[d].size();
            }
            buffer.resize(sum * K);

            for (int id = d_start; id < d_end; id++) {
                int sum = offsets[id - d_start];
                int d = perm[id];
                for (auto v: corpus.w[d])
                    v_indices[v].push_back(sum++);
            }
            shuffle_time += clk2.toc(); clk2.tic();

            #pragma omp parallel for
            for (int id = d_start; id < d_end; id++) {
                int d = perm[id];

//                std::vector<double> delta_theta(K);

                auto *th = theta.data() + d * K;
                int Len = corpus.w[d].size();
                auto *doc_buf = buffer.data() + offsets[id - d_start] * K;

                if (FLAGS_vr) {
                } else {
                    for (auto v: corpus.w[d]) {
                        auto *ph  = phi.data() + v * K;
                        auto ip = InnerProduct(th, ph) + 1e-8;
                        auto ss = Len / ip;

                        double sum = 0;
                        for (int k = 0; k < K; k++) {
                            th[k] = max(th[k] + D_step_size * (ss * ph[k] + alpha / th[k]), 0.0);
                            sum += th[k];
                        }
                        sum = (1.0 - FLAGS_min_param * K) / sum;
                        for (int k = 0; k < K; k++)
                            th[k] = FLAGS_min_param + sum * th[k];
                    }
                    for (int vid = 0; vid < corpus.w[d].size(); vid++) {
                        int v = corpus.w[d][vid];
                        auto *ph  = phi.data() + v * K;
                        auto ip = InnerProduct(th, ph) + 1e-8;
                        auto ss = step_size * T_scale / ip;

                        auto *buf = doc_buf + vid * K; // TODO we do not actually need buf...
                        for (int k = 0; k < K; k++)
                            buf[k] = ss * th[k];
                    }
                }
            }
            sample1_time += clk2.toc(); clk2.tic();

            Accumulator<double> ck_accumulator(K);
            #pragma omp parallel for schedule(static, 10)
            for (int v = 0; v < corpus.V; v++) {
                auto *ph = phi.data() + v * K;
                for (int k = 0; k < K; k++)
                    ph[k] += step_size * beta / ph[k];
                for (auto t: v_indices[v]) {
                    auto *buf = buffer.data() + t * K;
                    for (int k = 0; k < K; k++)
                        ph[k] += buf[k];
                }
                auto *ck = ck_accumulator.Get();
                for (int k = 0; k < K; k++) {
                    ph[k] = max(ph[k] + step_size * beta, 0.0);
                    ck[k] += ph[k];
                }
//                for (int k = 0; k < K; k++)
//                    cout << ph[k] << ' ';
//                cout << corpus.d[v].size() << ' ' << v << endl;
            }
//            exit(0);
            auto ck = ck_accumulator.Sum();
            for (int k = 0; k < K; k++)
                ck[k] = (1.0 - FLAGS_min_param * corpus.V) / ck[k];

            #pragma omp parallel for schedule(static, 10)
            for (int v = 0; v < corpus.V; v++) {
                auto *ph = phi.data() + v * K;
                for (int k = 0; k < K; k++)
                    ph[k] = FLAGS_min_param + ck[k] * ph[k];
            }
            sample2_time += clk2.toc(); clk2.tic();
        }
        cout << "Sample time " << sample1_time << ' ' << sample2_time
             << " misc time SINT " << shuffle_time << ' ' << init_time << ' ' << next_time << ' ' << thetaphi_time << endl;
        auto eval_time = clk.toc();

        clk.tic();
        double ll = BatchLogLikelihood();
        double perplexity = exp(-ll / corpus.T);
        double tperplexity = BaseLDA::Inference(phi);
        auto test_time = clk.toc();

        printf("Log likelihood = %.20f\n", ll);

        cout << "\e[0;33mIteration " << iter
             << " step size = " << step_size << ' ' << D_step_size << ' '
             << " perplexity = " << perplexity
             << " tperplexity = " << tperplexity << " test time = " << test_time
             << " Times " << ' ' << cv_time << ' ' << eval_time << "\e[0;0m" << endl;
    }
}

double SGD::BatchLogLikelihood()
{
    ScalarAccumulator<double> ll_buffer;
#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++)
        for (size_t n = 0; n < corpus.w[d].size(); n++) {
            auto v = corpus.w[d][n];
            double l = 0;
            for (int k = 0; k < K; k++)
                l += theta[d*K + k] * phi[v*K + k];
            l   = log(l);
            ll_buffer.Inc(l);
        }

    return ll_buffer.Sum();
}

double SGD::BatchPerplexity()
{
    double ll = BatchLogLikelihood();
    double perplexity = exp(-ll / corpus.T);
    return perplexity;
}
