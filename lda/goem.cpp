//
// Created by jianfei on 18-1-23.
//

#include "goem.h"
#include "corpus.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <exception>
#include "gflags/gflags.h"
#include "utils.h"
#include "flags.h"
#include "accumulator.h"
#include "clock.h"
using namespace std;

GOEM::GOEM(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
           int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          z(corpus.D), cdk(corpus.D * K), theta(corpus.D * K),
          cvk(corpus.V * K), next_cvk(corpus.V * K), phi(corpus.V * K)
{
#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++) {
        z[d].resize(corpus.w[d].size());
        for (size_t n = 0; n < corpus.w[d].size(); n++) {
            auto k = generator() % K;
            z[d][n] = k;
            cdk[d*K+k]++;
        }
    }

    std::vector<double> cvk(corpus.V * K);
#pragma omp parallel for schedule(static, 10)
    for (int v = 0; v < corpus.V; v++)
        for (auto d: corpus.d[v]) {
            int k = generator() % K;
            cvk[v * K + k]++;
        }

    Phi(cvk);

    cout << "Initialization Finished" << endl;
}

void GOEM::Phi(std::vector<double> &cvk)
{
    Accumulator<double> ck_buffer(K);
#pragma omp parallel for schedule(static, 10)
    for (int v = 0; v < corpus.V; v++) {
        auto *ck = ck_buffer.Get();
        auto *cv = cvk.data() + v * K;
        for (int k = 0; k < K; k++)
            ck[k] += cv[k];
    }
    auto ck = ck_buffer.Sum();

    std::vector<double> Zv = ck;
    for (int k = 0; k < K; k++)
        Zv[k] = 1.0 / (ck[k] + beta_bar);

#pragma omp parallel for
    for (int v = 0; v < corpus.V; v++) {
        double *ph = phi.data() + v * K;
        double *cv = cvk.data() + v * K;
        for (int k = 0; k < K; k++)
            ph[k] = (cv[k] + beta) * Zv[k];
    }
}

void GOEM::Estimate()
{
    Accumulator<double> prob_buffer(K);
    size_t batch_count = 0;
    double step_size = 0;
    int batch_size = corpus.D / FLAGS_num_batches + 1;

    for (int iter = 0; iter < FLAGS_num_iters; iter++) {
        std::vector<int> perm(corpus.D);
        iota(perm.begin(), perm.end(), 0);
        shuffle(perm.begin(), perm.end(), generator);

        for (int bs = 0; bs < corpus.D; bs += batch_size) {
            fill(next_cvk.begin(), next_cvk.end(), 0);

            int be = min(bs+batch_size, corpus.D);
            double Tscale = (double)corpus.D / (be - bs);
            // E step
#pragma omp parallel for
            for (int id = bs; id < be; id++) {
                int d = perm[id];
                auto *cd = cdk.data() + d * K;
                for (int iiter = 0; iiter < FLAGS_num_inf_iters; iiter++)
                    for (size_t n = 0; n < corpus.w[d].size(); n++) {
                        auto v = corpus.w[d][n];
                        auto k = z[d][n];
                        --cd[k];
                        auto *ph = phi.data() + v * K;

                        double sum = 0;
                        auto *prob = prob_buffer.Get();
                        for (int k = 0; k < K; k++)
                            prob[k] = sum += (cd[k] + alpha) * ph[k];

                        double u = u01(generator) * sum;
                        k = 0;
                        while (k < K-1 && prob[k]<u) k++;

                        z[d][n] = k;
                        ++cd[k];
                    }
                auto *th = theta.data() + d * K;
                double scale = 1.0 / (corpus.w[d].size() + alpha_bar);
                for (int k = 0; k < K; k++)
                    th[k] = (cd[k] + alpha) * scale;
            }
            for (int id = bs; id < be; id++) {
                int d = perm[id];
                for (size_t n = 0; n < corpus.w[d].size(); n++) {
                    auto v = corpus.w[d][n];
                    auto k = z[d][n];
                    next_cvk[v * K + k] += Tscale;
                }
            }

            // M step
            batch_count += 1;
            step_size = GetSchedule(batch_count);
//            step_size = 1;
#pragma omp parallel for
            for (size_t i = 0; i < cvk.size(); i++) cvk[i] = (1-step_size)*cvk[i] + step_size*next_cvk[i];

            Phi(cvk);

            Eval(iter, bs/batch_size, theta, phi);
        }

        Eval(iter, -1, theta, phi);
    }
}
