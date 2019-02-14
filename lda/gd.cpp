//
// Created by 陈键飞 on 2018/4/17.
//

#include "gd.h"
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

GD::GD(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
             int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          theta(corpus.D * K), phi(corpus.V * K),
          delta_theta(corpus.D * K), delta_phi(corpus.V * K)
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

double GD::InnerProduct(double *th, double *ph) {
    double result = 0;
    for (int k = 0; k < K; k++) result += th[k] * ph[k];
    return result;
}

void GD::Estimate()
{
    step_size = FLAGS_max_step;
    D_step_size = FLAGS_max_step;

    for (int iter = 0; iter < FLAGS_num_iters; iter++) {
        this->iter = iter;

        fill(delta_theta.begin(), delta_theta.end(), 0);
        fill(delta_phi.begin(), delta_phi.end(), 0);
        for (int d = 0; d < corpus.D; d++) {
            for (auto v: corpus.w[d]) {
                auto *th = theta.data() + d * K;
                auto *dth = delta_theta.data() + d * K;
                auto *ph = phi.data() + v * K;
                auto *dph = delta_phi.data() + v * K;

                double score = InnerProduct(th, ph) + 1e-8;
                double weight = 1.0 / score;
                for (int k = 0; k < K; k++) {
                    dth[k] += weight * ph[k];
                    dph[k] += weight * th[k];
                }
            }
        }
        for (int d = 0; d < corpus.D; d++) {
            auto *th = theta.data() + d * K;
            auto *dth = delta_theta.data() + d * K;
            double sum = 0;
            for (int k = 0; k < K; k++) {
                dth[k] += alpha / (th[d] + 1e-8);
                sum += th[k] *= exp(step_size * dth[k]);
            }
            sum = 1.0 / sum;
            for (int k = 0; k < K; k++)
                th[k] *= sum;
        }
        std::vector<double> ck(K);
        for (int v = 0; v < corpus.V; v++) {
            auto *ph = phi.data() + v * K;
            auto *dph = delta_phi.data() + v * K;
            for (int k = 0; k < K; k++) {
                dph[k] += beta / (ph[k] + 1e-8);
                ph[k] *= exp(step_size * dph[k]);
                ck[k] += ph[k];
            }
        }
        for (int k = 0; k < K; k++)
            ck[k] = 1.0 / ck[k];
        for (int v = 0; v < corpus.V; v++) {
            auto *ph = phi.data() + v * K;
            for (int k = 0; k < K; k++)
                ph[k] *= ck[k];
        }

        double ll = BatchLogLikelihood();
        double perplexity = exp(-ll / corpus.T);
        double tperplexity = BaseLDA::Inference(phi);

        printf("Log likelihood = %.20f\n", ll);

        cout << "\e[0;33mIteration " << iter
             << " step size = " << step_size << ' ' << D_step_size << ' '
             << " perplexity = " << perplexity
             << " tperplexity = " << tperplexity << "\e[0;0m" << endl;
    }
}

double GD::BatchLogLikelihood()
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

double GD::BatchPerplexity()
{
    double ll = BatchLogLikelihood();
    double perplexity = exp(-ll / corpus.T);
    return perplexity;
}
