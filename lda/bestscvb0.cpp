//
// Created by jianfei on 18-1-23.
//

#include "bestscvb0.h"
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
using namespace std;

BestSCVB0::BestSCVB0(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
           int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          theta(corpus.D * K),
          cvk(corpus.V * K), next_cvk(corpus.V * K), phi(corpus.V * K)
{
    std::vector<double> cdk(corpus.D * K);
#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++) {
        for (auto v: corpus.w[d]) {
            int k = generator() % K;
            cdk[d * K + k]++;
        }
        Theta(cdk.data() + d * K, theta.data() + d * K, corpus.w[d].size());
    }
#pragma omp parallel for schedule(static, 10)
    for (int v = 0; v < corpus.V; v++)
        for (auto d: corpus.d[v]) {
            int k = generator() % K;
            cvk[v * K + k]++;
        }

    Phi(cvk);

    cout << "Initialization Finished" << endl;
}

void BestSCVB0::Theta(double *cdk, double *theta, int N) {
    double Z = 1.0 / (N + alpha_bar);
    for (int k = 0; k < K; k++)
        theta[k] = (cdk[k] + alpha) * Z;
}

void BestSCVB0::Phi(std::vector<double> &cvk)
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

void BestSCVB0::Estimate()
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
            Accumulator<double> cd_buffer(K);
#pragma omp parallel for schedule(dynamic, 10)
            for (int id = bs; id < be; id++) {
                int d = perm[id];
                auto *cd = cd_buffer.Get();
                auto *th = theta.data() + d * K;
                for (int iiter = 0; iiter < 20; iiter++) {
                    fill(cd, cd+K, 0);
                    for (size_t n = 0; n < corpus.w[d].size(); n++) {
                        auto v = corpus.w[d][n];
                        auto *ph = phi.data() + v * K;

                        double sum = 0;
                        auto *prob = prob_buffer.Get();
                        for (int k = 0; k < K; k++)
                            sum += prob[k] = th[k] * ph[k];
                        sum = 1.0 / sum;
                        for (int k = 0; k < K; k++)
                            cd[k] += prob[k] * sum;
                    }
                    double scale = 1.0 / (corpus.w[d].size() + alpha_bar);
                    for (int k = 0; k < K; k++)
                        th[k] = (cd[k] + alpha) * scale;
                }
            }
            // M step
            for (int id = bs; id < be; id++) {
                int d = perm[id];
                auto *th = theta.data() + d * K;
                for (size_t n = 0; n < corpus.w[d].size(); n++) {
                    auto v = corpus.w[d][n];
                    auto *ph = phi.data() + v * K;
                    auto *cv = next_cvk.data() + v * K;

                    double sum = 0;
                    auto *prob = prob_buffer.Get();
                    for (int k = 0; k < K; k++)
                        sum += prob[k] = th[k] * ph[k];
                    sum = Tscale / sum;
                    for (int k = 0; k < K; k++)
                        cv[k] += prob[k] * sum;
                }
            }

            // M step
            batch_count += 1;
            step_size = GetSchedule(batch_count);
#pragma omp parallel for
            for (size_t i = 0; i < cvk.size(); i++) cvk[i] = (1-step_size)*cvk[i] + step_size*next_cvk[i];

            Phi(cvk);
        }

        // Compute perplexity
        double perplexity = BatchPerplexity();
        double tperplexity = BaseLDA::Inference(phi);
        cout << "Iteration " << iter
             << " step size = " << step_size
             << " perplexity = " << perplexity
             << " tperplexity = " << tperplexity << endl;
    }
}

double BestSCVB0::BatchLogLikelihood()
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

double BestSCVB0::BatchPerplexity()
{
    double ll = BatchLogLikelihood();
    double perplexity = exp(-ll / corpus.T);
    return perplexity;
}
