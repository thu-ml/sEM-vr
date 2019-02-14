//
// Created by jianfei on 18-1-23.
//

#include "cvb0oem.h"
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

CVB0OEM::CVB0OEM(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
           int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          cdk(corpus.D * K), theta(corpus.D * K),
          cvk(corpus.V * K), next_cvk(corpus.V * K), phi(corpus.V * K)
{
#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++) {
        auto *th = theta.data() + d * K;
        fill(th, th+K, alpha);
        for (size_t n = 0; n < corpus.w[d].size(); n++) {
            auto k = generator() % K;
            th[k]++;
        }
        double scale = 1.0 / (corpus.w[d].size() + alpha_bar);
        for (int k = 0; k < K; k++)
            th[k] *= scale;
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

void CVB0OEM::Phi(std::vector<double> &cvk)
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

void CVB0OEM::Estimate()
{
    Accumulator<double> cd_buffer(K);
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

            // Initialize buffer
            std::vector<int> offsets(be - bs);
            v_indices.resize(corpus.V);
            for (auto &l: v_indices) l.clear();

            int sum = 0;
            for (int id = bs; id < be; id++) {
                offsets[id - bs] = sum;
                int d = perm[id];
                sum += corpus.w[d].size();
            }
            buffer.resize(sum * K);

            for (int id = bs; id < be; id++) {
                int sum = offsets[id - bs];
                int d = perm[id];
                for (auto v: corpus.w[d])
                    v_indices[v].push_back(sum++);
            }

            // E step
#pragma omp parallel for
            for (int id = bs; id < be; id++) {
                int d = perm[id];
                auto *cd = cd_buffer.Get();
                auto *th = theta.data() + d * K;
                double scale = 1.0 / (corpus.w[d].size() + alpha_bar);

                auto *doc_buf = buffer.data() + offsets[id-bs] * K;
                // Initialize cd and doc_buf from theta
                fill(cd, cd+K, alpha);
                for (size_t n = 0; n < corpus.w[d].size(); n++) {
                    auto v = corpus.w[d][n];
                    auto *ph = phi.data() + v * K;
                    auto *prob = doc_buf + n * K;
                    double sum = 0;

                    for (int k = 0; k < K; k++)
                        sum += prob[k] = ph[k] * th[k];
                    sum = 1.0 / sum;
                    for (int k = 0; k < K; k++)
                        cd[k] += prob[k] *= sum;
                }
                // CVB0 inference
                for (int iiter = 0; iiter < 5; iiter++) {
                    for (size_t n = 0; n < corpus.w[d].size(); n++) {
                        auto v = corpus.w[d][n];
                        auto *ph = phi.data() + v * K;
                        auto *prob = doc_buf + n * K;
                        double sum = 0;

                        for (int k = 0; k < K; k++)
                            cd[k] -= prob[k];
                        for (int k = 0; k < K; k++)
                            sum += prob[k] = ph[k] * cd[k];
                        sum = 1.0 / sum;
                        for (int k = 0; k < K; k++)
                            cd[k] += prob[k] *= sum;
                    }
                }
                // Estimate theta
                for (int k = 0; k < K; k++)
                    th[k] = cd[k] * scale;
                // Update phi
                for (size_t n = 0; n < corpus.w[d].size(); n++) {
                    auto v = corpus.w[d][n];
                    auto *ph = phi.data() + v * K;
                    auto *prob = doc_buf + n * K;
                    double sum = 0;

                    for (int k = 0; k < K; k++)
                        sum += prob[k] = ph[k] * th[k];
                    sum = Tscale / sum;
                    for (int k = 0; k < K; k++)
                        prob[k] *= sum;
                }
            }

            #pragma omp parallel for schedule(static, 10)
            for (int v = 0; v < corpus.V; v++) {
                auto *cv = next_cvk.data() + v * K;
                for (auto t: v_indices[v]) {
                    auto *buf = buffer.data() + t * K;
                    for (int k = 0; k < K; k++)
                        cv[k] += buf[k];
                }
            }

            // M step
            batch_count += 1;
            step_size = GetSchedule(batch_count);
//            step_size = 1;
#pragma omp parallel for
            for (size_t i = 0; i < cvk.size(); i++) cvk[i] = (1-step_size)*cvk[i] + step_size*next_cvk[i];

            Phi(cvk);
#pragma omp parallel for schedule(static, 10)
            for (int v = 0; v < corpus.V; v++)
                for (int k = 0; k < K; k++) {
                    if (phi[v*K+k] < 0)
                        throw runtime_error("negative phi");
                    if (std::isnan(phi[v*K+k]))
                        throw runtime_error("phi is nan");
                }

            if (FLAGS_test_lag != -1 && batch_count % FLAGS_test_lag == 0) {
                cout << "Testing ";
                cout.flush();
                cout << "perplexity " << BaseLDA::Inference(phi) << endl;
            }
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

double CVB0OEM::BatchLogLikelihood()
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

double CVB0OEM::BatchPerplexity()
{
    double ll = BatchLogLikelihood();
    double perplexity = exp(-ll / corpus.T);
    return perplexity;
}
