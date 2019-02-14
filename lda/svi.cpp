//
// Created by jianfei on 18-1-23.
//

#include "svi.h"
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

SVI::SVI(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
           int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          theta(corpus.D * K), gamma(corpus.D * K),
          cvk(corpus.V * K), next_cvk(corpus.V * K), phi(corpus.V * K), smoothed_cvk(corpus.V * K)
{
#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++) {
        auto *gam = gamma.data() + d * K;
        fill(gam, gam+K, alpha);
        for (size_t n = 0; n < corpus.w[d].size(); n++) {
            auto k = generator() % K;
            gam[k]++;
        }
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

void SVI::Phi(std::vector<double> &cvk)
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
        Zv[k] = digamma(ck[k] + beta_bar);

    Accumulator<double> ck2_buffer(K);
#pragma omp parallel for
    for (int v = 0; v < corpus.V; v++) {
        double *ph = phi.data() + v * K;
        double *cv = cvk.data() + v * K;
        auto *ck2 = ck2_buffer.Get();
        for (int k = 0; k < K; k++)
            ck2[k] += ph[k] = exp(digamma(cv[k] + beta) - Zv[k]);
    }
    auto Zv2 = ck2_buffer.Sum();
    for (int k = 0; k < K; k++)
        Zv2[k] = 1.0 / Zv2[k];
#pragma omp parallel for
    for (int v = 0; v < corpus.V; v++) {
        double *ph = phi.data() + v * K;
        for (int k = 0; k < K; k++)
            ph[k] *= Zv2[k];
    }
}

void SVI::Estimate()
{
    Accumulator<double> prob_buffer(K);
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

            // Initialize bu    ffer
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
                auto *gam = gamma.data() + d * K;
                for (int iiter = 0; iiter < 6; iiter++) {
                    fill(cd, cd+K, alpha);
                    for (size_t n = 0; n < corpus.w[d].size(); n++) {
                        auto v = corpus.w[d][n];
                        auto *ph = phi.data() + v * K;

                        double sum = 0;
                        auto *prob = prob_buffer.Get();
                        for (int k = 0; k < K; k++)
                            sum += prob[k] = ph[k] * exp(digamma(gam[k]));

                        sum = 1.0 / sum;
                        for (int k = 0; k < K; k++)
                            cd[k] += sum * prob[k];
                    }
                    copy(cd, cd+K, gam);
                }
                double scale = 1.0 / (corpus.w[d].size() + alpha_bar);
                for (int k = 0; k < K; k++)
                    th[k] = gam[k] * scale;

                auto *doc_buf = buffer.data() + offsets[id-bs] * K;
                for (size_t n = 0; n < corpus.w[d].size(); n++) {
                    auto v = corpus.w[d][n];
                    auto *ph = phi.data() + v * K;

                    double sum = 0;
                    auto *prob = prob_buffer.Get();
                    for (int k = 0; k < K; k++)
                        sum += prob[k] = ph[k] * exp(digamma(gam[k]));

                    sum = Tscale / sum;
                    auto *buf = doc_buf + n * K;
                    for (int k = 0; k < K; k++)
                        buf[k] = sum * prob[k];
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

            // Smooth step
            ss.push_back(next_cvk);
#pragma omp parallel for
            for (size_t i = 0; i < next_cvk.size(); i++) smoothed_cvk[i] += next_cvk[i];
            if (ss.size() > FLAGS_ssvi_ws) {
                auto &prev_cvk = ss.front();
#pragma omp parallel for
                for (size_t i = 0; i < next_cvk.size(); i++) smoothed_cvk[i] -= prev_cvk[i];
                ss.pop_front();
            }
            double ss_scale = 1.0 / ss.size();

            // M step
            batch_count += 1;
            step_size = GetSchedule(batch_count);
//            step_size = 1;
#pragma omp parallel for
            for (size_t i = 0; i < cvk.size(); i++) cvk[i] = (1-step_size)*cvk[i] + step_size*smoothed_cvk[i]*ss_scale;

            Phi(cvk);
#pragma omp parallel for
            for (int v = 0; v < corpus.V; v++)
                for (int k = 0; k < K; k++) {
                    if (phi[v*K+k] < 0)
                        throw runtime_error("negative phi");
                    if (std::isnan(phi[v*K+k]))
                        throw runtime_error("phi is nan");
                }

            Eval(iter, bs/batch_size, theta, phi);
        }

        Eval(iter, -1, theta, phi);
    }
}
