//
// Created by jianfei on 18-1-23.
//

#include "rsgd.h"
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

RSGD::RSGD(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
           int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          theta(corpus.D * K), phi(corpus.V * K),
          cdk(corpus.D * K), cvk(corpus.V * K), cvk2(corpus.V * K),
          rtheta(corpus.D * K), rphi(corpus.V * K),
          theta_hist(corpus.D * K), phi_hist(corpus.V * K),
          cv_cdk(corpus.D * K), cv_cvk(corpus.V * K), cv_ck(K)
{
#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++) {
        auto *th = theta.data() + d * K;
        double sum = 0;
        for (int k = 0; k < K; k++)
            sum += th[k] = u01(generator) + alpha;
        sum = 1.0 / sum;
        for (int k = 0; k < K; k++)
            th[k] *= sum;
    }
    Accumulator<double> ck_buffer(K);
#pragma omp parallel for
    for (int v = 0; v < corpus.V; v++) {
        auto *ph = phi.data() + v * K;
        auto *ck = ck_buffer.Get();
        for (int k = 0; k < K; k++)
            ck[k] += ph[k] = (double)corpus.d[v].size() / K + u01(generator) + beta;
    }
    auto ck = ck_buffer.Sum();
    for (int k = 0; k < K; k++)
        ck[k] = 1.0 / ck[k];
#pragma omp parallel for
    for (int v = 0; v < corpus.V; v++) {
        auto *ph = phi.data() + v * K;
        for (int k = 0; k < K; k++)
            ph[k] *= ck[k];
    }
#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++) {
        auto *rth = rtheta.data() + d * K;
        auto *th  = theta.data() + d * K;
        for (int k = 0; k < K; k++)
            rth[k] = log(th[k]);
    }
#pragma omp parallel for
    for (int v = 0; v < corpus.V; v++) {
        auto *rph = rphi.data() + v * K;
        auto *ph  = phi.data() + v * K;
        for (int k = 0; k < K; k++)
            rph[k] = log(ph[k]);
    }

    cout << "Initialization Finished" << endl;
}

void RSGD::Phi() {
    std::vector<double> max_rph(K, -1e100);
#pragma omp parallel
    {
        std::vector<double> mrph(K, -1e100);
#pragma omp for
        for (int v = 0; v < corpus.V; v++) {
            auto *rph = rphi.data() + v * K;
            for (int k = 0; k < K; k++)
                mrph[k] = max(mrph[k], rph[k]);
        }
#pragma omp critical
        {
            for (int k = 0; k < K; k++)
                max_rph[k] = max(max_rph[k], mrph[k]);
        }
    }

    Accumulator<double> ck_acc(K);
#pragma omp parallel for
    for (int v = 0; v < corpus.V; v++) {
        auto *rph = rphi.data() + v * K;
        auto *ph  = phi.data() + v * K;
        auto *ck  = ck_acc.Get();
        for (int k = 0; k < K; k++)
            ck[k] += ph[k] = exp(rph[k] - max_rph[k]);
    }
    auto ck = ck_acc.Sum();
    for (int k = 0; k < K; k++)
        ck[k] = (1.0 - FLAGS_min_param * corpus.V) / ck[k];
#pragma omp parallel for
    for (int v = 0; v < corpus.V; v++) {
        auto *ph  = phi.data() + v * K;
        for (int k = 0; k < K; k++)
            ph[k] = ph[k] * ck[k] + FLAGS_min_param;
    }
}

void RSGD::Estimate()
{
    Accumulator<double> prob_buffer(K);
    Accumulator<double> hist_buffer(K);
    size_t batch_count = 0;
    int batch_size = corpus.D / FLAGS_num_batches + 1;

    step_size = FLAGS_max_step;
    int iter_cnt = 0;
    // Stochastic Theta, batch phi
    for (int iter = 0; iter < FLAGS_num_iters; iter++) {
        bool vr = FLAGS_vr;

        std::vector<int> perm(corpus.D);
        iota(perm.begin(), perm.end(), 0);
        shuffle(perm.begin(), perm.end(), generator);

#pragma omp parallel for
        for (int d = 0; d < corpus.D; d++)
            shuffle(corpus.w[d].begin(), corpus.w[d].end(), generator);

        if (vr) {
            parallel_copy(theta_hist, theta);
            parallel_copy(phi_hist, phi);
            Accumulator<double> cv_cvk_acc(corpus.V * K), cv_ck_acc(K);
            fill(cv_cdk.begin(), cv_cdk.end(), alpha);
#pragma omp parallel for
            for (int d = 0; d < corpus.D; d++) {
                auto *cd = cv_cdk.data() + d * K;
                auto *th = theta.data() + d * K;
                auto *cvk = cv_cvk_acc.Get();
                auto *ck = cv_ck_acc.Get();
                auto *prob = prob_buffer.Get();
                for (auto v: corpus.w[d]) {
                    auto *cv = cvk + v * K;
                    auto *ph = phi.data() + v * K;
                    double sum = 0;
                    for (int k = 0; k < K; k++)
                        sum += prob[k] = th[k] * ph[k];
                    sum = 1.0 / sum;
                    for (int k = 0; k < K; k++) {
                        auto p = prob[k] * sum;
                        cd[k] += p;
                        cv[k] += p;
                        ck[k] += p;
                    }
                }
            }
            cv_cvk = cv_cvk_acc.Sum();
            cv_ck = cv_ck_acc.Sum();
#pragma omp parallel for
            for (size_t i = 0; i < cv_cvk.size(); i++)
                cv_cvk[i] += beta;
            for (int k = 0; k < K; k++)
                cv_ck[k] += beta * corpus.V;
        }

        for (int bs = 0; bs < corpus.D; bs += batch_size) {
            step_size = GetSchedule(++iter_cnt);
            int be = std::min(corpus.D, bs + batch_size);
            auto Tscale = (double)corpus.D / (be - bs);

            // Initialize buffer
            std::vector<int> offsets(be - bs);
            v_indices.resize(corpus.V);
            for (auto &l: v_indices) l.clear();

            // TODO Here we need a parallel radix sort for the (v, t) pairs
            int sum = 0;
            for (int id = bs; id < be; id++) {
                offsets[id - bs] = sum;
                int d = perm[id];
                sum += corpus.w[d].size();
            }
            buffer.resize(sum * K);
            buffer2.resize(sum * K);

            for (int id = bs; id < be; id++) {
                int sum = offsets[id - bs];
                int d = perm[id];
                for (auto v: corpus.w[d])
                    v_indices[v].push_back(sum++);
            }

            // Optimize theta
#pragma omp parallel for
            for (int id = bs; id < be; id++) {
                auto d = perm[id];

                auto *cd_v = cv_cdk.data() + d * K;
                auto *th = theta.data() + d * K;
                auto *rth = rtheta.data() + d * K;
                auto *hth = theta_hist.data() + d * K;
                auto *prob = prob_buffer.Get();
                auto *hist = hist_buffer.Get();
                auto *doc_buf = buffer.data() + offsets[id - bs] * K;
                auto *doc_buf2 = buffer2.data() + offsets[id - bs] * K;
                double Z = corpus.w[d].size() + alpha_bar;

                if (!vr) {
                    for (auto v: corpus.w[d]) {
                        auto *ph = phi.data() + v * K;

                        double sum = 0;
                        for (int k = 0; k < K; k++)
                            sum += prob[k] = th[k] * ph[k];
                        auto scale = (double) corpus.w[d].size() / sum;

                        double max_rth = -1e100;
                        sum = 0;
                        for (int k = 0; k < K; k++) {
                            rth[k] += step_size * (alpha + scale * prob[k] - Z * th[k]);
                            max_rth = max(max_rth, rth[k]);
                        }
                        for (int k = 0; k < K; k++)
                            sum += th[k] = exp(rth[k] - max_rth);

                        sum = (1.0 - FLAGS_min_param * K) / sum;
                        for (int k = 0; k < K; k++)
                            th[k] = th[k] * sum + FLAGS_min_param;
                    }
                    for (int n = 0; n < corpus.w[d].size(); n++) {
                        int v = corpus.w[d][n];
                        auto *ph = phi.data() + v * K;
                        auto *prob = doc_buf + n * K;

                        double sum = 0;
                        for (int k = 0; k < K; k++)
                            sum += prob[k] = th[k] * ph[k];
                        double scale = Tscale / sum;
                        for (int k = 0; k < K; k++)
                            prob[k] *= scale;
                    }
                } else {
                    for (auto v: corpus.w[d]) {
                        auto *ph = phi.data() + v * K;
                        auto *hph = phi_hist.data() + v * K;

                        double sum_prob = 0;
                        double sum_hist = 0;
                        for (int k = 0; k < K; k++) {
                            sum_prob += prob[k] = th[k] * ph[k];
                            sum_hist += hist[k] = hth[k] * hph[k];
                        }
                        auto scale_prob = (double) corpus.w[d].size() / sum_prob;
                        auto scale_hist = (double) corpus.w[d].size() / sum_hist;

                        double max_rth = -1e100;
                        double sum = 0;
                        for (int k = 0; k < K; k++) {
                            rth[k] += step_size * (scale_prob * prob[k] - scale_hist * hist[k] + cd_v[k] - Z * th[k]);
                            max_rth = max(max_rth, rth[k]);
                        }
                        for (int k = 0; k < K; k++)
                            sum += th[k] = exp(rth[k] - max_rth);

                        sum = (1.0 - FLAGS_min_param * K) / sum;
                        for (int k = 0; k < K; k++)
                            th[k] = th[k] * sum + FLAGS_min_param;
                    }
                    for (int n = 0; n < corpus.w[d].size(); n++) {
                        int v = corpus.w[d][n];
                        auto *ph = phi.data() + v * K;
                        auto *hph = phi_hist.data() + v * K;
                        auto *prob = doc_buf + n * K;
                        auto *hist = doc_buf2 + n * K;

                        double sum_prob = 0;
                        double sum_hist = 0;
                        for (int k = 0; k < K; k++) {
                            sum_prob += prob[k] = th[k] * ph[k];
                            sum_hist += hist[k] = hth[k] * hph[k];
                        }
                        auto scale_prob = Tscale / sum_prob;
                        auto scale_hist = Tscale / sum_hist;
                        for (int k = 0; k < K; k++) {
                            prob[k] *= scale_prob;
                            hist[k] *= scale_hist;
                        }
                    }
                }
            }

            Accumulator<double> ck_buffer(K), ck2_buffer(K);
            fill(cvk.begin(), cvk.end(), beta);
            fill(cvk2.begin(), cvk2.end(), beta);
            if (!vr) {
#pragma omp parallel for schedule(static, 10)
                for (int v = 0; v < corpus.V; v++) {
                    auto *cv = cvk.data() + v * K;
                    auto *ck = ck_buffer.Get();
                    for (auto t: v_indices[v]) {
                        auto *buf = buffer.data() + t * K;
                        for (int k = 0; k < K; k++) {
                            cv[k] += buf[k];
                            ck[k] += buf[k];
                        }
                    }
                }
                auto ck = ck_buffer.Sum();
                for (int k = 0; k < K; k++) ck[k] += beta * corpus.V;

#pragma omp parallel for
                for (int v = 0; v < corpus.V; v++) {
                    auto *rph = rphi.data() + v * K;
                    auto *ph = phi.data() + v * K;
                    auto *cv = cvk.data() + v * K;
                    for (int k = 0; k < K; k++)
                        rph[k] += step_size * (cv[k] - ck[k] * ph[k]);
                }
            } else {
#pragma omp parallel for schedule(static, 10)
                for (int v = 0; v < corpus.V; v++) {
                    auto *cv = cvk.data() + v * K;
                    auto *cv2 = cvk2.data() + v * K;
                    auto *ck = ck_buffer.Get();
                    auto *ck2 = ck2_buffer.Get();
                    for (auto t: v_indices[v]) {
                        auto *buf = buffer.data() + t * K;
                        auto *buf2 = buffer2.data() + t * K;
                        for (int k = 0; k < K; k++) {
                            cv[k] += buf[k];
                            ck[k] += buf[k];
                            cv2[k] += buf2[k];
                            ck2[k] += buf2[k];
                        }
                    }
                }
                auto ck = ck_buffer.Sum();
                for (int k = 0; k < K; k++) ck[k] += beta * corpus.V;
                auto ck2 = ck2_buffer.Sum();
                for (int k = 0; k < K; k++) ck2[k] += beta * corpus.V;

#pragma omp parallel for
                for (int v = 0; v < corpus.V; v++) {
                    auto *rph = rphi.data() + v * K;
                    auto *ph = phi.data() + v * K;
                    auto *hph = phi_hist.data() + v * K;
                    auto *cv = cvk.data() + v * K;
                    auto *cv2 = cvk2.data() + v * K;
                    auto *cvv = cv_cvk.data() + v * K;
                    for (int k = 0; k < K; k++)
                        rph[k] += step_size * (cv[k] - cv2[k] + cvv[k] - ck[k] * ph[k] + ck2[k] * hph[k] - cv_ck[k] * hph[k]);
                }
            }
            Phi();

            Eval(iter, bs/batch_size, theta, phi);
        }

        Eval(iter, -1, theta, phi);
    }
}
