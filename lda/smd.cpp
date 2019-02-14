//
// Created by jianfei on 18-1-23.
//

#include "smd.h"
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

SMD::SMD(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
           int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          theta(corpus.D * K), phi(corpus.V * K),
          cvk(corpus.V * K), cvk2(corpus.V * K),
          theta_hist(corpus.D * K), phi_hist(corpus.V * K),
          cv_cdk(corpus.D * K), cv_cvk(corpus.V * K)
{
#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++) {
        auto *th = theta.data() + d * K;
        double sum = 0;
        for (int k = 0; k < K; k++)
            sum += th[k] = u01(generator);
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
            ck[k] += ph[k] = (double)corpus.d[v].size() / K + u01(generator);
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

    cout << "Initialization Finished" << endl;
}

void SMD::Estimate()
{
    Accumulator<double> prob_buffer(K), hist_buffer(K);
    size_t batch_count = 0;
    int batch_size = corpus.D / FLAGS_num_batches + 1;

    step_size = FLAGS_max_step;
    bool vr = FLAGS_vr;
    int iter_cnt = 0;
    for (int iter = 0; iter < FLAGS_num_iters; iter++) {
        std::vector<int> perm(corpus.D);
        iota(perm.begin(), perm.end(), 0);
        shuffle(perm.begin(), perm.end(), generator);

#pragma omp parallel for
        for (int d = 0; d < corpus.D; d++)
            shuffle(corpus.w[d].begin(), corpus.w[d].end(), generator);

        if (vr) {
            parallel_copy(theta_hist, theta);
            parallel_copy(phi_hist, phi);
            // Compute batch sufficient statistics cv_cdk, cv_cvk
            fill(cv_cdk.begin(), cv_cdk.end(), alpha);
            Accumulator<double> cv_cvk_acc(corpus.V * K);
#pragma omp parallel for
            for (int d = 0; d < corpus.D; d++) {
                auto *th = theta.data() + d * K;
                auto *cd = cv_cdk.data() + d * K;
                auto *prob = prob_buffer.Get();
                auto *cvk = cv_cvk_acc.Get();
                for (auto v: corpus.w[d]) {
                    auto *ph = phi.data() + v * K;
                    auto *cv = cvk + v * K;

                    double sum = 0;
                    for (int k = 0; k < K; k++)
                        sum += prob[k] = th[k] * ph[k];
                    sum = 1.0 / sum;
                    for (int k = 0; k < K; k++) {
                        auto p = prob[k] * sum;
                        cd[k] += p;
                        cv[k] += p;
                    }
                }
            }
            cv_cvk = cv_cvk_acc.Sum();
#pragma omp parallel for
            for (size_t i = 0; i < cv_cvk.size(); i++)
                cv_cvk[i] += beta;
        }

        for (int bs = 0; bs < corpus.D; bs += batch_size) {
            step_size = GetSchedule(++iter_cnt);
            int be = std::min(corpus.D, bs + batch_size);
            auto Tscale = (double) corpus.D / (be - bs);

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

            // A pass for doc
#pragma omp parallel for
            for (int id = bs; id < be; id++) {
                auto d = perm[id];
                auto *th = theta.data() + d * K;
                auto *hth = theta_hist.data() + d * K;
                auto *cv_cd = cv_cdk.data() + d * K;
                auto *prob = prob_buffer.Get();
                auto *hist = hist_buffer.Get();
                auto *doc_buf = buffer.data() + offsets[id - bs] * K;
                auto *doc_buf2 = buffer2.data() + offsets[id - bs] * K;

                if (!vr) {
                    for (auto v: corpus.w[d]) {
                        auto *ph = phi.data() + v * K;

                        double sum = 0;
                        for (int k = 0; k < K; k++)
                            sum += prob[k] = th[k] * ph[k];
                        double scale = (double) corpus.w[d].size() / sum;

                        sum = 0;
                        for (int k = 0; k < K; k++)
                            sum += th[k] *= exp(min(step_size * (scale * prob[k] + alpha) / th[k], 10.0));

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
                        double scale_prob = (double) corpus.w[d].size() / sum_prob;
                        double scale_hist = (double) corpus.w[d].size() / sum_hist;

                        double sum = 0;
                        for (int k = 0; k < K; k++) {
                            double current_term = (scale_prob * prob[k] + alpha) / th[k];
                            double old_term     = (scale_hist * hist[k] + alpha) / hth[k];
                            double cv_term      = (cv_cd[k] + alpha) / hth[k];
                            double gradient = min(step_size * (current_term - old_term + cv_term), 10.0);
                            sum += th[k] *= exp(gradient);
                        }

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
                        double scale_prob = Tscale / sum_prob;
                        double scale_hist = Tscale / sum_hist;
                        for (int k = 0; k < K; k++) {
                            prob[k] *= scale_prob;
                            hist[k] *= scale_hist;
                        }
                    }
                }
            }

            Accumulator<double> ck_buffer(K);
            if (!vr) {
                fill(cvk.begin(), cvk.end(), beta);
#pragma omp parallel for schedule(static, 10)
                for (int v = 0; v < corpus.V; v++) {
                    auto *cv = cvk.data() + v * K;
                    for (auto t: v_indices[v]) {
                        auto *buf = buffer.data() + t * K;
                        for (int k = 0; k < K; k++)
                            cv[k] += buf[k];
                    }
                }
#pragma omp parallel for
                for (int v = 0; v < corpus.V; v++) {
                    auto *ph = phi.data() + v * K;
                    auto *cv = cvk.data() + v * K;
                    for (int k = 0; k < K; k++)
                        cv[k] = step_size * (cv[k] + beta) / ph[k];
                }
            } else {
                fill(cvk.begin(), cvk.end(), beta);
                fill(cvk2.begin(), cvk2.end(), beta);
#pragma omp parallel for schedule(static, 10)
                for (int v = 0; v < corpus.V; v++) {
                    auto *cv = cvk.data() + v * K;
                    auto *cv2 = cvk2.data() + v * K;
                    for (auto t: v_indices[v]) {
                        auto *buf = buffer.data() + t * K;
                        auto *buf2 = buffer2.data() + t * K;
                        for (int k = 0; k < K; k++) {
                            cv[k] += buf[k];
                            cv2[k] += buf2[k];
                        }
                    }
                }
#pragma omp parallel for
                for (int v = 0; v < corpus.V; v++) {
                    auto *ph = phi.data() + v * K;
                    auto *hph = phi_hist.data() + v * K;
                    auto *cv = cvk.data() + v * K;
                    auto *cv2 = cvk2.data() + v * K;
                    auto *cvh = cv_cvk.data() + v * K;
                    for (int k = 0; k < K; k++) {
                        double current_term = (cv[k] + beta) / ph[k];
                        double old_term     = (cv2[k] + beta) / hph[k];
                        double cv_term      = (cvh[k] + beta) / hph[k];
                        cv[k] = step_size * (current_term - old_term + cv_term);
                    }
                }
            }
            // ph[k] *= exp(cv[k])
            std::vector<double> max_cv(K, -1e100);
#pragma omp parallel
            {
                std::vector<double> mcv(K, -1e100);
#pragma omp for
                for (int v = 0; v < corpus.V; v++) {
                    auto *cv = cvk.data() + v * K;
                    for (int k = 0; k < K; k++)
                        mcv[k] = max(mcv[k], cv[k]);
                }
#pragma omp critical
                {
                    for (int k = 0; k < K; k++)
                        max_cv[k] = max(max_cv[k], mcv[k]);
                }
            }
#pragma omp parallel for
            for (int v = 0; v < corpus.V; v++) {
                auto *cv = cvk.data() + v * K;
                auto *ck = ck_buffer.Get();
                auto *ph = phi.data() + v * K;
                for (int k = 0; k < K; k++)
                    ck[k] += ph[k] *= exp(cv[k] - max_cv[k]);
            }

            auto ck = ck_buffer.Sum();
            for (int k = 0; k < K; k++)
                ck[k] = (1.0 - FLAGS_min_param * corpus.V) / ck[k];
#pragma omp parallel for
            for (int v = 0; v < corpus.V; v++) {
                auto *ph = phi.data() + v * K;
                for (int k = 0; k < K; k++)
                    ph[k] = ph[k] * ck[k] + FLAGS_min_param;
            }
            Eval(iter, bs/batch_size, theta, phi);
        }

        Eval(iter, -1, theta, phi);
    }
}
