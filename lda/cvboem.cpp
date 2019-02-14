//
// Created by jianfei on 18-1-23.
//

#include "cvboem.h"
#include "corpus.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <exception>
#include "gflags/gflags.h"
#include "utils.h"
#include "flags.h"
#include "clock.h"
using namespace std;

CVBOEM::CVBOEM(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
           int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          cdk(corpus.D * K), theta(corpus.D * K), theta_hist(corpus.D * K),
          cvk(corpus.V * K), next_cvk(corpus.V * K), cv_cvk(corpus.V * K),
          phi(corpus.V * K), phi_hist(corpus.V * K),
          cd_buffer(K), var_buffer(K), work_buffer(K)
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

void CVBOEM::Phi(std::vector<double> &cvk)
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

void CVBOEM::InferenceDocument(int d, double *th, double *phi, double *doc_buf) {
    auto *cd = cd_buffer.Get();
    auto *var = var_buffer.Get();
    double scale = 1.0 / (corpus.w[d].size() + alpha_bar);
    // Initialize cd and doc_buf from theta
    fill(cd, cd+K, alpha);
    fill(var, var+K, 0);
    for (size_t n = 0; n < corpus.w[d].size(); n++) {
        auto v = corpus.w[d][n];
        auto *ph = phi + v * K;
        auto *prob = doc_buf + n * K;
        double sum = 0;

        for (int k = 0; k < K; k++)
            sum += prob[k] = ph[k] * th[k];
        sum = 1.0 / sum;
        for (int k = 0; k < K; k++) {
            cd[k] += prob[k] *= sum;
            var[k] += prob[k] * (1 - prob[k]);
        }
    }
    // CVB inference
    for (int iiter = 0; iiter < FLAGS_num_inf_iters; iiter++) {
        for (size_t n = 0; n < corpus.w[d].size(); n++) {
            auto v = corpus.w[d][n];
            auto *ph = phi + v * K;
            auto *prob = doc_buf + n * K;
            double sum = 0;

            for (int k = 0; k < K; k++) {
                cd[k] -= prob[k];
                var[k] -= prob[k] * (1 - prob[k]);
            }
            for (int k = 0; k < K; k++)
                sum += prob[k] = ph[k] * cd[k] * exp(-var[k] / (2*sqr(cd[k])));
            sum = 1.0 / sum;
            for (int k = 0; k < K; k++) {
                cd[k] += prob[k] *= sum;
                var[k] += prob[k] * (1 - prob[k]);
            }
        }
    }
    // Estimate theta
    for (int k = 0; k < K; k++)
        th[k] = cd[k] * scale;
}

void CVBOEM::UpdateStatistics(int d, double *th, double *phi, double *doc_buf, double Tscale) {
    auto *work = work_buffer.Get();
    for (size_t n = 0; n < corpus.w[d].size(); n++) {
        auto v = corpus.w[d][n];
        auto *ph = phi + v * K;
        auto *prob = doc_buf + n * K;
        double sum = 0;

        // TODO is the posterior expected SS really theta??? no it is prob.
        for (int k = 0; k < K; k++)
            sum += work[k] = ph[k] * th[k];
        sum = Tscale / sum;
        for (int k = 0; k < K; k++)
            prob[k] = work[k] * sum;
    }
}

void CVBOEM::Estimate()
{
    size_t batch_count = 0;
    double step_size = 0;
    int batch_size = corpus.D / FLAGS_num_batches + 1;
    std::vector<std::vector<double> > working_buffer(omp_get_max_threads());

    double running_time = 0;
    for (int iter = 0; iter < FLAGS_num_iters; iter++) {
        Clock clk;
        std::vector<int> perm(corpus.D);
        iota(perm.begin(), perm.end(), 0);
        shuffle(perm.begin(), perm.end(), generator);

        if (FLAGS_vr) {
            parallel_copy(theta_hist, theta);
            parallel_copy(phi_hist, phi);

            Accumulator<double> cvk_accumulator(corpus.V * K);
#pragma omp parallel for schedule(dynamic, 10)
            for (int d = 0; d < corpus.D; d++) {
                auto &buf = working_buffer[omp_get_thread_num()];
                buf.resize(corpus.w[d].size() * K);
                auto *th = theta_hist.data() + d * K;
                InferenceDocument(d, th, phi_hist.data(), buf.data());

                auto *cvk = cvk_accumulator.Get();
                auto *work = work_buffer.Get();
                for (size_t n = 0; n < corpus.w[d].size(); n++) {
                    auto v = corpus.w[d][n];
                    auto *ph = phi.data() + v * K;
                    auto *cv = cvk + v * K;
                    double sum = 0;

                    // TODO is the posterior expected SS really theta??? no it is prob.
                    for (int k = 0; k < K; k++)
                        sum += work[k] = ph[k] * th[k];
                    sum = 1.0 / sum;
                    for (int k = 0; k < K; k++)
                        cv[k] += work[k] * sum;
                }
            }
            cv_cvk = cvk_accumulator.Sum();
        }
        running_time += clk.toc();

        for (int bs = 0; bs < corpus.D; bs += batch_size) {
            clk.tic();
            if (!FLAGS_vr)
                parallel_zero(next_cvk);
            else
                parallel_copy(next_cvk, cv_cvk);

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
                auto *doc_buf = buffer.data() + offsets[id-bs] * K;
                InferenceDocument(d, theta.data() + d * K, phi.data(), doc_buf);
                UpdateStatistics(d, theta.data() + d * K, phi.data(), doc_buf, Tscale);
                if (FLAGS_vr) {
                    InferenceDocument(d, theta_hist.data() + d * K, phi_hist.data(), doc_buf);
                    UpdateStatistics(d, theta_hist.data() + d * K, phi_hist.data(), doc_buf, -Tscale);
                }
            }

            // M step
#pragma omp parallel for schedule(static, 10)
            for (int v = 0; v < corpus.V; v++) {
                auto *cv = next_cvk.data() + v * K;
                for (auto t: v_indices[v]) {
                    auto *buf = buffer.data() + t * K;
                    for (int k = 0; k < K; k++)
                        cv[k] += buf[k];
                }
            }

            batch_count += 1;
            if (FLAGS_vr)
                step_size = FLAGS_max_step;
            else
                step_size = GetSchedule(batch_count);
#pragma omp parallel for
            for (size_t i = 0; i < cvk.size(); i++) cvk[i] = (1-step_size)*cvk[i] + step_size*next_cvk[i];

            if (!FLAGS_vr)
                Phi(cvk);
            else
                RealPhi();

#pragma omp parallel for schedule(static, 10)
            for (int v = 0; v < corpus.V; v++)
                for (int k = 0; k < K; k++) {
                    if (phi[v*K+k] < 0)
                        throw runtime_error("negative phi");
                    if (std::isnan(phi[v*K+k]))
                        throw runtime_error("phi is nan");
                }
            running_time += clk.toc();

            if (FLAGS_test_lag != -1 && batch_count % FLAGS_test_lag == 0) {
                cout << "Testing ";
                cout.flush();
                cout << "perplexity " << BaseLDA::Inference(phi)
                     << " time " << running_time << endl;
            }
        }

        // Compute perplexity
        double perplexity = BatchPerplexity();
        double tperplexity = BaseLDA::Inference(phi);
        cout << "Iteration " << iter
             << " step size = " << step_size
             << " perplexity = " << perplexity
             << " tperplexity = " << tperplexity
             << " time = " << running_time << endl;
    }
}

double CVBOEM::BatchLogLikelihood()
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

double CVBOEM::BatchPerplexity()
{
    double ll = BatchLogLikelihood();
    double perplexity = exp(-ll / corpus.T);
    return perplexity;
}

void CVBOEM::RealPhi()
{
    Accumulator<double> sum_buffer(K);
#pragma omp parallel for
    for (int v = 0; v < corpus.V; v++) {
        double *ph = phi.data() + v * K;
        double *cv = cvk.data() + v * K;
        auto   *s  = sum_buffer.Get();
        for (int k = 0; k < K; k++)
            s[k] += ph[k] = max(cv[k] + beta, 0.0);
    }
    auto sum = sum_buffer.Sum();
    for (int k = 0; k < K; k++)
        sum[k] = (1.0 - FLAGS_min_param * corpus.V) / sum[k];

#pragma omp parallel for
    for (int v = 0; v < corpus.V; v++) {
        double *ph = phi.data() + v * K;
        for (int k = 0; k < K; k++)
            ph[k] = FLAGS_min_param + sum[k] * ph[k];
    }
}