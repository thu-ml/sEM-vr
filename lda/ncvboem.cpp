//
// Created by jianfei on 18-1-23.
//

#include "ncvboem.h"
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

NCVBOEM::NCVBOEM(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
           int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          cdk(corpus.D * K), theta(corpus.D * K),
          var(corpus.D * K), var_hist(corpus.D * K),
          phi(corpus.V * K), cvk(corpus.V * K), next_cvk(corpus.V * K), cv_cvk(corpus.V * K), phi_hist(corpus.V * K),
          var_buffer(K), work_buffer(K), next_cd_buffer(K), next_var_buffer(K)
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
    var = theta;

#pragma omp parallel for schedule(static, 10)
    for (int v = 0; v < corpus.V; v++)
        for (auto d: corpus.d[v]) {
            int k = generator() % K;
            cvk[v * K + k]++;
        }

    Phi(cvk);
    cout << "Initialization Finished" << endl;
}

void NCVBOEM::Phi(std::vector<double> &cvk)
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

void NCVBOEM::RealPhi()
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

void NCVBOEM::InferenceDocument(int d, double *th, double *phi, double *doc_buf, double *var_in, double *var_out) {
    auto *var = var_buffer.Get();
    copy(var_in, var_in+K, var);
    auto *next_cd = next_cd_buffer.Get();
    auto *next_var = next_var_buffer.Get();
    double scale = 1.0 / (corpus.w[d].size() + alpha_bar);

    // Initialize
    fill(next_cd, next_cd+K, alpha);
    fill(next_var, next_var+K, 0);
    for (size_t n = 0; n < corpus.w[d].size(); n++) {
        auto v = corpus.w[d][n];
        auto *ph = phi + v * K;
        double sum = 0;
        auto *prob = doc_buf + n * K;
        for (int k = 0; k < K; k++)
            sum += prob[k] = ph[k] * var[k];
        sum = 1.0 / sum;
        for (int k = 0; k < K; k++) {
            prob[k] *= sum;
            next_cd[k] += prob[k];
            next_var[k] += prob[k] * (1 - prob[k]);
        }
    }

    // CVB inference
    for (int iiter = 0; iiter < FLAGS_num_inf_iters; iiter++) {
        for (size_t n = 0; n < corpus.w[d].size(); n++) {
            auto v = corpus.w[d][n];
            auto *ph = phi + v * K;
            double sum = 0;
            auto *prob = doc_buf + n * K;

            for (int k = 0; k < K; k++) {
                next_cd[k] -= prob[k];
                next_var[k] -= prob[k] * (1 - prob[k]);
            }

            for (int k = 0; k < K; k++)
                sum += prob[k] = ph[k] * next_cd[k] * exp(-next_var[k] / (2 * sqr(next_cd[k])));
            sum = 1.0 / sum;

            for (int k = 0; k < K; k++) {
                prob[k] *= sum;
                next_cd[k] += prob[k];
                next_var[k] += prob[k] * (1 - prob[k]);
            }
        }
    }
    if (th != nullptr)
        for (int k = 0; k < K; k++)
                th[k] = next_cd[k] * scale;
    if (var_out != nullptr) {
        for (int k = 0; k < K; k++)
            var_out[k] = next_cd[k] * exp(-next_var[k] / (2 * sqr(next_cd[k])));
    }
}

void NCVBOEM::Estimate()
{
    size_t batch_count = 0;
    double step_size = 0;
    int batch_size = corpus.D / FLAGS_num_batches + 1;

    int iter_cnt = 0;
    int scsg_cnt = FLAGS_scsg_const;
    int next_scsg = 0;
    auto GetSCSG = [&]() {
        return (int)pow(scsg_cnt, 2);
    };

    std::vector<std::vector<double>> prob_buffer(omp_get_max_threads());
    double running_time = 0;
    bool vr = false;
    for (int iter = 0; iter < FLAGS_num_iters; iter++) {
        if (iter >= 1)
            vr = FLAGS_vr;
        std::vector<int> perm(corpus.D);
        iota(perm.begin(), perm.end(), 0);
        shuffle(perm.begin(), perm.end(), generator);

        next_scsg = 0;
        iter_cnt = 0;

        Clock clk;
        for (int bs = 0; bs < corpus.D; bs += batch_size, iter_cnt++) {
            clk.tic();

            if (vr && next_scsg == iter_cnt) {
                cout << "SCSG " << iter_cnt << ' ' << step_size << endl;
                if (FLAGS_proj) {
                    PositiveProjection(cvk);
                    Phi(cvk);
                }
                auto scsg_delta = GetSCSG();
                next_scsg += scsg_delta;
                scsg_cnt++;
                int scsg_end = min(next_scsg * batch_size , corpus.D);
                if ( (next_scsg+scsg_delta) * batch_size > corpus.D)
                    scsg_end = corpus.D;

                parallel_copy(var_hist, var);
                parallel_copy(phi_hist, phi);

                double Tscale = (double)corpus.D / (scsg_end - bs);
                Accumulator<double> cvk_accumulator(corpus.V * K);
#pragma omp parallel for schedule(dynamic, 10)
                for (int id = bs; id < scsg_end; id++) {
                    int d = perm[id];
                    auto &buf = prob_buffer[omp_get_thread_num()];
                    int N = corpus.w[d].size();
                    buf.resize(N * K);

                    InferenceDocument(d, nullptr, phi_hist.data(), buf.data(), var_hist.data() + d * K, nullptr);

                    auto *cvk = cvk_accumulator.Get();
                    for (int n = 0; n < N; n++) {
                        auto v = corpus.w[d][n];
                        auto *cv = cvk + v * K;
                        auto *prob = buf.data() + n * K;
                        for (int k = 0; k < K; k++)
                            cv[k] += Tscale * prob[k];
                    }
                }
                cv_cvk = cvk_accumulator.Sum();
            }

            if (!vr)
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
            buffer_hist.resize(sum * K);

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
                InferenceDocument(d, theta.data() + d * K, phi.data(), doc_buf, var.data() + d * K, var.data() + d * K);

                if (vr) {
                    auto *doc_hist_buf = buffer_hist.data() + offsets[id-bs] * K;
                    InferenceDocument(d, nullptr, phi_hist.data(), doc_hist_buf, var_hist.data() + d * K, nullptr); // TODO: can be optimized
                }
            }

            // M step
            if (vr) {
#pragma omp parallel for schedule(static, 10)
                for (int v = 0; v < corpus.V; v++) {
                    auto *cv = next_cvk.data() + v * K;
                    for (auto t: v_indices[v]) {
                        auto *buf = buffer.data() + t * K;
                        auto *buf_hist = buffer_hist.data() + t * K;
                        for (int k = 0; k < K; k++)
                            cv[k] += Tscale * (buf[k] - buf_hist[k]);
                    }
                }
            } else {
#pragma omp parallel for schedule(static, 10)
                for (int v = 0; v < corpus.V; v++) {
                    auto *cv = next_cvk.data() + v * K;
                    for (auto t: v_indices[v]) {
                        auto *buf = buffer.data() + t * K;
                        for (int k = 0; k < K; k++)
                            cv[k] += Tscale * buf[k];
                    }
                }
            }

            batch_count += 1;
            if (vr && !FLAGS_decay_step)
                step_size = FLAGS_max_step;
            else
                step_size = GetSchedule(batch_count);

            double sum_next = accumulate(next_cvk.begin(), next_cvk.end(), 0.0);
            double sum_curr = accumulate(cvk.begin(), cvk.end(), 0.0);

#pragma omp parallel for
            for (size_t i = 0; i < cvk.size(); i++) cvk[i] = (1-step_size)*cvk[i] + step_size*next_cvk[i];

            if (FLAGS_proj) {
                RenormalizeProjection(cvk, phi);
            } else {
                if (!vr)
                    Phi(cvk);
                else
                    RealPhi();
            }

#pragma omp parallel for schedule(static, 10)
            for (int v = 0; v < corpus.V; v++)
                for (int k = 0; k < K; k++) {
                    if (phi[v*K+k] < 0)
                        throw runtime_error("negative phi");
                    if (std::isnan(phi[v*K+k])) {
                        cout << v << endl;
                        print(phi.data() + v * K, K);
                        throw runtime_error("phi is nan");
                    }
                }
            running_time += clk.toc();

            if (FLAGS_test_lag != -1 && batch_count % FLAGS_test_lag == 0) {
                cout << "Testing ";
                cout.flush();
//                double train_ppl = BaseLDA::Inference(phi, th_corpus);
                double train_ppl = 0;
                double test_ppl  = BaseLDA::Inference(phi);
                cout << "perplexity " << train_ppl << ' ' << test_ppl
                     << " time elapsed " << running_time << endl;
            }
        }

        // Compute perplexity
        double perplexity = BatchPerplexity();
//        double train_ppl = BaseLDA::Inference(phi, th_corpus);
        double train_ppl = 0;
        double test_ppl  = BaseLDA::Inference(phi);
        cout << "Iteration " << iter
             << " step size = " << step_size
             << " perplexity = " << perplexity << ' ' << train_ppl
             << " tperplexity = " << test_ppl
             << " time = " << running_time << endl;
//        SaveMatrix(phi, corpus.V, K, "phi.mat");
//        SaveMatrix(theta, corpus.D, K, "theta.mat");
    }
}

double NCVBOEM::BatchLogLikelihood()
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

double NCVBOEM::BatchPerplexity()
{
    double ll = BatchLogLikelihood();
    double perplexity = exp(-ll / corpus.T);
    return perplexity;
}