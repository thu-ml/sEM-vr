//
// Created by 陈键飞 on 2018/4/17.
//

#include "cvsEM.h"
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

CVSEM::CVSEM(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
             int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          cdk(corpus.D * K), cvk(corpus.V * K),
          cv_cdk(corpus.D * K), cv_cvk(corpus.V * K),
          next_cvk(corpus.V * K),
          theta(corpus.D * K), phi(corpus.V * K),
          ma1((corpus.D+corpus.V)*K)
{
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

    batch_size = corpus.D / FLAGS_num_batches + 1;
    cout << "Initialization Finished" << endl;
}

void CVSEM::GetPosterior(double *theta, double *phi, double *prob) {
    double sum = 0;
    for (int k = 0; k < K; k++) {
        sum += prob[k] = theta[k] * phi[k];
    }
    sum = 1.0 / sum;
    for (int k = 0; k < K; k++)
        prob[k] *= sum;
}

void CVSEM::Estimate()
{
    Start();
    ofstream f_perplexity("perplexity.log");

    std::vector<double> lls;
    Accumulator<double> prob_buffer(K), hist_buffer(K);

    step_size = FLAGS_max_step;
    D_step_size = FLAGS_max_step;
    int iter_cnt = 0;
    int scsg_cnt = FLAGS_scsg_const;
    int next_scsg = 0;
    auto GetSCSG = [&]() {
        return (int)pow(scsg_cnt, 1.5);
    };

//    double running_time = 0;
    bool vr = false;
    for (int iter = 0; iter < FLAGS_num_iters; iter++) {
        if (iter >= 1)
            vr = FLAGS_vr;
        next_scsg = 0;
        step_size_seq.clear();
        this->iter = iter;

        Clock clk;
        if (iter >= FLAGS_batch_after) {
            fill(cdk.begin(), cdk.end(), 0);
            fill(cvk.begin(), cvk.end(), 0);
            std::vector<int> docs(corpus.D);
            iota(docs.begin(), docs.end(), 0);
            BatchIteration(docs, theta, phi, cdk, cvk);
            auto cv_time = clk.toc();

            for (int d = 0; d < corpus.D; d++)
                Theta(cdk.data()+d*K, theta.data()+d*K, corpus.w[d].size());

            Phi(cvk);
            Eval(iter, -1, theta, phi);
            continue;
        }

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
            if (vr && iiter == next_scsg) {
                cout << "SCSG " << iiter << endl;
                int scsg_size = GetSCSG();
                scsg_cnt += 1;

                fill(cv_cvk.begin(), cv_cvk.end(), 0);
                int d_start = iiter * batch_size;
                int d_end   = min((iiter + scsg_size) * batch_size, corpus.D);
                if (d_start >= d_end)
                    break;
                std::vector<int> docs;
                for (int d = d_start; d < d_end; d++)
                    docs.push_back(perm[d]);
                for (int d: docs)
                    fill(cv_cdk.data()+d*K, cv_cdk.data()+(d+1)*K, 0);

                BatchIteration(docs, theta, phi, cv_cdk, cv_cvk);
                for (int d: docs)
                    copy(cv_cdk.data()+d*K, cv_cdk.data()+(d+1)*K, cdk.data()+d*K);
                parallel_copy(theta_hist, theta);
                parallel_copy(phi_hist, phi);
                double scale = corpus.D / (d_end - d_start);
                for (size_t i = 0; i < cv_cvk.size(); i++)
                    cv_cvk[i] *= scale;

                next_scsg += scsg_size;
                cout << "Done" << endl;
            }
            cv_time += clk2.toc();

            iter_cnt++;

            if (!vr || FLAGS_decay_step) {
                step_size = D_step_size = GetSchedule(iter_cnt);
            } else
                step_size = FLAGS_max_step;

            clk2.tic();
            if (vr)
                parallel_copy(next_cvk, cv_cvk);
            else
                parallel_zero(next_cvk);

            int d_start = iiter * batch_size;
            int d_end   = min((iiter + 1) * batch_size, corpus.D);
            if (d_start >= d_end)
                break;

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

                auto *prob = prob_buffer.Get();
                auto *hist = hist_buffer.Get();

                auto *th = theta.data() + d * K;
                auto *cd = cdk.data() + d * K;
                auto *cv_cd = cv_cdk.data() + d * K;

                int Len = corpus.w[d].size();
                auto *doc_buf = buffer.data() + offsets[id - d_start] * K;

                if (vr) {
                    for (int vid = 0; vid < corpus.w[d].size(); vid++) {
                        int v = corpus.w[d][vid];
                        GetPosterior(theta_hist.data() + d * K, phi_hist.data() + v * K, hist);
                        GetPosterior(th, phi.data() + v * K, prob);

                        double sum = 0;
                        for (int k = 0; k < K; k++) {
                            double delta = prob[k] - hist[k];
                            double next_cdk = delta * Len + cv_cd[k];
                            double old_cdk = cd[k];
                            cd[k] += D_step_size * (next_cdk - old_cdk);
                            sum += th[k] = max(cd[k] + alpha, 0.0);
                        }
                        sum = (1.0 - FLAGS_min_param * K) / sum;
                        for (int k = 0; k < K; k++)
                            th[k] = FLAGS_min_param + sum * th[k];
                    }
                    for (int vid = 0; vid < corpus.w[d].size(); vid++) {
                        int v = corpus.w[d][vid];
                        GetPosterior(theta_hist.data() + d * K, phi_hist.data() + v * K, hist);
                        GetPosterior(th, phi.data() + v * K, prob);

                        auto *buf = doc_buf + vid * K;
                        for (int k = 0; k < K; k++)
                            buf[k] = T_scale * (prob[k] - hist[k]);
                    }
                } else {
                    for (int vid = 0; vid < corpus.w[d].size(); vid++) {
                        int v = corpus.w[d][vid];
                        GetPosterior(th, phi.data() + v * K, prob);

                        double scale = 1.0 / (corpus.w[d].size() + alpha_bar);
                        for (int k = 0; k < K; k++) {
                            double next_cdk = prob[k] * Len;
                            double old_cdk = cd[k];
                            cd[k] += D_step_size * (next_cdk - old_cdk);
                            th[k] = (cd[k] + alpha) * scale;
                        }
                    }
                    for (int vid = 0; vid < corpus.w[d].size(); vid++) {
                        int v = corpus.w[d][vid];
                        GetPosterior(th, phi.data() + v * K, prob);

                        auto *buf = doc_buf + vid * K;
                        for (int k = 0; k < K; k++)
                            buf[k] = T_scale * prob[k];
                    }
                }
            }
            sample1_time += clk2.toc(); clk2.tic();

            #pragma omp parallel for schedule(static, 10)
            for (int v = 0; v < corpus.V; v++) {
                auto *cv = next_cvk.data() + v * K;
                for (auto t: v_indices[v]) {
                    auto *buf = buffer.data() + t * K;
                    for (int k = 0; k < K; k++)
                        cv[k] += buf[k];
                }
            }
            sample2_time += clk2.toc(); clk2.tic();

            // MyAdaptStepsize(iiter);

#pragma omp parallel for
            for (size_t i = 0; i < cvk.size(); i++) cvk[i] = (1-step_size)*cvk[i] + step_size*next_cvk[i];
            next_time += clk2.toc(); clk2.tic();

            if (vr)
                RealPhi();
            else
                Phi(cvk);
            thetaphi_time += clk2.toc(); clk2.tic();

            Eval(iter, iiter, theta, phi);
        }
        cout << "Sample time " << sample1_time << ' ' << sample2_time
             << " misc time SINT " << shuffle_time << ' ' << init_time << ' ' << next_time << ' ' << thetaphi_time << endl;
        auto eval_time = clk.toc();

        Eval(iter, -1, theta, phi);
    }
}

void CVSEM::Theta(double *cdk, double *theta, int N) {
    double Z = 1.0 / (N + alpha_bar);
    for (int k = 0; k < K; k++)
        theta[k] = (cdk[k] + alpha) * Z;
}

void CVSEM::Phi(std::vector<double> &cvk)
{
    Accumulator<double> ck_buffer(K);
    #pragma omp parallel for
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

void CVSEM::RealPhi()
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

void CVSEM::BatchIteration(std::vector<int> &docs,
                           std::vector<double> &theta, std::vector<double> &phi,
                           std::vector<double> &cdk,   std::vector<double> &cvk)
{
    Accumulator<double> buffer(K);
    Accumulator<double> cvk_buffer(corpus.V * K);

// Do a batch iteration to initialize the control variate
#pragma omp parallel for
    for (int id = 0; id < docs.size(); id++) {
        auto d = docs[id];
        auto *prob  = buffer.Get();
        auto *cd    = cdk.data() + d * K;
        auto *cvk   = cvk_buffer.Get();
        for (auto v: corpus.w[d]) {
            GetPosterior(theta.data()+d*K, phi.data()+v*K, prob);

            auto *cv = cvk + v * K;
            for (int k = 0; k < K; k++) {
                cd[k] += prob[k];
                cv[k] += prob[k];
            }
        }
    }
    cvk = cvk_buffer.Sum();
}
