//
// Created by jianfei on 2018/1/25.
//

#include "scvb0.h"
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

SCVB0::SCVB0(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
             int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          cdk(corpus.D * K), cvk(corpus.V * K), ck(K),
          next_cdk(corpus.D * K), next_cvk(corpus.V * K), next_ck(K),
          theta(corpus.D * K), phi(corpus.V * K)
{
    for (auto &token: corpus.tokens) {
        int k = generator() % K;

        cdk[token.d * K + k]++;
        cvk[token.v * K + k]++;
        ck[k]++;
    }
    cout << "Initialization Finished" << endl;
    ThetaPhi();
    theta_hist = theta;
    phi_hist   = phi;

    batch_size = corpus.T / FLAGS_num_batches * 2;
}

void SCVB0::GetPosterior(double *theta, double *phi, double *prob) {
    double sum = 0;
    for (int k = 0; k < K; k++) {
        sum += prob[k] = theta[k] * phi[k];
    }
    sum = 1.0 / sum;
    for (int k = 0; k < K; k++)
        prob[k] *= sum;
}

double SCVB0::GetPosterior2(double *theta, double *phi, double *prob) {
    double sum = 0;
    for (int k = 0; k < K; k++) {
        sum += prob[k] = theta[k] * phi[k];
    }
    return 1.0 / sum;
}

void SCVB0::Estimate()
{
    int adapt_threshold = FLAGS_adapt_step ? FLAGS_adapt_after : 10000;
    int batch_count = 0;
    double step_size = 0;
    ofstream f_perplexity("perplexity.log");

    std::vector<double> lls;
    int multi = FLAGS_num_batches;

    Accumulator<double> prob_buffer(K), hist_buffer(K);

    auto tokens = corpus.tokens;
    for (int iter = 0; iter < FLAGS_num_iters; iter++) {
        std::atomic<int> num_samples(0);
        this->iter = iter;

        if (iter >= FLAGS_batch_after) {
            Clock clk;
            BatchIteration();
            cout << "Finished BatchIteration" << endl;
            cdk = next_cdk;
            cvk = next_cvk;
            ck  = next_ck;
            ThetaPhi();
            auto t1 = clk.toc(); clk.tic();

            double ll = BatchLogLikelihood();
            double perplexity = exp(-ll / corpus.T);
            auto t2 = clk.toc(); clk.tic();

            printf("Log likelihood = %.20f\n", ll);
            cout << "\e[0;33mIteration " << iter
                 << " perplexity = " << perplexity 
                 << " tperplexity = " << BaseLDA::Inference(phi)
                 << "\e[0;0m" << endl;
            auto t3 = clk.toc();
            cout << t1 << ' ' << t2 << ' ' << t3 << endl;
            continue;
        }

        Clock clk;
        double variance_term;
        double decay;
        VarianceTerm(step_size, variance_term, decay);
        variance_term /= step_size * step_size;
        auto var_time = clk.toc();

        theta_hist = theta;
        phi_hist   = phi;

        clk.tic();
        std::vector<double> cv_cdk(corpus.D * K);
        std::vector<double> cv_cvk(corpus.V * K);
        Accumulator<double> cv_ck_buffer(K);
        // Do a batch iteration to initialize the control variate
#pragma omp parallel for
        for (int d = 0; d < corpus.D; d++) {
            auto *hist  = hist_buffer.Get();
            auto *cd    = cv_cdk.data() + d * K;
            for (auto v: corpus.w[d]) {
                GetPosterior(theta_hist.data()+d*K,
                             phi_hist.data()  +v*K,
                             hist);

                for (int k = 0; k < K; k++)
                    cd[k] += hist[k];
            }
        }
#pragma omp parallel for schedule(static, 1)
        for (int v = 0; v < corpus.V; v++) {
            auto *hist  = hist_buffer.Get();
            auto *cv_ck = cv_ck_buffer.Get();
            auto *cv    = cv_cvk.data() + v * K;
            for (auto d: corpus.d[v]) {
                GetPosterior(theta_hist.data()+d*K,
                             phi_hist.data()  +v*K,
                             hist);

                for (int k = 0; k < K; k++) {
                    cv[k]     += hist[k];
                    cv_ck[k] += hist[k];
                }
            }
        }
        auto cv_ck = cv_ck_buffer.Sum();
        auto cv_time = clk.toc();

        clk.tic();
//        double ll          = BatchLogLikelihood();
        // Stochastic Inference++
        Clock clk2;
        double sample1_time = 0;
        double sample2_time = 0;
        double init_time = 0;
        double next_time = 0;
        double fix_d_time = 0;
        double fix_v_time = 0;
        double thetaphi_time = 0;
        for (int iiter = 0; iiter < FLAGS_num_batches/2; iiter++) {
            clk2.tic();
            if (FLAGS_vr) {
                next_cdk.resize(cv_cdk.size());
                next_cvk.resize(cv_cvk.size());
                std::copy(cv_cdk.begin(), cv_cdk.end(), next_cdk.begin());
                std::copy(cv_cvk.begin(), cv_cvk.end(), next_cvk.begin());
            } else {
                fill(next_cdk.begin(), next_cdk.end(), 0);
                fill(next_cvk.begin(), next_cvk.end(), 0);
            }
            init_time += clk2.toc(); clk2.tic();

            Accumulator<double> next_ck_buffer(K);

#pragma omp parallel for
            for (int d = 0; d < corpus.D; d++) {
                auto *prob = prob_buffer.Get();
                auto *hist = hist_buffer.Get();

                int D = corpus.w[d].size();
                if (!D) continue;
                int M = max(1, D / multi);
                if (iter && FLAGS_adapt_dist)
                    M = dFreq[d];
                num_samples += M;
                double s = (double)D / M;
                partially_shuffle(M, corpus.w[d]); //TODO this takes 4 seconds!!
                for (int i = 0; i < M; i++) {
                    int v = corpus.w[d][i];
                    double sum = 0;
                    GetPosterior(theta_hist.data()+d*K,
                                 phi_hist.data()+v*K,
                                 hist);
                    double *th    = theta.data() + d * K;
                    double *ph    = phi.data()   + v * K;
                    double *n_cdk = next_cdk.data() + d * K;
                    for (int k = 0; k < K; k++)
                        sum += prob[k] = th[k] * ph[k];
                    sum = 1.0 / sum;
                    if (FLAGS_vr)
                        for (int k = 0; k < K; k++) {
                            prob[k] *= sum;
                            n_cdk[k] += s * (prob[k] - hist[k]);
                        }
                    else
                        for (int k = 0; k < K; k++) {
                            prob[k] *= sum;
                            n_cdk[k] += s * prob[k];
                        }
                }
            }
            sample1_time += clk2.toc(); clk2.tic();

#pragma omp parallel for schedule(static, 1)
            for (int v = 0; v < corpus.V; v++) {
                auto *next_ck = next_ck_buffer.Get();
                auto *prob = prob_buffer.Get();
                auto *hist = hist_buffer.Get();

                int V = corpus.d[v].size();
                if (!V) continue;
                int M = max(1, V / multi);
                if (iter && FLAGS_adapt_dist)
                    M = vFreq[v];
                double s = (double)V / M;
                partially_shuffle(M, corpus.d[v]);
                num_samples += M;
                for (int i = 0; i < M; i++) {
                    int d = corpus.d[v][i];
                    double sum = 0;
                    GetPosterior(theta_hist.data()+d*K,
                                 phi_hist.data()+v*K,
                                 hist);
                    double *th    = theta.data() + d * K;
                    double *ph    = phi.data()   + v * K;
                    double *n_cvk = next_cvk.data() + v * K;
                    for (int k = 0; k < K; k++)
                        sum += prob[k] = th[k] * ph[k];
                    sum = 1.0 / sum;
                    if (FLAGS_vr)
                        for (int k = 0; k < K; k++) {
                            prob[k] *= sum;
                            n_cvk[k] += s * (prob[k] - hist[k]);
                            next_ck[k] += s * (prob[k] - hist[k]);
                        }
                    else
                        for (int k = 0; k < K; k++) {
                            prob[k] *= sum;
                            n_cvk[k] += s * prob[k];
                            next_ck[k] += s * prob[k];
                        }
                }
            }
            next_ck = next_ck_buffer.Sum();
            if (FLAGS_vr)
                for (int k = 0; k < K; k++)
                    next_ck[k] += cv_ck[k];

            sample2_time += clk2.toc(); clk2.tic();
            batch_count += 1;
            if (iter < adapt_threshold)
                step_size = GetSchedule(batch_count);
            else
                step_size = -decay / 2 / variance_term;
            step_size = min(step_size, FLAGS_max_step); // Limit the step size
            step_size = max(step_size, FLAGS_min_step);

#pragma omp parallel
            {
#pragma omp for
                for (size_t i = 0; i < cdk.size(); i++) cdk[i] = (1-step_size)*cdk[i] + step_size*next_cdk[i];
#pragma omp for
                for (size_t i = 0; i < cvk.size(); i++) cvk[i] = (1-step_size)*cvk[i] + step_size*next_cvk[i];
            }

            for (auto &d: ck)  d *= (1-step_size);
            for (size_t i = 0; i < ck.size(); i++)  ck[i]  += step_size * next_ck[i];
            next_time += clk2.toc(); clk2.tic();

            ScalarAccumulator<double> fixed_cdk_buffer;
#pragma omp parallel for
            for (int d = 0; d < corpus.D; d++) {
                double doc_sum = 0;
                double *cd = cdk.data() + d * K;
                for (int k = 0; k < K; k++) {
                    if (cd[k] < 0) cd[k] = 0;
                    doc_sum += cd[k];
                }
                fixed_cdk_buffer.Inc(corpus.w[d].size() - doc_sum);
                doc_sum = corpus.w[d].size() / doc_sum;
                if (fabs(doc_sum - 1) > 1e-7)
                    for (int k = 0; k < K; k++)
                        cd[k] *= doc_sum;
            }
            double fixed_cdk = fixed_cdk_buffer.Sum();
            fix_d_time += clk2.toc(); clk2.tic();

            double fixed_ck = 0;
            for (int k = 0; k < K; k++)
                if (ck[k] < 0)
                    fixed_ck += ck[k];

            ScalarAccumulator<double> fixed_cvk_buffer;
#pragma omp parallel for
            for (int k = 0; k < K; k++) {
                double top_sum = 0;
                for (int v = 0; v < corpus.V; v++)
                {
                    double &val = cvk[v*K+k];
                    if (val < 0)
                        val = 0;
                    top_sum += val;
                }
                fixed_cvk_buffer.Inc(ck[k] - top_sum);
                top_sum = ck[k] / top_sum;
                if (fabs(top_sum - 1) > 1e-7)
                    for (int v = 0; v < corpus.V; v++)
                        cvk[v*K+k] *= top_sum;
            }
            double fixed_cvk = fixed_cvk_buffer.Sum();
            if (iiter+1 == FLAGS_num_batches/2) {
                cout << "Fixed cdk " << fixed_cdk << endl;
                cout << "Fixed ck " << fixed_ck << endl;
                cout << "Fixed cvk " << fixed_cvk << endl;
            }
            fix_v_time += clk2.toc(); clk2.tic();

            ThetaPhi();
            thetaphi_time += clk2.toc(); clk2.tic();

//            if (iter == 0)
//                cout << exp(-BatchLogLikelihood() / corpus.T) << endl;
        }
        cout << "Sample time " << sample1_time << ' ' << sample2_time
             << " misc time " << init_time << ' ' << next_time << ' ' << fix_d_time << ' '
             << fix_v_time << ' ' << thetaphi_time << endl;
        auto eval_time = clk.toc();

        double ll = BatchLogLikelihood();
        double perplexity = exp(-ll / corpus.T);
        double tperplexity = iter % 5 == 0 ? BaseLDA::Inference(phi) : 0;

        printf("Log likelihood = %.20f\n", ll);
        cout << "Processed tokens " << num_samples << endl;
        cout << "\e[0;33mIteration " << iter
             << " step size = " << step_size
             << " perplexity = " << perplexity
             << " tperplexity = " << tperplexity
             << " Times " << var_time << ' ' << cv_time << ' ' << eval_time << "\e[0;0m" << endl;
    }
}

double SCVB0::BatchLogLikelihood() 
{
    ThetaPhi();
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

double SCVB0::BatchPerplexity()
{
    double ll = BatchLogLikelihood();
    double perplexity = exp(-ll / corpus.T);
    return perplexity;
}

void SCVB0::ThetaPhi()
{
#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++) {
        auto *th = theta.data() + d * K;
        auto *cd = cdk.data()   + d * K;
        double Z = 1.0 / (corpus.w[d].size() + alpha_bar);
        for (int k = 0; k < K; k++)
            th[k] = (cd[k] + alpha) * Z;
    }
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

void SCVB0::BatchIteration()
{
    Clock clk;
    Accumulator<double> prob_buffer(K);
    Accumulator<double> next_ck_buffer(K);
    fill(next_cdk.begin(), next_cdk.end(), 0);
    fill(next_cvk.begin(), next_cvk.end(), 0);
#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++) {
        auto *prob = prob_buffer.Get();
        auto *ncd  = next_cdk.data() + d * K;
        for (auto v: corpus.w[d]) {
            GetPosterior(theta.data()+d*K, phi.data()+v*K, prob);
            for (int k = 0; k < K; k++)
                ncd[k] += prob[k];
        }
    }
    cout << "Finished D" << endl;
    auto t1 = clk.toc(); clk.tic();
#pragma omp parallel for schedule(static,1)
    for (int v = 0; v < corpus.V; v++) {
        auto *prob = prob_buffer.Get();
        auto *ncv  = next_cvk.data() + v * K;
        auto *nck  = next_ck_buffer.Get();
        for (auto d: corpus.d[v]) {
            GetPosterior(theta.data()+d*K, phi.data()+v*K, prob);
            for (int k = 0; k < K; k++) {
                ncv[k] += prob[k];
                nck[k] += prob[k];
            }
        }
    }
    cout << "Finished V" << endl;
    auto t2 = clk.toc(); clk.tic();
    next_ck  = next_ck_buffer.Sum();
    auto t3 = clk.toc();
    cout << "Sample D " << t1 << " V " << t2 << ' ' << t3 << endl;
}


void SCVB0::VarianceTerm(double step_size, double &var, double &decay)
{
    Clock clk;

    auto theta_0 = theta;
    auto phi_0   = phi;
    int  multi   = FLAGS_num_batches;

    // Compute theta_bar, theta1_bar, theta2_bar, phi1_bar, phi2_bar
    // theta_bar is the batch theta
    // If no CV, theta1_bar is almost theta_bar, except having no alpha. theta1_bar = \sum prob / Zd
    // If CV,    theta1_bar = \sum (prob - hist) / Zd
    Accumulator<double> prob_buffer(K), hist_buffer(K);
    std::vector<double> theta_bar(corpus.D*K), theta1_bar(corpus.D*K), theta2_bar(corpus.D*K);
    std::vector<double> phi_bar(corpus.V*K),   phi1_bar(corpus.V*K),   phi2_bar(corpus.V*K);

#pragma omp parallel for
    for (int d = 0; d < corpus.D; d++) {
        auto prob = prob_buffer.Get();
        auto hist = hist_buffer.Get();
        for (auto v: corpus.w[d]) {
            auto *first_t  = theta_bar.data()  + d * K;
            auto *first1_t = theta1_bar.data() + d * K;
            auto *first2_t = theta2_bar.data() + d * K;

            GetPosterior(theta_0.data()+d*K,    phi_0.data()+v*K,    prob);
            GetPosterior(theta_hist.data()+d*K, phi_hist.data()+v*K, hist);

            double sum_0   = 0;
            double sum_bar = 0;
            double sum_p   = 0;
            for (int k = 0; k < K; k++) {
                double p = prob[k];
                first_t[k]  += p;

                if (FLAGS_vr)
                    p -= hist[k];

                double pp = sqr(p);
                first1_t[k] += p;
                first2_t[k] += pp;
            }
        }
    }
    int MaxT = omp_get_max_threads();
    Accumulator<double> Zv_buffer(K);
#pragma omp parallel for schedule(static, 1)
    for (int v = 0; v < corpus.V; v++) {
        auto prob = prob_buffer.Get();
        auto hist = hist_buffer.Get();
        int thread_id = omp_get_thread_num();
        auto *Zv = Zv_buffer.Get();
        for (auto d: corpus.d[v]) {
            auto *first_p  = phi_bar.data()    + v * K;
            auto *first1_p = phi1_bar.data()   + v * K;
            auto *first2_p = phi2_bar.data()   + v * K;

            GetPosterior(theta_0.data()+d*K,    phi_0.data()+v*K,    prob);
            GetPosterior(theta_hist.data()+d*K, phi_hist.data()+v*K, hist);

            double sum_0   = 0;
            double sum_bar = 0;
            double sum_p   = 0;
            for (int k = 0; k < K; k++) {
                double p = prob[k];
                Zv[k]       += p;
                first_p[k]  += p;

                if (FLAGS_vr)
                    p -= hist[k];

                double pp = sqr(p);
                first1_p[k] += p;
                first2_p[k] += pp;
            }
        }
    }
    auto Zv = Zv_buffer.Sum();

    for (int d = 0; d < corpus.D; d++) {
        auto *first  = theta_bar.data()  + d * K;
        auto *first1 = theta1_bar.data() + d * K;
        auto *first2 = theta2_bar.data() + d * K;
        double Z  = 1.0 / (corpus.w[d].size() + alpha_bar);
        double Z2 = Z * Z;

        for (int k = 0; k < K; k++) {
            first[k]  = (first[k] + alpha) * Z;
            first1[k] *= Z;
            first2[k] *= Z2;
        }
    }
    auto inv_Zv = Zv;
    for (int k = 0; k < K; k++) inv_Zv[k] = 1.0 / (Zv[k] + beta_bar);

    for (int v = 0; v < corpus.V; v++) {
        auto *first  = phi_bar.data()  + v * K;
        auto *first1 = phi1_bar.data() + v * K;
        auto *first2 = phi2_bar.data() + v * K;

        for (int k = 0; k < K; k++) {
            first[k]  = (first[k] + beta) * inv_Zv[k];
            first1[k] *= inv_Zv[k];
            first2[k] *= sqr(inv_Zv[k]);
        }
    }

    std::vector<double> theta_p(theta.size()), phi_p(phi.size());
    for (size_t i = 0; i < theta.size(); i++)
        theta_p[i] = (1-step_size) * theta_0[i] + step_size * theta_bar[i];
    for (size_t i = 0; i < phi.size();   i++)
        phi_p[i]   = (1-step_size) * phi_0[i]   + step_size * phi_bar[i];

    // Compute decay and true_decay
    ScalarAccumulator<double> decay_buffer, true_decay_buffer;
#pragma omp parallel for
    for (int t = 0; t < corpus.T; t++) {
        auto &token = corpus.tokens[t];
        int d = token.d;
        int v = token.v;
        double sum_0   = 0;
        double sum_bar = 0;
        double sum_p   = 0;
        auto *t0 = theta_0.data() + d * K;
        auto *p0 = phi_0.data()   + v * K;
        auto *tb = theta_bar.data() + d * K;
        auto *pb = phi_bar.data()   + v * K;
        auto *tp = theta_p.data() + d * K;
        auto *pp = phi_p.data()   + v * K;
        for (int k = 0; k < K; k++) {
            sum_0   += t0[k] * p0[k];
            sum_bar += tb[k] * pb[k];
            sum_p   += tp[k] * pp[k];
        }
        decay_buffer.Inc((sum_bar - sum_0) / sum_0);
        true_decay_buffer.Inc(log(sum_p) - log(sum_0));
    }
    decay = decay_buffer.Sum();
    auto true_decay = true_decay_buffer.Sum();
    auto init_time = clk.toc();

    // Compute the variance terms
    clk.tic();
    Accumulator<double> dVar_buffer(corpus.D);
    // Deterministic estimation of the first variance
    ScalarAccumulator<double> E_DthetaPhi_sqr_buffer;

#pragma omp parallel for
    for (int t = 0; t < corpus.T; t++) {
        auto &token = corpus.tokens[t];
        auto *dVar = dVar_buffer.Get();
        auto *prob = prob_buffer.Get();
        auto *hist = hist_buffer.Get();

        int d = token.d;
        int v = token.v;
        int D = corpus.w[d].size();
        int M = max(1, D/multi);
        if (D < 2)
            continue;
        auto *first  = theta1_bar.data() + d * K;

        double p = 0;
        for (int k = 0; k < K; k++)
            p += theta_p[d*K+k] * phi_p[v*K+k];
        double mul = -0.5 / p / p;

        double R = 1 - (double)D / M * (M-1) / (D-1);
        double term1 = 0;
        for (int k = 0; k < K; k++)
            term1 += phi_p[v*K+k] * first[k];

        double term2 = 0;
        // Randomly sample a token from the same doc
        int v2 = corpus.w[d][generator() % D];
        double t = 0;
        GetPosterior(theta_0.data()+d*K, phi_0.data()+v2*K, prob);
        if (!FLAGS_vr) {
            for (int k = 0; k < K; k++)
                t += phi_p[v*K+k] * prob[k];
        } else {
            GetPosterior(theta_hist.data()+d*K, phi_hist.data()+v2*K, hist);
            for (int k = 0; k < K; k++)
                t += phi_p[v*K+k] * (prob[k] - hist[k]);

        }
        term2 = t * t * D / ((D + alpha_bar) * (D + alpha_bar));

        E_DthetaPhi_sqr_buffer.Inc(mul * R * (term2 * D - term1 * term1));
        dVar[d]         += mul * (term2 * D - term1 * term1);
    }
    auto E_DthetaPhi_sqr = E_DthetaPhi_sqr_buffer.Sum();
    auto dVar            = dVar_buffer.Sum();

    cout << "First variance is " << E_DthetaPhi_sqr << endl;

    ScalarAccumulator<double> E_ThetaDPhi_sqr_buffer;
    Accumulator<double> vVar_buffer(corpus.V);
#pragma omp parallel for
    for (int t = 0; t < corpus.T; t++) {
        auto &token = corpus.tokens[t];
        auto *vVar = vVar_buffer.Get();
        auto *prob = prob_buffer.Get();
        auto *hist = hist_buffer.Get();

        int d = token.d;
        int v = token.v;
        int V = corpus.d[v].size();
        int M = max(1, V/multi);
        if (V < 2)
            continue;

        auto *first  = phi1_bar.data() + v * K;

        double p = 0;
        for (int k = 0; k < K; k++)
            p += theta_p[d*K+k] * phi_p[v*K+k];
        double mul = -0.5 / p / p;

        double R = 1 - (double)V / M * (M-1) / (V-1);
        //double R = 1.0 / M;   // Approximate
        double term1 = 0;
        for (int k = 0; k < K; k++)
            term1 += theta_p[d*K+k] * first[k];

        double term2 = 0;

        auto GetTerm2 = [&](int d2) {
            double t = 0;
            GetPosterior(theta_0.data() + d2 * K, phi_0.data() + v * K, prob);
            if (!FLAGS_vr) {
                for (int k = 0; k < K; k++)
                    t += theta_p[d * K + k] * prob[k] * inv_Zv[k];
            } else {
                GetPosterior(theta_hist.data() + d2 * K, phi_hist.data() + v * K, hist);
                for (int k = 0; k < K; k++)
                    t += theta_p[d * K + k] * (prob[k] - hist[k]) * inv_Zv[k];
            }
            return t * t;
        };
        if (V < 10)
            for (auto d2: corpus.d[v])
                term2 += GetTerm2(d2);
        else
            term2 = GetTerm2(corpus.d[v][generator() % V]) * V;

        E_ThetaDPhi_sqr_buffer.Inc(mul * R * (term2 * V - term1 * term1));
        vVar[v]         += mul * (term2 * V - term1 * term1);
    }
    auto E_ThetaDPhi_sqr = E_ThetaDPhi_sqr_buffer.Sum();
    auto vVar            = vVar_buffer.Sum();
    cout << "Second variance is " << E_ThetaDPhi_sqr << endl;
    //cout << "Second variance is " << E_ThetaDPhi_sqr << " emp value is " << second_var << endl;
    auto compute2_time = clk.toc();

    double E_DTheta_sqr = 0;
    double prior_decay_theta = 0;
    double E_DPhi_sqr = 0;
    double prior_decay_phi = 0;
    if (FLAGS_prior) {
        int fixed_d = 0;
        for (int d = 0; d < corpus.D; d++) {
            int D = corpus.w[d].size();
            if (D < 2)
                continue;
            int M = max(1, D / multi);
            double R = 1 - (double) D / M * (M - 1) / (D - 1);
            double theta_lb = alpha / (D + alpha_bar);

            for (int k = 0; k < K; k++) {
                double T1 = theta1_bar[d * K + k];
                double T2 = theta2_bar[d * K + k];
                double tp = theta_p[d * K + k];

                double max_delta = (tp - theta_lb) / step_size;
                double max_var   = -0.5 * alpha * max_delta * max_delta / (tp * tp);
                double inc0      = -0.5 * alpha * (T2 * D - T1 * T1) / (tp * tp);
                double inc       = inc0 * R;
                if (abs(inc) > abs(max_var)) {
                    E_DTheta_sqr += max_var;
                    dVar[d]      += max_var / R;
                    ++fixed_d;
                } else {
                    E_DTheta_sqr += inc;
                    dVar[d]      += inc0;
                }
            }
        }
        for (int d = 0; d < corpus.D; d++)
            for (int k = 0; k < K; k++)
                prior_decay_theta += alpha * (log(theta_p[d * K + k]) - log(theta_0[d * K + k]));

//        cout << "Prior Phi " << E_DPhi_sqr << endl;
        int fixed_v = 0;
        for (int v = 0; v < corpus.V; v++) {
            int V = corpus.d[v].size();
            if (V < 2)
                continue;
            int M = max(1, V / multi);
            double R = 1 - (double) V / M * (M - 1) / (V - 1);

            for (int k = 0; k < K; k++) {
                double T1 = phi1_bar[v * K + k];
                double T2 = phi2_bar[v * K + k];
                double tp = phi_p[v * K + k];

                double phi_lb = beta / (beta_bar + Zv[k]);
                double max_delta = (tp - phi_lb) / step_size;
                double max_var   = -0.5 * beta * max_delta * max_delta / (tp * tp);
                double inc0      = -0.5 * beta * (T2 * V - T1 * T1) / (tp * tp);
                double inc       = inc0 * R;
                if (abs(inc) > abs(max_var)) {
                    E_DPhi_sqr += max_var;
                    vVar[v]    += max_var / R;
                    ++fixed_v;
                } else {
                    E_DPhi_sqr += inc;
                    vVar[v]    += inc0;
                }
            }
        }
        for (int v = 0; v < corpus.V; v++)
            for (int k = 0; k < K; k++)
                prior_decay_phi += beta * (log(phi_p[v * K + k]) - log(phi_0[v * K + k]));
        cout << "Fixed " << fixed_d << " D and " << fixed_v << " V." << endl;
    }
    for (int d = 0; d < corpus.D; d++)
        if (dVar[d] > 0)
            dVar[d] = 0;
    for (int v = 0; v < corpus.V; v++)
        if (vVar[v] > 0)
            vVar[v] = 0;
    cout << "Prior variance terms " << E_DTheta_sqr << ' ' << E_DPhi_sqr << endl;
    cout << (E_ThetaDPhi_sqr + E_DthetaPhi_sqr) * step_size * step_size << endl;
    cout << true_decay << ' ' << prior_decay_theta << ' ' << prior_decay_phi << endl;

    clk.tic();
    // Alternative allocation of sampling frequency
    dFreq = std::vector<int>(corpus.D, 1);
    vFreq = std::vector<int>(corpus.V, 1);
    AdaptiveSchedule(dVar, vVar);
    double alternative_var  = 0;
    double alternative_var2 = 0;
    for (int d = 0; d < corpus.D; d++)
        alternative_var  += dVar[d] / dFreq[d];
    for (int v = 0; v < corpus.V; v++)
        alternative_var2 += vVar[v] / vFreq[v];
    
    cout << "Alternative first variance is " <<  alternative_var << endl;
    cout << "Alternative second variance is " << alternative_var2 << endl;

    //var = (E_DthetaPhi_sqr + E_ThetaDPhi_sqr) * step_size * step_size;
    if (FLAGS_adapt_dist)
        var = (alternative_var + alternative_var2) * step_size * step_size;
    else
        var = (E_ThetaDPhi_sqr + E_DTheta_sqr + E_DthetaPhi_sqr + E_DPhi_sqr) * step_size * step_size;
    cout << "Decay is " << decay * step_size
         << " True decay is " << true_decay << ' ' << prior_decay_theta << ' ' << prior_decay_phi
         << " var is " << var << endl;

    auto freq_time = clk.toc();

//    decay = true_decay / step_size;     // Use true expectation term
    decay = (true_decay + prior_decay_theta + prior_decay_phi) / step_size;
    cout << "Times " << init_time << ' ' << compute2_time << ' ' << freq_time << endl;
}

void SCVB0::AdaptiveSchedule(std::vector<double> &dVar, std::vector<double> &vVar) {
    // Greedy select of largest marginal benefit
    std::priority_queue<pair<double, int>> heap;

    auto vMarginalBenefit = [&](int v) {
        int V   = corpus.d[v].size();
        int cur = vFreq[v];
        double next_R = 1 - (double)V / (cur+1) * cur / (V-1);
        double cur_R  = 1 - (double)V / cur * (cur-1) / (V-1);
        double mb = vVar[v] * (next_R - cur_R);
        return mb;
    };

    auto dMarginalBenefit = [&](int d) {
        int D   = corpus.w[d].size();
        int cur = dFreq[d];
        double next_R = 1 - (double)D / (cur+1) * cur / (D-1);
        double cur_R  = 1 - (double)D / cur * (cur-1) / (D-1);
        double mb = dVar[d] * (next_R - cur_R);
        return mb;
    };

    for (int d = 0; d < corpus.D; d++) {
        dFreq[d] = 1;
        if (corpus.w[d].size() > 1)
            heap.emplace(dMarginalBenefit(d), d + 1048576);
    }

    for (int v = 0; v < corpus.V; v++) {
        vFreq[v] = 1;
        if (corpus.d[v].size() > 1)
            heap.emplace(vMarginalBenefit(v), v);
    }

    int budget = batch_size - corpus.V - corpus.D;
    while (budget) {
        int vd = heap.top().second;
        heap.pop();
        if (vd >= 1048576) {
            vd -= 1048576;
            dFreq[vd]++;
            if (dFreq[vd] < corpus.w[vd].size())
                heap.emplace(dMarginalBenefit(vd), vd + 1048576);
        } else {
            vFreq[vd]++;
            if (vFreq[vd] < corpus.d[vd].size())
                heap.emplace(vMarginalBenefit(vd), vd);
        }
        budget--;
    }
}
