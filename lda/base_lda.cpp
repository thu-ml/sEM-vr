//
// Created by jianfei on 2018/1/26.
//

#include "base_lda.h"
#include "corpus.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <exception>
#include <iostream>
#include "gflags/gflags.h"
#include "utils.h"
#include "flags.h"
#include "accumulator.h"
#include "omp.h"
using namespace std;

BaseLDA::BaseLDA(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
                 int K, float alpha, float beta)
    : corpus(corpus), to_corpus(to_corpus), th_corpus(th_corpus),
      K(K), alpha(alpha), beta(beta), alpha_bar(alpha*K), beta_bar(beta*corpus.V),
      running_time(0)
{

}

double BaseLDA::GetSchedule(int t) {
    double schedule = FLAGS_sch_a * pow(1.0 / (t+FLAGS_sch_t0), FLAGS_sch_p);
    if (schedule >= 1)
        throw runtime_error("Learning rate is too large");
    return schedule;
}

double BaseLDA::CVB0Inference(const std::vector<double> &phi, Corpus &to_corpus) {
    Accumulator<double> theta_buffer(K);
    Accumulator<double> prob_buffer(K);
    Accumulator<double> all_theta_buffer(K);
    Accumulator<double> new_theta_buffer(K);
    ScalarAccumulator<double> log_likelihood_buffer;

    Accumulator<double> lls_buffer(corpus.V);
#pragma omp parallel for
    for (int d = 0; d < to_corpus.D; d++) {
        auto *theta     = theta_buffer.Get();
        auto *prob      = prob_buffer.Get();
        auto *all_theta = all_theta_buffer.Get();
        auto *new_theta = new_theta_buffer.Get();

        auto &doc = to_corpus.w[d];
        int N = doc.size();

        double sum = 0;
        for (int k = 0; k < K; k++)
            sum += theta[k] = 100 + u01(generator);
        sum = 1.0 / sum;
        for (int k = 0; k < K; k++)
            theta[k] *= sum;

        fill(all_theta, all_theta+K, 0);
        for (int iter = 0; iter < FLAGS_num_iiters*2; iter++) {
            fill(new_theta, new_theta+K, alpha);
            for (auto v: doc) {
                sum = 0;
                auto *ph = phi.data() + v * K;
                for (int k = 0; k < K; k++)
                    sum += prob[k] = theta[k] * ph[k];
                sum = 1.0 / sum;
                for (int k = 0; k < K; k++)
                    new_theta[k] += prob[k] * sum;
            }
            for (int k = 0; k < K; k++)
                theta[k] = new_theta[k] / (N + alpha_bar);
            if (iter >= FLAGS_num_iiters)
                for (int k = 0; k < K; k++)
                    all_theta[k] += theta[k];
        }
        for (int k = 0; k < K; k++)
            theta[k] = all_theta[k] / FLAGS_num_iiters;

        auto *lls = lls_buffer.Get();
        for (size_t n = 0; n < to_corpus.w[d].size(); n++) {
            auto v = to_corpus.w[d][n];
            auto *ph = phi.data() + v * K;
            double l = 0;
            for (int k = 0; k < K; k++)
                l += theta[k] * ph[k];
            log_likelihood_buffer.Inc(log(l));
            lls[v] += log(l);
        }
    }
    auto lls_sum = lls_buffer.Sum();
    double log_likelihood = log_likelihood_buffer.Sum();
    return log_likelihood;
//    return exp(-log_likelihood / to_corpus.T);
}

double BaseLDA::LeftToRightInference(const std::vector<double> &phi, Corpus &to_corpus) {
    int NP = FLAGS_num_particles;
    Accumulator<double> prob_buffer(K);
    Accumulator<int> cdk_buffer(K * NP);
    ScalarAccumulator<double> log_likelihood_buffer;

    std::vector<std::vector<int> > chains(omp_get_max_threads());

    int max_steps = 500;

    Accumulator<double> lls_buffer(corpus.V);
#pragma omp parallel for schedule(dynamic, 1)
    for (int d = 0; d < to_corpus.D; d++) {
        auto &chain = chains[omp_get_thread_num()];
        auto *prob = prob_buffer.Get();
        auto *cdk = cdk_buffer.Get();
        fill(cdk, cdk + K * NP, 0);
        std::vector<int> steps;

        auto &doc = to_corpus.w[d];
        int N = doc.size();
        chain.resize(N * NP);

        for (int n = 0; n < to_corpus.w[d].size(); n++) {
            double pn = 0;
            for (int particle = 0; particle < NP; particle++) {
                auto *cd = cdk + particle * K;
                auto *z = chain.data() + particle * N;
                for (int n2 = 0; n2 < n; n2++) {
                    int k = z[n2];
                    int v = to_corpus.w[d][n2];
                    auto *ph = phi.data() + v * K;

                    --cd[k];
                    double sum = 0;
                    for (int i = 0; i < K; i++)
                        prob[i] = sum += (cd[i] + alpha) * ph[i];

                    double u = u01(generator) * sum;
                    k = 0;
                    while (k < K - 1 && prob[k] < u) k++;

                    z[n2] = k;
                    ++cd[k];
                }
                double sum = 0;
                double scale = 1.0 / (alpha_bar + n);
                int v = to_corpus.w[d][n];
                auto *ph = phi.data() + v * K;
                for (int i = 0; i < K; i++) {
                    double uterm = (cd[i] + alpha) * ph[i];
                    prob[i] = sum += uterm;
                    pn += uterm * scale;
                }
                double u = u01(generator) * sum;
                int k = 0;
                while (k < K - 1 && prob[k] < u) k++;
                z[n] = k;
                ++cd[k];
            }
            log_likelihood_buffer.Inc(log(pn / NP));
        }
    }
    double log_likelihood = log_likelihood_buffer.Sum();
    return log_likelihood;
//    return exp(-log_likelihood / to_corpus.T);
}

double BaseLDA::Inference(const std::vector<double> &phi) {
    return Inference(phi, to_corpus);
}

double BaseLDA::Inference(const std::vector<double> &phi, Corpus &to_corpus) {
//    return CVBELBO(phi);
//    return LeftToRightInference(phi, to_corpus);
    return CVB0Inference(phi, to_corpus);

    cout << "Running CVB" << endl;
    Accumulator<double> prob_buffer(K);
    Accumulator<double> cdk_buffer(K);
    Accumulator<double> var_buffer(K);
    Accumulator<double> theta_buffer(K);
    std::vector<std::vector<double> > gamma_buffer(omp_get_max_threads());

    ScalarAccumulator<double> log_likelihood_buffer;

#pragma omp parallel for
    for (int d = 0; d < to_corpus.D; d++) {
        auto &gamma = gamma_buffer[omp_get_thread_num()];
        auto *prob = prob_buffer.Get();
        auto *cd = cdk_buffer.Get();
        auto *var = var_buffer.Get();
        auto &doc = to_corpus.w[d];
        auto *th = theta_buffer.Get();
        int N = doc.size();
        fill(cd, cd + K, alpha);
        fill(var, var + K, 0);
        gamma.resize(N * K);
        fill(gamma.begin(), gamma.end(), 0);

        // Initialize
        for (int n = 0; n < to_corpus.w[d].size(); n++) {
            auto *gam = gamma.data() + n * K;
            int k = generator() % K;
            gam[k]++;
            cd[k]++;
        }
        // CVB Inference
        for (int iiter = 0; iiter < FLAGS_num_iiters; iiter++) {
            for (int n = 0; n < to_corpus.w[d].size(); n++) {
                int v = to_corpus.w[d][n];
                auto *ph = phi.data() + v * K;
                auto *prob = gamma.data() + n * K;
                double sum = 0;

                for (int k = 0; k < K; k++) {
                    cd[k] -= prob[k];
                    var[k] -= prob[k] * (1 - prob[k]);
                }
                for (int k = 0; k < K; k++)
                    sum += prob[k] = ph[k] * cd[k] * exp(-var[k] / (2 * sqr(cd[k])));
                sum = 1.0 / sum;
                for (int k = 0; k < K; k++) {
                    cd[k] += prob[k] *= sum;
                    var[k] += prob[k] * (1 - prob[k]);
                }
            }
        }
        double scale = 1.0 / (to_corpus.w[d].size() + alpha_bar);
        for (int k = 0; k < K; k++)
            th[k] = cd[k] * scale;

        // Compute ELBO = Eq log p - log q
        for (int n = 0; n < th_corpus.w[d].size(); n++) {
            int v = th_corpus.w[d][n];
            auto *ph = phi.data() + v * K;
            double sum = 0;
            for (int k = 0; k < K; k++)
                sum += th[k] * ph[k];
            log_likelihood_buffer.Inc(log(sum));
        }
    }
    double log_likelihood = log_likelihood_buffer.Sum();
    return exp(-log_likelihood / th_corpus.T);
}

double BaseLDA::CVBELBO(const std::vector<double> &phi) {
    cout << "Computing CVB ELBO" << endl;
    Accumulator<double> prob_buffer(K);
    Accumulator<double> cdk_buffer(K);
    Accumulator<double> var_buffer(K);
    std::vector <std::vector<double>> gamma_buffer(omp_get_max_threads());

    ScalarAccumulator<double> log_likelihood_buffer;

#pragma omp parallel for
    for (int d = 0; d < to_corpus.D; d++) {
        auto &gamma = gamma_buffer[omp_get_thread_num()];
        auto *prob = prob_buffer.Get();
        auto *cd = cdk_buffer.Get();
        auto *var = var_buffer.Get();
        auto &doc = to_corpus.w[d];
        int N = doc.size();
        fill(cd, cd + K, alpha);
        fill(var, var + K, 0);
        gamma.resize(N * K);
        fill(gamma.begin(), gamma.end(), 0);

        // Initialize
        for (int n = 0; n < to_corpus.w[d].size(); n++) {
            auto *gam = gamma.data() + n * K;
            int k = generator() % K;
            gam[k]++;
            cd[k]++;
        }
        // CVB Inference
        for (int iiter = 0; iiter < FLAGS_num_iiters; iiter++) {
            for (int n = 0; n < to_corpus.w[d].size(); n++) {
                int v = to_corpus.w[d][n];
                auto *ph = phi.data() + v * K;
                auto *prob = gamma.data() + n * K;
                double sum = 0;

                for (int k = 0; k < K; k++) {
                    cd[k] -= prob[k];
                    var[k] -= prob[k] * (1 - prob[k]);
                }
                for (int k = 0; k < K; k++)
                    sum += prob[k] = ph[k] * cd[k] * exp(-var[k] / (2 * sqr(cd[k])));
                sum = 1.0 / sum;
                for (int k = 0; k < K; k++) {
                    cd[k] += prob[k] *= sum;
                    var[k] += prob[k] * (1 - prob[k]);
                }
            }
        }

        // Eq log p(z|alpha)
        double ll = 0;
        for (int k = 0; k < K; k++) {
            ll += lgamma(cd[k]) - var[k] / (2 * sqr(cd[k]));
        }
        ll -= K * lgamma(alpha);
        ll += lgamma(alpha_bar);
        ll -= lgamma(alpha_bar + N);

        // Eq log p(w | z, phi) - log q
        for (int n = 0; n < to_corpus.w[d].size(); n++) {
            auto *prob = gamma.data() + n * K;
            int v = to_corpus.w[d][n];
            auto *ph = phi.data() + v * K;
            for (int k = 0; k < K; k++)
                ll += prob[k] * (log(ph[k]) - log(prob[k]));
        }
        log_likelihood_buffer.Inc(ll);
    }
    double log_likelihood = log_likelihood_buffer.Sum();
    return exp(-log_likelihood / to_corpus.T);
}

void BaseLDA::PositiveProjection(std::vector<double> &cvk) {
#pragma omp parallel for
    for (int v = 0; v < corpus.V; v++) {
        auto *cv = cvk.data() + v * K;
        for (int k = 0; k < K; k++)
            cv[k] = max(cv[k], 0.0);
    }
}

void BaseLDA::RenormalizeProjection(std::vector<double> &cvk, std::vector<double> &phi) {
    Accumulator<double> ck_buffer(K);
#pragma omp parallel for
    for (int v = 0; v < corpus.V; v++) {
        auto *cv = cvk.data() + v * K;
        auto *ck = ck_buffer.Get();
        double old_sum = 0, new_sum = 0;
        for (int k = 0; k < K; k++) {
            old_sum += cv[k];
            cv[k] = max(cv[k], 0.0);
            new_sum += cv[k];
        }
        if (new_sum < 1e-10) {
            double con = corpus.d[v].size() / K;
            for (int k = 0; k < K; k++)
                ck[k] += cv[k] = con;
            continue;
        }
        double scale = old_sum / new_sum;
        for (int k = 0; k < K; k++) {
            cv[k] *= scale;
            ck[k] += cv[k];
        }
    }

//    for (int v = 0; v < corpus.V; v++) {
//        auto *cv = cvk.data() + v * K;
//        double sum = 0;
//        for (int k = 0; k < K; k++)
//            sum += cv[k];
//        if (fabs(sum - corpus.d[v].size()) > 1e-6) {
//            cout << sum << ' ' << corpus.d[v].size() << endl;
//            for (int k = 0; k < K; k++)
//                cout << cv[k] << ' ';
//            cout << endl;
//            exit(0);
//        }
//    }

    auto ck = ck_buffer.Sum();
    auto Zv = ck;
    for (int k = 0; k < K; k++)
        Zv[k] = 1.0 / (ck[k] + beta_bar);
    #pragma omp parallel for
    for (int v = 0; v < corpus.V; v++) {
        double *ph = phi.data() + v * K;
        auto *cv = cvk.data() + v * K;
        for (int k = 0; k < K; k++)
            ph[k] = (cv[k] + beta) * Zv[k];
    }

    double ck_sum = accumulate(ck.begin(), ck.end(), 0.0);
    cout << "Ck sum " << ck_sum << endl;

//    Accumulator<double> ck_buffer(K);
//    Accumulator<double> ck_buffer_pos(K);
//#pragma omp parallel for
//    for (int v = 0; v < corpus.V; v++) {
//        auto *ck = ck_buffer.Get();
//        auto *ck_pos = ck_buffer_pos.Get();
//        auto *cv = cvk.data() + v * K;
//        for (int k = 0; k < K; k++) {
//            ck[k] += cv[k];
//            ck_pos[k] += cv[k] = max(cv[k], 0.0);
//        }
//    }
//    auto ck = ck_buffer.Sum();
//    auto ck_pos = ck_buffer_pos.Sum();
//    auto scale = ck_pos;
//    for (int k = 0; k < K; k++)
//        scale[k] = ck[k] / ck_pos[k];
//    auto Zv = ck;
//    for (int k = 0; k < K; k++)
//        Zv[k] = 1.0 / (ck[k] + beta_bar);
//
//#pragma omp parallel for
//    for (int v = 0; v < corpus.V; v++) {
//        double *ph = phi.data() + v * K;
//        auto *cv = cvk.data() + v * K;
//        for (int k = 0; k < K; k++) {
//            cv[k] *= scale[k];
//            ph[k] = (cv[k] + beta) * Zv[k];
//        }
//    }
//
//    double ck_sum = accumulate(ck.begin(), ck.end(), 0.0);
//    double ck_pos_sum = accumulate(ck_pos.begin(), ck_pos.end(), 0.0);
//    cout << "Ck sum " << ck_sum << " ck pos " << ck_pos_sum << " gap " << ck_pos_sum - ck_sum << endl;
}

template <class T>
void OutputField(string name, T val, bool is_first = false) {
    if (!is_first) cout << ", ";
    cout << "\"" << name << "\":" << val;
}

void BaseLDA::Eval(int iter, int subiter, std::vector<double> &theta, std::vector<double> &phi) {
    bool should_evaluate = subiter == -1;
    if (FLAGS_test_lag != -1 && (subiter+1) % FLAGS_test_lag == 0) should_evaluate = true;
    if (!should_evaluate) return;

    running_time += clock.toc();
    Clock test_clk;
    cout << "{";
    OutputField("iter", iter, true);
    OutputField("subiter", subiter);
    OutputField("running_time", running_time);
    OutputField("step_size", step_size);

    if (FLAGS_test_on_train) {
        double training_ll;
        if (FLAGS_test_alg == "lr")
            training_ll = LeftToRightInference(phi, th_corpus);
        else
            training_ll = CVB0Inference(phi, th_corpus);

        auto training_ppl = exp(-training_ll / th_corpus.T);
        OutputField("train_ll", training_ll);
        OutputField("train_ppl", training_ppl);
    }
    double testing_ll;
    if (FLAGS_test_alg == "lr")
        testing_ll = LeftToRightInference(phi, to_corpus);
    else
        testing_ll = CVB0Inference(phi, to_corpus);
    auto testing_ppl = exp(-testing_ll / to_corpus.T);
    OutputField("test_ll", testing_ll);
    OutputField("test_ppl", testing_ppl);

    if (subiter == -1) {
        // Compute prior
        ScalarAccumulator<double> theta_prior_acc, phi_prior_acc;
#pragma omp parallel for
        for (int d = 0; d < corpus.D; d++) {
            auto *th = theta.data() + d * K;
            for (int k = 0; k < K; k++)
                theta_prior_acc.Inc(log(th[k]));
        }
#pragma omp parallel for
        for (int v = 0; v < corpus.V; v++) {
            auto *ph = phi.data() + v * K;
            for (int k = 0; k < K; k++)
                phi_prior_acc.Inc(log(ph[k]));
        }
        auto theta_prior = alpha * theta_prior_acc.Sum();
        auto phi_prior = beta * phi_prior_acc.Sum();
        auto likelihood = BatchLogLikelihood(theta, phi);
        OutputField("theta_prior", theta_prior);
        OutputField("phi_prior", phi_prior);
        OutputField("log_likelihood", likelihood);
        OutputField("log_posterior", likelihood + theta_prior + phi_prior);
        OutputField("joint_ppl", exp(-likelihood / corpus.T));
    }
    auto testing_time = test_clk.toc();
    OutputField("testing_time", testing_time);
    cout << "}" << endl;

    clock.tic();
}

void BaseLDA::Start() {
    clock.tic();
}

double BaseLDA::BatchLogLikelihood(std::vector<double> &theta, std::vector<double> &phi)
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