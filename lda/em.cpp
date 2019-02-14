//
// Created by jianfei on 2018/1/25.
//

#include "em.h"
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
using namespace std;

EM::EM(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
       int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          cdk(corpus.D * K), cvk(corpus.V * K), ck(K),
          theta(corpus.D * K), phi(corpus.V * K),
          delta_theta(corpus.D * K), delta_phi(corpus.V * K),
          next_cdk(corpus.D * K), next_cvk(corpus.V * K), next_ck(K)
{
    for (size_t t = 0; t < corpus.T; t++) {
        auto &token = corpus.tokens[t];

        double *cd = cdk.data() + token.d * K;
        double *cv = cvk.data() + token.v * K;

        int k = generator() % K;
        cd[k]++;
        cv[k]++;
        ck[k]++;
    }
    UpdateParams();
    cout << "Initialization Finished" << endl;
}

void EM::Estimate()
{
    vector<double> g(K);

    vector<double> idk(corpus.D * K), ivk(corpus.V * K);        // Observed Fisher information
    vector<double> icdk(corpus.D * K), icvk(corpus.V * K);      // Complete Fisher information
    for (int iter = 0; iter < FLAGS_num_iters; iter++) {
        fill(next_cdk.begin(), next_cdk.end(), 0);
        fill(next_cvk.begin(), next_cvk.end(), 0);
        fill(next_ck.begin(), next_ck.end(), 0);
        fill(idk.begin(), idk.end(), 0);
        fill(ivk.begin(), ivk.end(), 0);
        for (size_t t = 0; t < corpus.T; t++) {
            auto &token = corpus.tokens[t];
            double *thetad = theta.data() + token.d * K;
            double *phiv = phi.data() + token.v * K;
            double *id = idk.data() + token.d * K;
            double *iv = ivk.data() + token.v * K;
            double *next_cd = next_cdk.data() + token.d * K;
            double *next_cv = next_cvk.data() + token.v * K;

            double sum = 0;
            for (int k = 0; k < K; k++)
                sum += g[k] = thetad[k] * phiv[k];

            sum = 1.0 / sum;
            for (int k = 0; k < K; k++) {
                g[k] *= sum;
                next_cd[k] += g[k];
                next_cv[k] += g[k];
                next_ck[k] += g[k];

                // Update observed Fisher information
                id[k] += phiv[k]*phiv[k] * sum*sum;
                iv[k] += thetad[k]*thetad[k] * sum*sum;
            }
        }
        // Compute 2nd order derivative by finite difference
        double eps = 1e-3;
        phi[0] += eps;
        double Lp = LogLC();
        phi[0] -= eps;
        double L0 = LogLC();
        phi[0] -= eps;
        double Ln = LogLC();
        phi[0] += eps;
        double est_idk = (Lp-2*L0+Ln) / sqr(eps);
        cout << L0 << endl;

        // Compute complete Fisher information
        for (int d = 0; d < corpus.D; d++)
            for (int k = 0; k < K; k++)
                icdk[d*K+k] = (next_cdk[d*K+k] + alpha) / sqr(theta[d*K+k]);
        for (int v = 0; v < corpus.V; v++)
            for (int k = 0; k < K; k++)
                icvk[v*K+k] = (next_cvk[v*K+k] + beta) / sqr(phi[v*K+k]);

        // Write some files
        WriteFile("delta_theta", iter, delta_theta, corpus.D, K);
        WriteFile("idk", iter, idk, corpus.D, K);
        WriteFile("icdk", iter, icdk, corpus.D, K);
        WriteFile("delta_phi", iter, delta_phi, corpus.V, K);
        WriteFile("ivk", iter, ivk, corpus.V, K);
        WriteFile("icvk", iter, icvk, corpus.V, K);

        WriteFile("theta", iter, theta, corpus.D, K);
        WriteFile("phi",   iter, phi,   corpus.V, K);

        // Compute convergence rate (1 - Ic^-1 I)^2 I
        double convergence = 0;
        double weight = 0;
        for (size_t i = 0; i < cdk.size(); i++) {
            double oi = icdk[i] - idk[i];
            double c = sqr(1 - idk[i]/icdk[i]);
            double w = sqr(theta[i]) * oi;
//            double w = sqr(delta_theta[i]) * oi;
//            double w = sqr(delta_theta[i]);
//            double w = oi;
            convergence += w * c;
            weight += w;
        }
        for (size_t i = 0; i < cvk.size(); i++) {
            double oi = icvk[i] - ivk[i];
            double c = sqr(1 - ivk[i]/icvk[i]);
            double w = sqr(phi[i]) * oi;
//            double w = sqr(delta_phi[i]) * oi;
//            double w = sqr(delta_phi[i]);
//            double w = oi;
            convergence += w * c;
            weight += w;
        }

        cdk.swap(next_cdk);
        cvk.swap(next_cvk);
        ck.swap(next_ck);

        // Compute perplexity
        UpdateParams();

        double ll = LogLikelihood();
        double perplexity = exp(-ll / corpus.T);
        double test_perplexity = Inference(phi);
        cout << "Iteration " << iter
             << " perplexity = " << perplexity
             << " test perplexity = " << test_perplexity
             << " convergence = " << convergence / weight << endl;
    }
}

void EM::UpdateParams() {
    std::vector<double> best_theta(theta.size());
    std::vector<double> best_phi(phi.size());
    if (FLAGS_theta_file != "") {
        ifstream fin(FLAGS_theta_file.c_str());
        for (size_t d = 0; d < theta.size(); d++)
            fin >> best_theta[d];
    }
    if (FLAGS_phi_file != "") {
        ifstream fin(FLAGS_phi_file.c_str());
        for (size_t d = 0; d < phi.size(); d++)
            fin >> best_phi[d];
    }
    for (int d = 0; d < corpus.D; d++)
        for (int k = 0; k < K; k++) {
            double old_theta = theta[d * K + k];
            theta[d * K + k] = (cdk[d * K + k] + alpha) / (corpus.w[d].size() + alpha_bar);
            delta_theta[d * K + k] = theta[d * K + k] - best_theta[d * K + k];
//            delta_theta[d * K + k] = theta[d * K + k] - old_theta;
        }
    for (int v = 0; v < corpus.V; v++)
        for (int k = 0; k < K; k++) {
            double old_phi = phi[v * K + k];
            phi[v * K + k] = (cvk[v * K + k] + beta) / (ck[k] + beta_bar);
//            delta_phi[v * K + k] = phi[v * K + k] - old_phi;
            delta_phi[v * K + k] = phi[v * K + k] - best_phi[v * K + k];
        }
}

void EM::WriteFile(std::string prefix, int iter, std::vector<double> &a, int R, int C) {
    ofstream fout(prefix + "." + to_string(iter));
    for (int r = 0; r < R; r++) {
        for (int c = 0; c < C; c++)
            fout << a[r*C+c] << ' ';
        fout << endl;
    }
}

double EM::LogLikelihood() {
    double ll = 0;
    for (int d = 0; d < corpus.D; d++)
        for (size_t n = 0; n < corpus.w[d].size(); n++) {
            auto v = corpus.w[d][n];
            double l = 0;
            for (int k = 0; k < K; k++)
                l += theta[d*K + k] * phi[v*K + k];
            l   = log(l);
            ll += l;
        }
    return ll;
}

double EM::LogLC() {
    double ll = 0;
    for (int d = 0; d < corpus.D; d++)
        for (int k = 0; k < K; k++)
            ll += next_cdk[d*K+k] * log(theta[d*K+k]);
    for (int v = 0; v < corpus.V; v++)
        for (int k = 0; k < K; k++)
            ll += next_cvk[v*K+k] * log(phi[v*K+k]);
    return ll;
}