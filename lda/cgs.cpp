//
// Created by jianfei on 18-1-23.
//

#include <iostream>
#include <random>
#include <deque>
#include "cgs.h"
#include "corpus.h"
#include "utils.h"
#include "flags.h"
using namespace std;

CGS::CGS(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
         int K, float alpha, float beta)
        : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
          cdk(corpus.D*K, 0), cvk(corpus.V*K, 0), ck(K, 0)
{
    for (int d = 0; d < corpus.D; d++)
        for (auto v: corpus.w[d]) {
            token.k = generator() % K;
            cdk[token.d*K + token.k]++;
            cvk[token.v*K + token.k]++;
            ck[token.k]++;
        }
}

void CGS::Estimate()
{
    std::vector<double> prob(K);

    std::deque<std::vector<double> > phi_ma;
    std::vector<double> phi_sum(corpus.V * K);

    for (int iter = 0; iter < FLAGS_num_iters; iter++) {
        for (auto &token: corpus.tokens) {
            cdk[token.d*K + token.k]--;
            cvk[token.v*K + token.k]--;
            ck[token.k]--;
            double sum = 0;
            for (int k = 0; k < K; k++) {
                double theta = cdk[token.d*K + k] + alpha;
                double phi   = (cvk[token.v*K + k] + beta) /
                               (ck[k] + beta_bar);
                prob[k] = sum += theta * phi;
            }
            double u = u01(generator) * sum;
            int k = 0;
            while (k < K-1 && prob[k]<u) k++;
            token.k = k;

            cdk[token.d*K + k]++;
            cvk[token.v*K + k]++;
            ck[k]++;
        }

        // Compute perplexity
        vector<double> theta(corpus.D * K);
        vector<double> phi  (corpus.V * K);
        for (int d = 0; d < corpus.D; d++) {
            double sum = 0;
            for (int k = 0; k < K; k++) {
                double val = cdk[d * K + k] + alpha;
                theta[d * K + k] = val;
                sum += val;
            }
            for (int k = 0; k < K; k++)
                theta[d * K + k] /= sum;
        }
        for (int v = 0; v < corpus.V; v++)
            for (int k = 0; k < K; k++)
                phi[v * K + k] = (cvk[v * K + k] + beta) /
                                 (ck[k] + beta_bar);

        phi_ma.push_back(phi);
        for (int i = 0; i < corpus.V * K; i++) phi_sum[i] += phi[i];
        if (phi_ma.size() > 10) {
            auto &p = phi_ma.front();
            for (int i = 0; i < corpus.V * K; i++) phi_sum[i] -= p[i];
            phi_ma.pop_front();
        }

        std::vector<double> new_phi = phi_sum;
        for (int i = 0; i < corpus.V * K; i++) new_phi[i] *= 0.1;

        double ll = 0;
        for (auto &token: corpus.tokens) {
            double l = 0;
            for (int k = 0; k < K; k++)
                l += theta[token.d*K + k] * new_phi[token.v*K + k];
            ll += log(l);
        }
        double perplexity = exp(-ll / corpus.T);
        double test_perplexity = Inference(new_phi);
        cout << "Iteration " << iter
             << " perplexity = " << perplexity
             << " test perplexity = " << test_perplexity << endl;
    }
}
