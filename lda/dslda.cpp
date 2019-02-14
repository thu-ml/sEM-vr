//
// Created by jianfei on 18-1-23.
//

#include "dslda.h"
#include "corpus.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <exception>
#include "gflags/gflags.h"
#include "utils.h"
#include "flags.h"
using namespace std;

DSLDA::DSLDA(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
             int K, float alpha, float beta)
    : BaseLDA(corpus, to_corpus, th_corpus, K, alpha, beta),
      cdk(corpus.D, K, corpus.T), cvk(corpus.V, K, corpus.T),
      ll_doc(corpus.D), ll_word(corpus.V)
{

}

void DSLDA::Estimate()
{
    std::vector<double> prob(K);
    size_t batch_count = 0;
    double step_size = 0;
    std::vector<int> cd(corpus.D);
    std::vector<int> cv(corpus.V);

    cdk.Fill((double)corpus.T / corpus.D / K);
    cvk.Fill((double)corpus.T / corpus.V / K);

    for (int iter = 0; iter < FLAGS_num_iters; iter++) {
        fill(cd.begin(), cd.end(), 0);
        fill(cv.begin(), cv.end(), 0);

        for (size_t bs = 0; bs < corpus.T; bs += FLAGS_sch_bs) {
            size_t be    = min(bs+FLAGS_sch_bs, corpus.T);

            auto tokens = corpus.Sample(be - bs);
            // E step
            for (auto &token: tokens) {
                double sum = 0;
                for (int k = 0; k < K; k++) {
                    double theta = cdk.Get(token.d, k) + alpha;
                    double phi   = (cvk.Get(token.v, k) + beta) /
                                   (cvk.GetSum(k) + beta_bar);
                    prob[k] = sum += theta * phi;
                }
                double u = u01(generator) * sum;
                int k = 0;
                while (k < K-1 && prob[k]<u) k++;
//                if (iter == 0)
//                    k = generator() % K;
                token.k = k;

                cd[token.d]++;
                cv[token.v]++;
            }

            // M step
            batch_count += 1;
            step_size = GetSchedule(batch_count);
            double ratio     = (double)step_size / (be - bs);
            cdk.Decay(step_size);
            cvk.Decay(step_size);
            for (auto &token: tokens) {
                cdk.Add(token.d, token.k, ratio * token.weight);
                cvk.Add(token.v, token.k, ratio * token.weight);
            }
            cdk.Compress();
            cvk.Compress();
        }
        // Compute perplexity
        vector<double> theta(corpus.D * K);
        vector<double> phi  (corpus.V * K);

        vector<double> new_ll_doc(corpus.D, 0);
        vector<double> new_ll_word(corpus.V, 0);
        for (int d = 0; d < corpus.D; d++) {
            double sum = 0;
            for (int k = 0; k < K; k++) {
                double val = cdk.Get(d, k) + alpha;
                theta[d * K + k] = val;
                sum += val;
            }
            for (int k = 0; k < K; k++)
                theta[d*K+k] /= sum;
        }
        for (int v = 0; v < corpus.V; v++)
            for (int k = 0; k < K; k++)
                phi[v*K+k] = (cvk.Get(v, k) + beta) /
                             (cvk.GetSum(k) + beta_bar);
        for (int k = 0; k < K; k++) {
            double sum = 0;
            for (int v = 0; v < corpus.V; v++)
                sum += phi[v*K+k];
            if (fabs(sum - 1) > 1e-7)
                cout << "Error " << sum << endl;
        }

        double ll = 0;
        for (auto &token: corpus.tokens) {
            double l = 0;
            for (int k = 0; k < K; k++)
                l += theta[token.d*K + k] * phi[token.v*K + k];
            l   = log(l);
            ll += l;
            new_ll_doc[token.d]  += l;
            new_ll_word[token.v] += l;
        }
        std::vector<double> avg_imp_d(corpus.D);
        std::vector<double> avg_imp_v(corpus.V);
        for (int d = 0; d < corpus.D; d++) {
            if (cd[d])
                avg_imp_d[d] = (new_ll_doc[d] - ll_doc[d]) / cd[d];
            else
                avg_imp_d[d] = 0;
        }
        for (int v = 0; v < corpus.V; v++) {
            if (cv[v])
                avg_imp_v[v] = (new_ll_word[v] - ll_word[v]) / cv[v];
            else
                avg_imp_v[v] = 0;
        }

        if (iter) {
            corpus.SetDocDist(avg_imp_d);
            corpus.SetWordDist(avg_imp_v);
        }

        double perplexity = exp(-ll / corpus.T);
        double test_perplexity = Inference(phi);
        cout << "Iteration " << iter
             << " step size = " << step_size
             << " perplexity = " << perplexity
             << " test perplexity = " << test_perplexity << endl;

        ll_doc = new_ll_doc;
        ll_word = new_ll_word;
    }
}
