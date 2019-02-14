//
// Created by jianfei on 2018/1/24.
//

#ifndef DSLDA_CVBOEM_H
#define DSLDA_CVBOEM_H

#include "base_lda.h"
#include "stochastic_count.h"
#include "accumulator.h"

class Corpus;

class CVBOEM : public BaseLDA {
public:
    CVBOEM(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
         int K, float alpha, float beta);

    void Estimate();

    void Phi(std::vector<double> &cvk);

    void RealPhi();

    double BatchLogLikelihood();

    double BatchPerplexity();

    void InferenceDocument(int d, double *th, double *phi, double *doc_buf);

    void UpdateStatistics(int d, double *th, double *phi, double *doc_buf, double Tscale);

private:
    std::vector<int> cdk;
    std::vector<double> theta, phi, cvk, next_cvk, cv_cvk, theta_hist, phi_hist;

    std::vector<std::vector<int> > v_indices;
    std::vector<double> buffer;

    Accumulator<double> cd_buffer, var_buffer, work_buffer;
};

#endif
