//
// Created by jianfei on 2018/1/24.
//

#ifndef DSLDA_NCVBOEM_H
#define DSLDA_NCVBOEM_H

#include "base_lda.h"
#include "stochastic_count.h"
#include "accumulator.h"

class Corpus;

class NCVBOEM : public BaseLDA {
public:
    NCVBOEM(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
         int K, float alpha, float beta);

    void Estimate();

    void Phi(std::vector<double> &cvk);

    void RealPhi();

    double BatchLogLikelihood();

    double BatchPerplexity();

    void InferenceDocument(int d, double *th, double *phi, double *doc_buf, double *var_in, double *var_out);

private:
    std::vector<int> cdk;
    std::vector<double> theta, var, var_hist, phi, cvk, next_cvk, cv_cvk, phi_hist;

    std::vector<std::vector<int> > v_indices;
    std::vector<double> buffer, buffer_hist;

    Accumulator<double> var_buffer, work_buffer, next_cd_buffer, next_var_buffer;
};

#endif
