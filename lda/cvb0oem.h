//
// Created by jianfei on 2018/1/24.
//

#ifndef DSLDA_CVB0OEM_H
#define DSLDA_CVB0OEM_H

#include "base_lda.h"
#include "stochastic_count.h"

class Corpus;

class CVB0OEM : public BaseLDA {
public:
    CVB0OEM(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
         int K, float alpha, float beta);

    void Estimate();

    void Phi(std::vector<double> &cvk);

    double BatchLogLikelihood();

    double BatchPerplexity();

private:
    std::vector<int> cdk;
    std::vector<double> theta, phi, cvk, next_cvk;

    std::vector<std::vector<int> > v_indices;
    std::vector<double> buffer;
};

#endif //DSLDA_MFVOEM_H
