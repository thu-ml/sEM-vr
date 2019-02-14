//
// Created by jianfei on 2018/1/24.
//

#ifndef DSLDA_MFVOEM_H
#define DSLDA_MFVOEM_H

#include "base_lda.h"
#include "stochastic_count.h"

class Corpus;

class MFVOEM : public BaseLDA {
public:
    MFVOEM(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
         int K, float alpha, float beta);

    void Estimate();

    void Phi(std::vector<double> &cvk);

    double BatchLogLikelihood();

    double BatchPerplexity();

private:
    std::vector<int> cdk;
    std::vector<double> theta, gamma, phi, cvk, next_cvk;

    std::vector<std::vector<int> > v_indices;
    std::vector<double> buffer;
};

#endif //DSLDA_MFVOEM_H
