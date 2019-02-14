//
// Created by jianfei on 2018/1/24.
//

#ifndef DSLDA_GOEM_H
#define DSLDA_GOEM_H

#include "base_lda.h"
#include "stochastic_count.h"

class Corpus;

class GOEM : public BaseLDA {
public:
    GOEM(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
         int K, float alpha, float beta);

    void Estimate();

    void Phi(std::vector<double> &cvk);

private:
    std::vector<std::vector<int> > z;
    std::vector<int> cdk;
    std::vector<double> theta, phi, cvk, next_cvk;
};

#endif //DSLDA_GOEM_H
