//
// Created by jianfei on 2018/1/24.
//

#ifndef DSLDA_BestSCVB1_H
#define DSLDA_BestSCVB1_H

#include "base_lda.h"
#include "stochastic_count.h"

class Corpus;

class BestSCVB1 : public BaseLDA {
public:
    BestSCVB1(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
         int K, float alpha, float beta);

    void Estimate();

    void Theta(double *cdk, double *theta, int N);

    void Phi(std::vector<double> &cvk);

    double BatchLogLikelihood();

    double BatchPerplexity();

private:
    std::vector<double> cdk;
    std::vector<double> theta, phi, cvk, next_cvk;
};

#endif //DSLDA_BestSCVB1_H
