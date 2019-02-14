//
// Created by jianfei on 2018/1/24.
//
// SGD with softmax reparameterization
#ifndef DSLDA_SMD_H
#define DSLDA_SMD_H

#include "base_lda.h"
#include "stochastic_count.h"

class Corpus;

class SMD : public BaseLDA {
public:
    SMD(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
         int K, float alpha, float beta);

    void Estimate();

private:
    std::vector<double> theta, phi, cvk, cvk2, theta_hist, phi_hist, cv_cdk, cv_cvk;
    std::vector<std::vector<int> > v_indices;
    std::vector<double> buffer, buffer2;
};

#endif //DSLDA_SMD_H
