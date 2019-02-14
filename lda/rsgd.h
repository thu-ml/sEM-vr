//
// Created by jianfei on 2018/1/24.
//
// SGD with softmax reparameterization
#ifndef DSLDA_RSGD_H
#define DSLDA_RSGD_H

#include "base_lda.h"
#include "stochastic_count.h"

class Corpus;

class RSGD : public BaseLDA {
public:
    RSGD(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
         int K, float alpha, float beta);

    void Estimate();

    void Theta();

    void Phi();

private:
    std::vector<double> theta, phi, cdk, cvk, cvk2, rtheta, rphi;
    std::vector<double> theta_hist, phi_hist, cv_cdk, cv_cvk, cv_ck;
    std::vector<std::vector<int> > v_indices;
    std::vector<double> buffer, buffer2;
};

#endif //DSLDA_RSGD_H
