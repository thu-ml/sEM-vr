//
// Created by jianfei on 2018/1/24.
//

#ifndef DSLDA_SVI_H
#define DSLDA_SVI_H

#include <deque>
#include "base_lda.h"
#include "stochastic_count.h"

class Corpus;

class SVI : public BaseLDA {
public:
    SVI(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
         int K, float alpha, float beta);

    void Estimate();

    void Phi(std::vector<double> &cvk);

private:
    std::vector<double> theta, gamma, phi, cvk, next_cvk, smoothed_cvk;
    std::deque<std::vector<double>> ss;

    std::vector<std::vector<int> > v_indices;
    std::vector<double> buffer;
};

#endif //DSLDA_SVI_H
