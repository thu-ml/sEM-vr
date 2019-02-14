//
// Created by jianfei on 2018/1/25.
//

#ifndef DSLDA_EM_H
#define DSLDA_EM_H

#include "base_lda.h"
#include <vector>
#include <string>

class Corpus;

class EM : public BaseLDA{
public:
    EM(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
       int K, float alpha, float beta);

    void Estimate();

    void UpdateParams();

    void WriteFile(std::string prefix, int iter, std::vector<double> &a, int R, int C);

    double LogLikelihood();

    double LogLC();

private:
    std::vector<double> cdk, cvk, ck;
    std::vector<double> theta, phi;
    std::vector<double> delta_theta, delta_phi;
    std::vector<double> next_cdk, next_cvk, next_ck;
};

#endif //DSLDA_EM_H
