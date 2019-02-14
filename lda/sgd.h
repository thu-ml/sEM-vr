//
// Created by 陈键飞 on 2018/4/17.
//

#ifndef DSLDA_SGD_H
#define DSLDA_SGD_H

#include "base_lda.h"
#include <vector>
#include "corpus.h"

class Corpus;

class SGD : public BaseLDA {
public:
    SGD(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
          int K, float alpha, float beta);

    double InnerProduct(double *th, double *ph);

    void Estimate();

    double BatchLogLikelihood();

    double BatchPerplexity();

    void RealPhi();

private:
    std::vector<double> theta, phi, theta_hist, phi_hist;

    double step_size;
    double D_step_size;

    std::vector<std::vector<int> > v_indices;
    std::vector<double> buffer;

    std::vector<double> step_size_seq;

    int batch_size;
    int iter;
    double C;
};

#endif //DSLDA_SGD_H
