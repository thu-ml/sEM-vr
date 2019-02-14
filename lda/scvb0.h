//
// Created by jianfei on 2018/1/25.
//

#ifndef DSLDA_SCVB0_H
#define DSLDA_SCVB0_H

#include "base_lda.h"
#include <vector>
#include "corpus.h"

class Corpus;

class SCVB0 : public BaseLDA {
public:
    SCVB0(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
          int K, float alpha, float beta);

    void GetPosterior(double *theta, double *phi, double *prob);
    double GetPosterior2(double *theta, double *phi, double *prob);

    void Estimate();

    double BatchLogLikelihood();

    double BatchPerplexity();

    void ThetaPhi();

    void VarianceTerm(double step_size, double &var, double &decay);

    void BatchIteration();

    void AdaptiveSchedule(std::vector<double> &dVar, std::vector<double> &vVar);

private:
    std::vector<double> cdk, cvk, ck;
    std::vector<double> next_cdk, next_cvk, next_ck;
    std::vector<double> theta, phi, theta_hist, phi_hist;

    std::vector<int> dFreq, vFreq;

    int batch_size;
    int iter;
};


#endif //DSLDA_SCVB0_H
