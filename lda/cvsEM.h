//
// Created by 陈键飞 on 2018/4/17.
//

#ifndef DSLDA_CVSEM_H
#define DSLDA_CVSEM_H

#include "base_lda.h"
#include <vector>
#include "corpus.h"

class Corpus;

class CVSEM : public BaseLDA {
public:
    CVSEM(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
          int K, float alpha, float beta);

    void GetPosterior(double *theta, double *phi, double *prob);

    void Estimate();

    void Theta(double *cdk, double *theta, int N);

    void Phi(std::vector<double> &cvk);

    void RealPhi();

    void BatchIteration(std::vector<int> &docs,
                        std::vector<double> &theta, std::vector<double> &phi,
                        std::vector<double> &cdk, std::vector<double> &cvk);

private:
    std::vector<double> cdk, cvk, cv_cdk, cv_cvk;
    std::vector<double> next_cvk;
    std::vector<double> theta, phi, theta_hist, phi_hist;

    std::vector<double> ma1;
    double D_step_size;

    std::vector<std::vector<int> > v_indices;
    std::vector<double> buffer;

    std::vector<double> step_size_seq;

    int batch_size;
    int iter;
    double C;
};

#endif //DSLDA_CVSEM_H
