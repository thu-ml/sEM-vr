//
// Created by jianfei on 2018/1/26.
//

#ifndef DSLDA_BASE_LDA_H
#define DSLDA_BASE_LDA_H

#include <vector>
#include "clock.h"

class Corpus;

class BaseLDA {
public:
    BaseLDA(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
            int K, float alpha, float beta);

    virtual void Estimate() = 0;

    double GetSchedule(int t);

    double CVB0Inference(const std::vector<double> &phi, Corpus &to_corpus);

    double LeftToRightInference(const std::vector<double> &phi, Corpus &to_corpus);

    double Inference(const std::vector<double> &phi);
    double Inference(const std::vector<double> &phi, Corpus &to_corpus);

    double CVBELBO(const std::vector<double> &phi);

    void PositiveProjection(std::vector<double> &cvk);

    void RenormalizeProjection(std::vector<double> &cvk, std::vector<double> &phi);

    void Eval(int iter, int subiter, std::vector<double> &theta, std::vector<double> &phi);

    void Start();

    double BatchLogLikelihood(std::vector<double> &theta, std::vector<double> &phi);

protected:
    Corpus &corpus, &to_corpus, &th_corpus;
    int K;
    float alpha, alpha_bar;
    float beta, beta_bar;
    Clock clock;
    double running_time;
    double step_size;
};

#endif //DSLDA_BASE_LDA_H
