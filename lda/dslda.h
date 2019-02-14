//
// Created by jianfei on 18-1-23.
//

#ifndef DSLDA_DSLDA_H
#define DSLDA_DSLDA_H

#include "base_lda.h"
#include "stochastic_count.h"

class Corpus;

class DSLDA : public BaseLDA {
public:
    DSLDA(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
          int K, float alpha, float beta);

    void Estimate();

private:
    StochasticCount cdk, cvk;
    std::vector<double> ll_doc, ll_word;
};


#endif //DSLDA_DSLDA_H
