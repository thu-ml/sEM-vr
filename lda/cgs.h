//
// Created by jianfei on 18-1-23.
//

#ifndef DSLDA_CGS_H
#define DSLDA_CGS_H

#include "base_lda.h"
#include "stochastic_count.h"

class Corpus;

class CGS : public BaseLDA {
public:
    CGS(Corpus &corpus, Corpus &to_corpus, Corpus &th_corpus,
        int K, float alpha, float beta);

    void Estimate();

private:
    std::vector<int> cdk, cvk, ck;
};


#endif //DSLDA_CGS_H
