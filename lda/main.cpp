//
// Created by jianfei on 18-1-23.
//
#include <exception>
#include <stdexcept>
#include "corpus.h"
#include "base_lda.h"
#include "cvsEM.h"
#include "sgd.h"
#include "gd.h"
#include "goem.h"
#include "bestscvb0.h"
#include "bestscvb1.h"
#include "mfvoem.h"
#include "cvboem.h"
#include "cvb0oem.h"
#include "icvboem.h"
#include "ncvboem.h"
#include "rsgd.h"
#include "smd.h"
#include "svi.h"
#include "gflags/gflags.h"
using namespace std;

DEFINE_string(alg, "ds", "cgs or ds or em or goem or scvb0");
DEFINE_string(prefix, "../data/nips", "Prefix of the dataset");
DEFINE_int32(k, 10, "Number of topics");
DEFINE_double(alpha_sum, 50, "Sum of alpha");
DEFINE_double(beta, 0.01, "Beta");

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    string vocab_file = FLAGS_prefix + ".vocab";
    string train_file = FLAGS_prefix + ".libsvm.train";
    string to_file    = FLAGS_prefix + ".libsvm.test";
    string th_file    = FLAGS_prefix + ".libsvm.train";
    Corpus corpus(train_file.c_str(), vocab_file.c_str());
    Corpus to_corpus(to_file.c_str(), vocab_file.c_str());
    Corpus th_corpus(th_file.c_str(), vocab_file.c_str());

    BaseLDA *lda = nullptr;
    if (FLAGS_alg=="cvsem")
        lda = new CVSEM(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else if (FLAGS_alg=="sgd")
        lda = new SGD(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else if (FLAGS_alg=="gd")
        lda = new GD(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else if (FLAGS_alg=="goem")
        lda = new GOEM(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else if (FLAGS_alg=="bestscvb0")
        lda = new BestSCVB0(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else if (FLAGS_alg=="bestscvb1")
        lda = new BestSCVB1(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else if (FLAGS_alg=="mfvoem")
        lda = new MFVOEM(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else if (FLAGS_alg=="cvboem")
        lda = new CVBOEM(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else if (FLAGS_alg=="cvb0oem")
        lda = new CVB0OEM(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else if (FLAGS_alg=="icvboem")
        lda = new ICVBOEM(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else if (FLAGS_alg=="ncvboem")
        lda = new NCVBOEM(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else if (FLAGS_alg=="rsgd")
        lda = new RSGD(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else if (FLAGS_alg=="smd")
        lda = new SMD(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else if (FLAGS_alg=="svi")
        lda = new SVI(corpus, to_corpus, th_corpus, FLAGS_k, FLAGS_alpha_sum / FLAGS_k, FLAGS_beta);
    else
        throw runtime_error("Unknown algorithm");

    lda->Estimate();
}
