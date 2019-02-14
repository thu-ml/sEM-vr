//
// Created by jianfei on 18-1-23.
//

#ifndef DSLDA_CORPUS_H
#define DSLDA_CORPUS_H

#include <string>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <exception>
#include "utils.h"
#include "flags.h"

struct Token {
    Token() {}
    Token(int d, int v, int k, float weight=0) : d(d), v(v), k(k), weight(weight) {}
    int d, v, k;
    float weight;
};

class Corpus {
public:
    Corpus(const std::string &data_file, const std::string &vocab_filename);

    void PrintSize();

    std::vector<std::vector<int> > w, d;
    std::vector<std::string> vocab;

    void SaveArray(const std::string &data_file, std::vector<std::vector<int> > &arr);
    void LoadArray(const std::string &data_file, std::vector<std::vector<int> > &arr);
    void Save(const std::string &data_file);
    bool Load(const std::string &data_file);

    int D, V;
    size_t T;
};


#endif //DSLDA_CORPUS_H
