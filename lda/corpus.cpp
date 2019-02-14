//
// Created by jianfei on 18-1-23.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <random>
#include <algorithm>
#include "corpus.h"
using namespace std;

bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

Corpus::Corpus(const std::string &data_file, const std::string &vocab_filename) {
    ifstream fvocab(vocab_filename.c_str());
    string s;
    while (fvocab >> s)
        vocab.push_back(s);
    V = min((int)vocab.size(), FLAGS_max_vocab);
    vocab.resize(V);

    if (!Load(data_file)) {
        ifstream fdata(data_file.c_str());
        std::string line;
        d.resize(vocab.size());
        T = 0;
        while (getline(fdata, line)) {
            std::vector<int> doc;
            for (auto &c: line)
                if (c == ':')
                    c = ' ';
            int d = (int)w.size();
            int id, k, v;
            istringstream sin(line);
            sin >> id;
            while (sin >> k >> v) {
                if (k < V) {
                    while (v--) {
                        doc.push_back(k);
                        this->d[k].push_back(d);
                    }
                }
            }
            T += doc.size();
            w.push_back(move(doc));
        }
        D = (int)w.size();
        V = (int)vocab.size();
        Save(data_file);
    }

    cout << "Read " << D << " docs, "
         << V << " words, "
         << T << " tokens." << endl;
    PrintSize();
}

void Corpus::PrintSize() {
    for (auto &ww: w) ww.shrink_to_fit();
    for (auto &dd: d) dd.shrink_to_fit();
    double size = 0;
    for (auto &ww: w) size += sizeof(int) * ww.capacity();
    for (auto &dd: d) size += sizeof(int) * dd.capacity();
    cout << "Corpus is " << size / 1024 / 1024 / 1024 << " GB." << endl;
}

void Corpus::SaveArray(const std::string &data_file, std::vector <std::vector<int>> &arr) {
    ofstream fout(data_file, ios::binary);
    std::vector<int> sizes;
    sizes.push_back(arr.size());
    for (auto &a: arr) sizes.push_back(a.size());
    fout.write((char*)sizes.data(), sizes.size()*sizeof(int));

    for (auto &a: arr)
        fout.write((char*)a.data(), a.size()*sizeof(int));
}

void Corpus::LoadArray(const std::string &data_file, std::vector <std::vector<int>> &arr) {
    ifstream fin(data_file, ios::binary);
    std::vector<int> sizes;
    int N;
    fin.read((char*)&N, sizeof(int));
    sizes.resize(N);
    fin.read((char*)sizes.data(), N*sizeof(int));
    arr.resize(N);
    for (int i = 0; i < N; i++)
        arr[i].resize(sizes[i]);
    for (auto &a: arr)
        fin.read((char*)a.data(), a.size()*sizeof(int));
}

bool Corpus::Load(const std::string &data_file) {
    auto w_file = data_file + ".w";
    auto d_file = data_file + ".d";
    if (!is_file_exist(w_file.c_str()))
        return false;

    cout << "Found existing data. Loading..." << endl;
    LoadArray(w_file, w);
    LoadArray(d_file, d);
    D = w.size();
    V = d.size();
    T = 0;
    for (auto &a: w) T += a.size();
    return true;
}

void Corpus::Save(const std::string &data_file) {
    auto w_file = data_file + ".w";
    auto d_file = data_file + ".d";
    SaveArray(w_file, w);
    SaveArray(d_file, d);
}
