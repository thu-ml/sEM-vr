#ifndef __STOCHASTIC_COUNT
#define __STOCHASTIC_COUNT

#include <vector>
#include <iostream>

class StochasticCount {
public:
    StochasticCount(int R, int C, size_t T);

    void Decay(double step_size);

    void Add(int r, int c, double ratio);

    double Get(int r, int c) const;

    double GetSum(int c) const;

    void Compress();

    void Fill(double v);

//private:
    int R, C;
    size_t T;
    std::vector<double> data;
    std::vector<double> sum;
    double s;
};

std::ostream& operator << (std::ostream &out, const StochasticCount &m);

#endif
