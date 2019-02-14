//
// Created by jianfei on 2018/1/24.
//

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <cassert>
#include "utils.h"
#include "flags.h"

std::uniform_real_distribution<double> u01;
thread_local std::mt19937 generator(FLAGS_seed);

double digamma(double x) {
    double result = 0, xx, xx2, xx4;
    assert(x > 0);
    for ( ; x < 7; ++x)
        result -= 1/x;
    x -= 1.0/2.0;
    xx = 1.0/x;
    xx2 = xx*xx;
    xx4 = xx2*xx2;
    result += log(x)+(1./24.)*xx2-(7.0/960.0)*xx4+(31.0/8064.0)*xx4*xx2-(127.0/30720.0)*xx4*xx4;
    return result;
}

double sqr(double x) { return x*x; }
