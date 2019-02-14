#include "stochastic_count.h"
#include <cmath>
using namespace std;

int main() {
    StochasticCount m(4, 3, 1);

    m.Decay(1.0);
    m.Add(1, 1, 1);
    cout << m << endl;

    m.Decay(sqrt(0.5));
    m.Add(2, 1, sqrt(0.5));
    cout << m << endl;

    m.Decay(sqrt(1./3));
    m.Add(3, 2, sqrt(1./3));
    cout << m << endl;
}
