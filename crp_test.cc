#include <iostream>
#include <vector>
#include <cassert>
#include <string>

#include "cpyp/crp.h"
#include "cpyp/random.h"

using namespace std;

int main() {
  cpyp::MT19937 eng;
  double tot = 0;
  double xt = 0;
  cpyp::crp<int> crp(0.5, 1.0);
  unsigned cust = 10;
  vector<int> hist(cust + 1, 0);
  for (unsigned i = 0; i < cust; ++i) { crp.increment(1, 1.0, eng); }
  const int samples = 200000;
  const bool simulate = true;
  for (int k = 0; k < samples; ++k) {
  if (!simulate) {
    crp.clear();
    for (unsigned i = 0; i < cust; ++i) { crp.increment(1, 1.0, eng); }
  } else {
    unsigned da = cpyp::sample_uniform01<double>(eng) * cust;
    bool a = cpyp::sample_uniform01<double>(eng) < 0.5;
    if (a) {
      for (unsigned i = 0; i < da; ++i) { crp.increment(1, 1.0, eng); }
      for (unsigned i = 0; i < da; ++i) { crp.decrement(1, eng); }
      xt += 1.0;
    } else {
      for (unsigned i = 0; i < da; ++i) { crp.decrement(1, eng); }
      for (unsigned i = 0; i < da; ++i) { crp.increment(1, 1.0, eng); }
    }
  }
  int c = crp.num_tables(1);
  ++hist[c];
  tot += c;
  }
  assert(cust == crp.num_customers());
  cerr << "P(a) = " << (xt / samples) << endl;
  cerr << "mean num tables = " << (tot / samples) << endl;
  double error = fabs((tot / samples) - 5.4);
  cerr << "  error = " << error << endl;
  for (unsigned i = 1; i <= cust; ++i)
  cerr << i << ' ' << (hist[i]) << endl;
  if (error > 0.1) {
  cerr << "*** error is too big = " << error << endl;
  return 1;
  }
  return 0;
}

