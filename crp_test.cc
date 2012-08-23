#include <iostream>
#include <vector>
#include <cassert>
#include <string>

#include "cpyp/crp.h"
#include "cpyp/random.h"

using namespace std;

static const double p0[] = { 0.1, 0.2, 0.7 };  // actual base distribution
static const double q0[] = { 0.8, 0.1, 0.1 };  // proposal base distribution

void test_mh1() {
  cpyp::MT19937 eng;
  const vector<double> ref = {0, 0, 0, 0.00466121, 0.0233846, 0.0647365, 0.125693, 0.183448, 0.204806, 0.177036, 0.119629, 0.0627523, 0.02507, 0.00725451, 0.0013911};
  cpyp::crp<int> crp(0.5, 1.0);
  vector<int> hist(15, 0);
  double c = 0;
  double ac = 0;
  double tmh = 0;
  for (int s = 0; s < 200000; ++s) {
    for (int i = 0; i < 15; ++i) {
      int y_i = i % 3;
      cpyp::crp<int> prev = crp;  // save old state
      bool wasrem = false;
      if (s > 0)
        wasrem = crp.decrement(y_i, eng);

      bool wasnew = crp.increment(y_i, q0[y_i], eng);
      if (s > 0) {
        double p_new = 1, q_new = 1;
        if (wasnew) { p_new = p0[y_i]; q_new = q0[y_i]; }
        double p_old = 1, q_old = 1;
        if (wasrem) { p_old = p0[y_i]; q_old = q0[y_i]; }
        double a = p_new / p_old * q_old / q_new;
        ++tmh;
        if (a >= 1.0 || cpyp::sample_uniform01<double>(eng) < a) { // mh accept
          ++ac;
        } else { // mh reject
          std::swap(crp, prev);  // swap is faster than =
        }
      }
    }
    if (s > 300 && s % 4 == 3) { ++c; hist[crp.num_tables()]++; }
  }
  ac /= tmh;
  cerr << "ACCEPTANCE: " << ac << endl;
  int j =0;
  double te = 0;
  double me = 0;
  for (auto i : hist) {
    double err = (i / c - ref[j++]);
    cerr << err << "\t" << i/c << endl;
    te += fabs(err);
    if (fabs(err) > me) { me = fabs(err); }
  }
  te /= 12;
  cerr << "Average error: " << te;
  if (te > 0.01) { cerr << "  ** TOO HIGH **"; }
  cerr << endl << "    Max error: " << me;
  if (me > 0.01) { cerr << "  ** TOO HIGH **"; }
  cerr << endl;
}

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
  cerr << crp << endl;
  cerr << crp.log_likelihood() << endl;
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
  test_mh1();
  return 0;
}

