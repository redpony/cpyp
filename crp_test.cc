#include <iostream>
#include <vector>
#include <cassert>
#include <string>

#include "cpyp/crp.h"
#include "cpyp/random.h"

using namespace std;

static const double p0[] = { 0.1, 0.2, 0.7 };  // actual base distribution
static const double q0[] = { 0.5, 0.4, 0.1 };  // proposal base distribution

// log likelihood of a CRP including draws from a static base distribution p0
double llh(const cpyp::crp<int>& crp, const double* pp0) {
  double l = crp.log_likelihood();
  for (int i = 0; i < 3; ++i)
    l += crp.num_tables(i) * log(pp0[i]);
  return l;
}

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

// same model as test_mh1, same proposal, different way of computing
// p's and q's
void test_mh1a() {
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
      double lq_old = 0;
      double lp_old = llh(crp, p0);
      if (s > 0) crp.decrement(y_i, eng, &lq_old);

      double lq_new = 0;
      crp.increment_no_base(y_i, eng, &lq_new);
      if (s > 0) {
        double lp_new = llh(crp, p0);
        double a = exp(lp_new - lp_old + lq_old - lq_new);
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

static const double p0_a[] = { 0.1, 0.2, 0.7 };  // actual base distribution
static const double p0_b[] = { 0.6, 0.3, 0.1 };  // actual base of d2

static const double q0_a[] = { 0.05, 0.1, 0.35 };  // estimated base distribution
static const double q0_b[] = { 0.3, 0.15, 0.05 };  // estimated base of d2

void test_mh2() {
  cpyp::MT19937 eng;
  cpyp::crp<int> a(0.5, 1.0);
  cpyp::crp<int> b(0.5, 1.0);
  vector<int> hist_a(16, 0);
  vector<int> hist_b(16, 0);
  vector<double> ref_a = {3.70004e-06, 0.0144611, 0.0614236, 0.138702, 0.211308, 0.231733, 0.183865, 0.103685, 0.0409419, 0.0114253, 0.00214592, 0.000281003, 2.33002e-05, 7.00007e-07, 2.12303e-07, 0};
  vector<double> ref_b = {2.90003e-06, 0.00876079, 0.0471061, 0.115841, 0.187615, 0.221488, 0.196102, 0.130352, 0.063905, 0.0224984, 0.00542085, 0.000828908, 7.53008e-05, 4.90005e-06, 1.00001e-07, 0};
  double ref_t0b = 1.53498;

  vector<bool> z(15);
  double c = 0;
  double ac = 0;
  double tmh = 0;
  double zz = 0;
  double t0b = 0;
  for (int s = 0; s < 200000; ++s) {
    for (int i = 0; i < 15; ++i) {
      const unsigned int y_i = i % 3;
      const bool old_z = z[i];
      cpyp::crp<int> old_a = a;
      cpyp::crp<int> old_b = b;
      double lp_old = llh(a, p0_a) + llh(b, p0_b);

      double lq_old = 0;
      double lq_new = 0;
      if (s > 0) (z[i] ? b : a).decrement(y_i, eng, &lq_old);

      double aa = 0.5;  // these can be better estimates
      double bb = 0.5;
      z[i] = cpyp::sample_bernoulli(aa, bb, eng);
      (z[i] ? b : a).increment_no_base(y_i, eng, &lq_new);
      lq_new += log((z[i] ? bb : aa) / (aa + bb));
      lq_old += log((old_z ? bb : aa) / (aa + bb));

      if (s > 0) { 
        double lp_new = llh(a, p0_a) + llh(b, p0_b);
        double acc = exp(lp_new - lp_old + lq_old - lq_new);
        ++tmh;
        if (acc >= 1.0 || cpyp::sample_uniform01<double>(eng) < acc) { // mh accept
          ++ac;
        } else { // mh reject
          std::swap(a, old_a);
          std::swap(b, old_b);
          z[i] = old_z;
        }
      }
    }
    // record sample
    if (s> 200 && s % 10 == 9) {
      assert(a.num_tables() < hist_a.size());
      assert(b.num_tables() < hist_b.size());
      hist_a[a.num_tables()]++; hist_b[b.num_tables()]++; ++c;
      for (int i = 0; i < 15; ++i) {
        unsigned int y_i = i % 3;
        if (!z[i]) { t0b += y_i; ++zz; }
      }
    }
  }
  ac /= tmh;
  cerr << "ACCEPTANCE: " << ac << endl;
  int j =0;
  double te = 0;
  double me = 0;
  for (int i = 0; i < 15; ++i) {
    double a = hist_a[i];
    double b = hist_b[i];
    double err_a = (a / c - ref_a[j]);
    double err_b = (b / c - ref_b[j++]);
    cerr << err_a << "\t" << err_b << endl;
    te += fabs(err_a) + fabs(err_b);
    if (fabs(err_a) > me) { me = fabs(err_a); }
    if (fabs(err_b) > me) { me = fabs(err_b); }
  }
  te /= 30;
  cerr << "t0b = " << (t0b / zz) << endl;
  double ee = fabs((t0b / zz) - ref_t0b);
  cerr << "err t0b = " << ee;
  if (ee > 0.01) { cerr << "  ** TOO HIGH **"; }
  cerr << endl;
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
  test_mh1a();
  test_mh2();
  return 0;
}

