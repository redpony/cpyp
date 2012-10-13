#ifndef _UNIFORM_VOCAB_H_
#define _UNIFORM_VOCAB_H_

#include <cassert>
#include <vector>

namespace cpyp {

// uniform distribution over a fixed vocabulary
struct UniformVocabulary {
  UniformVocabulary(unsigned vs, double, double, double, double) : p0(1.0 / vs), draws() {}
  template<typename Engine>
  void increment(unsigned, const std::vector<unsigned>&, Engine&) { ++draws; }
  template<typename Engine>
  void decrement(unsigned, const std::vector<unsigned>&, Engine&) { --draws; assert(draws >= 0); }
  double prob(unsigned, const std::vector<unsigned>&) const { return p0; }
  template<typename Engine>
  void resample_hyperparameters(Engine&) {}
  double log_likelihood() const { return draws * log(p0); }
  const double p0;
  int draws;
};

}

#endif
