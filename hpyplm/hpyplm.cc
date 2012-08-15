#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "corpus/corpus.h"
#include "cpyp/m.h"
#include "cpyp/random.h"
#include "cpyp/crp.h"
#include "cpyp/tied_parameter_resampler.h"

// A not very memory-efficient implementation of an N-gram LM based on PYPs
// as described in Y.-W. Teh. (2006) A Hierarchical Bayesian Language Model
// based on Pitman-Yor Processes. In Proc. ACL.

// I use templates to handle the recursive formalation of the prior, so
// the order of the model has to be specified here, at compile time:
#define kORDER 3

using namespace std;
using namespace cpyp;

Dict dict;

struct uvector_hash {
  size_t operator()(const vector<unsigned>& v) const {
    size_t h = v.size();
    for (auto e : v)
      h ^= e + 0x9e3779b9 + (h<<6) + (h>>2);
    return h;
  }
};

// uniform distribution over a fixed vocabulary
struct UniformVocabulary {
  UniformVocabulary(unsigned vs, double, double, double, double) : p0(1.0 / vs), draws() {}
  template<typename Engine>
  void increment(unsigned, const vector<unsigned>&, Engine&) { ++draws; }
  template<typename Engine>
  void decrement(unsigned, const vector<unsigned>&, Engine&) { --draws; assert(draws >= 0); }
  double prob(unsigned, const vector<unsigned>&) const { return p0; }
  template<typename Engine>
  void resample_hyperparameters(Engine&) {}
  double log_likelihood() const { return draws * log(p0); }
  const double p0;
  int draws;
};

template <unsigned N> struct PYPLM;

template<> struct PYPLM<0> : public UniformVocabulary {
  PYPLM(unsigned vs, double a, double b, double c, double d) :
    UniformVocabulary(vs, a, b, c, d) {}
};

// represents an N-gram LM
template <unsigned N> struct PYPLM {
  PYPLM(unsigned vs, double da, double db, double ss, double sr) :
      backoff(vs, da, db, ss, sr),
      tr(da, db, ss, sr, 0.8, 1.0),
      lookup(N-1) {}
  template<typename Engine>
  void increment(unsigned w, const vector<unsigned>& context, Engine& eng) {
    const double bo = backoff.prob(w, context);
    for (unsigned i = 0; i < N-1; ++i)
      lookup[i] = context[context.size() - 1 - i];
    auto it = p.find(lookup);
    if (it == p.end()) {
      it = p.insert(make_pair(lookup, crp<unsigned>(0.8,1))).first;
      tr.insert(&it->second);  // add to resampler
    }
    if (it->second.increment(w, bo, eng))
      backoff.increment(w, context, eng);
  }
  template<typename Engine>
  void decrement(unsigned w, const vector<unsigned>& context, Engine& eng) {
    for (unsigned i = 0; i < N-1; ++i)
      lookup[i] = context[context.size() - 1 - i];
    auto it = p.find(lookup);
    assert(it != p.end());
    if (it->second.decrement(w, eng))
      backoff.decrement(w, context, eng);
  }
  double prob(unsigned w, const vector<unsigned>& context) const {
    const double bo = backoff.prob(w, context);
    for (unsigned i = 0; i < N-1; ++i)
      lookup[i] = context[context.size() - 1 - i];
    auto it = p.find(lookup);
    if (it == p.end()) return bo;
    return it->second.prob(w, bo);
  }

  double log_likelihood() const {
    double llh = backoff.log_likelihood();
    for (auto crp : p)
      llh += crp.second.log_likelihood();
    llh += tr.log_likelihood();
    return llh;
  }

  template<typename Engine>
  void resample_hyperparameters(Engine& eng) {
    tr.resample_hyperparameters(eng);
    backoff.resample_hyperparameters(eng);
  }

  PYPLM<N-1> backoff;
  tied_parameter_resampler<crp<unsigned>> tr;
  double discount_a, discount_b, strength_s, strength_r;
  double d, strength;
  mutable vector<unsigned> lookup;  // thread-local
  unordered_map<vector<unsigned>, crp<unsigned>, uvector_hash> p;  // .first = context .second = CRP
};

int main(int argc, char** argv) {
  if (argc != 4) {
    cerr << argv[0] << " <training.txt> <test.txt> <nsamples>\n\nEstimate a " << kORDER << "-gram HPYP LM and report perplexity\n100 is usually sufficient for <nsamples>\n";
    return 1;
  }
  MT19937 eng;
  string train_file = argv[1];
  string test_file = argv[2];
  int samples = atoi(argv[3]);
  
  vector<vector<unsigned> > corpuse;
  set<unsigned> vocabe, tv;
  const unsigned kSOS = dict.Convert("<s>");
  const unsigned kEOS = dict.Convert("</s>");
  cerr << "Reading corpus...\n";
  ReadFromFile(train_file, &dict, &corpuse, &vocabe);
  cerr << "E-corpus size: " << corpuse.size() << " sentences\t (" << vocabe.size() << " word types)\n";
  vector<vector<unsigned> > test;
  if (test_file.size()) ReadFromFile(test_file, &dict, &test, &tv); else test = corpuse;
  PYPLM<kORDER> lm(vocabe.size(), 1, 1, 1, 1);
  vector<unsigned> ctx(kORDER - 1, kSOS);
  for (int sample=0; sample < samples; ++sample) {
    for (const auto& s : corpuse) {
      ctx.resize(kORDER - 1);
      for (unsigned i = 0; i <= s.size(); ++i) {
        unsigned w = (i < s.size() ? s[i] : kEOS);
        if (sample > 0) lm.decrement(w, ctx, eng);
        lm.increment(w, ctx, eng);
        ctx.push_back(w);
      }
    }
    if (sample % 10 == 9) {
      cerr << " [LLH=" << lm.log_likelihood() << "]" << endl;
      if (sample % 30u == 29) lm.resample_hyperparameters(eng);
    } else { cerr << '.' << flush; }
  }
  double llh = 0;
  unsigned cnt = 0;
  unsigned oovs = 0;
  for (auto& s : test) {
    ctx.resize(kORDER - 1);
    for (unsigned i = 0; i <= s.size(); ++i) {
      unsigned w = (i < s.size() ? s[i] : kEOS);
      double lp = log(lm.prob(w, ctx)) / log(2);
      if (i < s.size() && vocabe.count(w) == 0) {
        cerr << "**OOV ";
        ++oovs;
        lp = 0;
      }
      cerr << "p(" << dict.Convert(w) << " |";
      for (unsigned j = ctx.size() + 1 - kORDER; j < ctx.size(); ++j)
        cerr << ' ' << dict.Convert(ctx[j]);
      cerr << ") = " << lp << endl;
      ctx.push_back(w);
      llh -= lp;
      cnt++;
    }
  }
  cerr << "  Log_10 prob: " << (-llh * log(2) / log(10)) << endl;
  cerr << "        Count: " << cnt << endl;
  cerr << "         OOVs: " << oovs << endl;
  cerr << "Cross-entropy: " << (llh / cnt) << endl;
  cerr << "   Perplexity: " << pow(2, llh / cnt) << endl;
  return 0;
}

