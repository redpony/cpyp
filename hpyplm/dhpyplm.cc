#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "corpus/corpus.h"
#include "cpyp/m.h"
#include "cpyp/random.h"
#include "cpyp/crp.h"
#include "cpyp/mf_crp.h"
#include "cpyp/tied_parameter_resampler.h"

// A not very memory-efficient implementation of a domain adapting
// HPYP language model, as described by Wood & Teh (AISTATS, 2009)
//
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
    return backoff.log_likelihood() + tr.log_likelihood();
  }

  template<typename Engine>
  void resample_hyperparameters(Engine& eng) {
    tr.resample_hyperparameters(eng);
    backoff.resample_hyperparameters(eng);
  }

  PYPLM<N-1> backoff;
  tied_parameter_resampler<crp<unsigned>> tr;
  mutable vector<unsigned> lookup;  // thread-local
  unordered_map<vector<unsigned>, crp<unsigned>, uvector_hash> p;  // .first = context .second = CRP
};

// represents an N-gram domain adapted LM
template <unsigned N> struct DAPYPLM;

// zero-gram model
template<> struct DAPYPLM<0> : PYPLM<0> {
  DAPYPLM(PYPLM<0>& rllm) : PYPLM(rllm) {}
};

template <unsigned N> struct DAPYPLM {
  DAPYPLM(PYPLM<N>& rllm) : path(1,1,1,1,0.1,1.0), tr(1,1,1,1), in_domain_backoff(rllm.backoff), llm(rllm), lookup(N-1) {}
  template<typename Engine>
  void increment(unsigned w, const vector<unsigned>& context, Engine& eng) {
    const double p0[2]{in_domain_backoff.prob(w, context), llm.prob(w, context)};
    double b = path.prob(0, 0.5);
    const double lam[2]{b, 1.0 - b};
    for (unsigned i = 0; i < N-1; ++i)
      lookup[i] = context[context.size() - 1 - i];
    auto it = p.find(lookup);
    if (it == p.end()) {
      it = p.insert(make_pair(lookup, mf_crp<2, unsigned>(0.8,1))).first;
      tr.insert(&it->second);  // add to resampler
    }
    const pair<unsigned, int> floor_count = it->second.increment(w, p0, lam, eng);
    if (floor_count.second) {
      if (floor_count.first == 0) { // in-domain backoff
        //cerr << "Increment<" << N << "> in domain\n";
        path.increment(0, 0.5, eng);
        in_domain_backoff.increment(w, context, eng);
      } else { // domain general backoff
        //cerr << "Increment<" << N << "> out of domain\n";
        path.increment(1, 0.5, eng);
        llm.increment(w, context, eng);
      }
    }
  }

  template<typename Engine>
  void decrement(unsigned w, const vector<unsigned>& context, Engine& eng) {
    for (unsigned i = 0; i < N-1; ++i)
      lookup[i] = context[context.size() - 1 - i];
    auto it = p.find(lookup);
    assert(it != p.end());
    const pair<unsigned, int> floor_count = it->second.decrement(w, eng);
    //cerr << "Dec: floor=" << floor_count.first << endl;
    if (floor_count.second) {
      if (floor_count.first == 0) { // in-domain backoff
        //cerr << "Decrement<" << N << "> in domain\n";
        path.decrement(0, eng);
        in_domain_backoff.decrement(w, context, eng);
      } else { // domain general backoff
        //cerr << "Decrement<" << N << "> out of domain\n";
        path.decrement(1, eng);
        llm.decrement(w, context, eng);
      }
    }
  }

  double prob(unsigned w, const vector<unsigned>& context) const {
    const double p0[2]{in_domain_backoff.prob(w, context), llm.prob(w, context)};
    double b = path.prob(0, 0.5);
    const double lam[2]{b, 1.0 - b};
    for (unsigned i = 0; i < N-1; ++i)
      lookup[i] = context[context.size() - 1 - i];
    auto it = p.find(lookup);
    if (it == p.end()) return lam[0] * p0[0] + lam[1] * p0[1];
    return it->second.prob(w, p0, lam);
  }

  double log_likelihood() const {
    return path.log_likelihood() + path.num_customers() * log(0.5) + tr.log_likelihood();
  }

  template<typename Engine>
  void resample_hyperparameters(Engine& eng) {
    path.resample_hyperparameters(eng);
    cerr << "Path<" << N << "> d=" << path.discount() << ",s=" << path.strength() << " p(in_domain) = " << path.prob(0, 0.5) << endl;
    tr.resample_hyperparameters(eng);
    in_domain_backoff.resample_hyperparameters(eng);
  }

  crp<unsigned> path;
  tied_parameter_resampler<mf_crp<2, unsigned>> tr;
  DAPYPLM<N-1> in_domain_backoff;
  PYPLM<N>& llm;
  mutable vector<unsigned> lookup;  // thread-local
  unordered_map<vector<unsigned>, mf_crp<2, unsigned>, uvector_hash> p;  // .first = context .second = 2-floor CRP
};

int main(int argc, char** argv) {
  if (argc < 4) {
    cerr << argv[0] << " <training1.txt> <training2.txt> [...] <test.txt> <nsamples>\n\nEstimate a " << kORDER << "-gram HPYP LM and report perplexity\n100 is usually sufficient for <nsamples>\n";
    return 1;
  }
  MT19937 eng;
  vector<string> train_files;
  for (int i = 1; i < argc - 2; ++i)
    train_files.push_back(argv[i]);
  string test_file = argv[argc - 2];
  int samples = atoi(argv[argc - 1]);
  int d = 1;
  for (auto& tf : train_files)
    cerr << (d++==1 ? "*" : "") << "Corpus "<< ": " << tf << endl;
  set<unsigned> vocab;
  const unsigned kSOS = dict.Convert("<s>");
  const unsigned kEOS = dict.Convert("</s>");
  vector<vector<unsigned> > test;
  ReadFromFile(test_file, &dict, &test, &vocab);
  vector<vector<vector<unsigned> > > corpora(train_files.size());
  d = 0;
  for (const auto& train_file : train_files)
    ReadFromFile(train_file, &dict, &corpora[d++], &vocab);

  vector<vector<unsigned>> corpus = corpora[0];
  cerr << "E-corpus size: " << corpus.size() << " sentences\t (" << vocab.size() << " word types)\n";
  PYPLM<kORDER> lm(vocab.size(), 1, 1, 1, 1);
  DAPYPLM<kORDER> dalm0(lm);
  DAPYPLM<kORDER> dalm1(lm);
  vector<unsigned> ctx(kORDER - 1, kSOS);
  for (int sample=0; sample < samples; ++sample) {
    int ci = 0;
    for (const auto& corpus : corpora) {
      DAPYPLM<kORDER>& lm = (ci == 0 ? dalm0 : dalm1);
      ++ci;
      for (const auto& s : corpus) {
        ctx.resize(kORDER - 1);
        for (unsigned i = 0; i <= s.size(); ++i) {
          unsigned w = (i < s.size() ? s[i] : kEOS);
          if (sample > 0) lm.decrement(w, ctx, eng);
          lm.increment(w, ctx, eng);
          ctx.push_back(w);
        }
      }
    }
    if (sample % 10 == 9) {
      cerr << " [LLH=" << (lm.log_likelihood() + dalm0.log_likelihood() + dalm1.log_likelihood()) << "]" << endl;
      if (sample % 30u == 29) {
        dalm0.resample_hyperparameters(eng);
        dalm1.resample_hyperparameters(eng);
        lm.resample_hyperparameters(eng);
      }
    } else { cerr << '.' << flush; }
  }
  double llh = 0;
  unsigned cnt = 0;
  unsigned oovs = 0;
  for (auto& s : test) {
    ctx.resize(kORDER - 1);
    for (unsigned i = 0; i <= s.size(); ++i) {
      unsigned w = (i < s.size() ? s[i] : kEOS);
      double lp = log(dalm0.prob(w, ctx)) / log(2);
      if (i < s.size() && vocab.count(w) == 0) {
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

