#include <iostream>
#include <unordered_map>
#include <cstdlib>

#include "corpus/corpus.h"
#include "cpyp/m.h"
#include "cpyp/random.h"
#include "cpyp/crp.h"
#include "cpyp/tied_parameter_resampler.h"

using namespace std;
using namespace cpyp;

Dict dict;

double log_likelihood(const crp<short>& dt,
                      double ut,
                      const vector<crp<unsigned>>& tt,
                      double uw) {
  double llh = dt.num_tables() * log(ut);
  for (auto& crp : tt) llh += crp.log_likelihood() + crp.num_tables() * log(uw);
  return llh;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    cerr << argv[0] << " <training.txt> <nclasses> <nsamples>\n\nEstimate an naive Bayes model with PY priors.\nInput format: each line in <training.txt> is a document\n";
    return 1;
  }
  cerr << "[WARNING] Gibbs sampling implementation may be somewhat incorrect. Use pynb-mh if this matters.\n";
  MT19937 eng;
  string train_file = argv[1];
  const unsigned labels = atoi(argv[2]);
  const unsigned samples = atoi(argv[3]);
  
  vector<vector<unsigned> > corpus;
  set<unsigned> vocab;
  ReadFromFile(train_file, &dict, &corpus, &vocab);
  cerr << "Corpus size: " << corpus.size() << " documents\t (" << vocab.size() << " word types)\n";
  const double uniform_label = 1.0 / labels;
  const double uniform_word = 1.0 / vocab.size();
  vector<short> z(corpus.size());  // label indicators
  vector<crp<unsigned>> label_term(labels, crp<unsigned>(1,1,1,1));
  crp<short> label(1,1,1,1); // label.prob(k, ...) = conditional prior probability of label
  vector<double> probs(labels);
  for (unsigned sample=0; sample < samples; ++sample) {
    for (unsigned i = 0; i < corpus.size(); ++i) {
      const auto& doc = corpus[i];
      if (sample > 0) {
        label.decrement(z[i], eng);
        for (auto& w : doc) label_term[z[i]].decrement(w, eng);
      }

      // compute posteriors z_i = k
      for (unsigned k = 0; k < labels; ++k) {
        probs[k] = log(label.prob(k, uniform_label));
        for (auto& w : doc) {
          probs[k] += log(label_term[k].prob(w, uniform_word));
          label_term[k].increment(w, uniform_word, eng);
        }
        for (auto& w : doc) label_term[k].decrement(w, eng);
        probs[k] = exp(probs[k]); assert(probs[k] > 0);
      }
      multinomial_distribution<double> mult(probs);
      unsigned k = z[i] = mult(eng);  // resample z_i
      if (sample == 0) { z[i] = k = labels * sample_uniform01<double>(eng); }
      label.increment(k, uniform_label, eng);
      for (auto& w : doc)
        label_term[k].increment(w, uniform_word, eng);
    }
    if (sample == 0 || sample % 10 == 9) {
      cerr << " [LLH=" << log_likelihood(label, uniform_label, label_term, uniform_word) << "]" << endl;
      if (sample % 30u == 29) {
        label.resample_hyperparameters(eng);
        cerr << "label.crp(d=" << label.discount() << ",s=" << label.strength() << ")\n";
        for (auto& crp : label_term)
          crp.resample_hyperparameters(eng);
      }
    } else { cerr << '.' << flush; }
  }

  // print out highest probability words in each label
  vector<double> p(vocab.size());
  vector<unsigned> ind(vocab.size());
  int k = 0;
  for (auto& lt : label_term) {
    if (lt.num_customers() < 5) { k++; cerr << "LABEL NOT USED\n"; continue; }
    for (unsigned j = 0; j < vocab.size(); ++j) {
      p[j] = lt.prob(j, uniform_word);
      ind[j] = j;
    }
    cerr << "LABEL " << k << " (d=" << lt.discount() << ", s=" << lt.strength() << ") p=" << label.prob(k, uniform_label) << endl;
    ++k;
    partial_sort(ind.begin(), ind.begin() + 10, ind.end(), [&p](unsigned a, unsigned b) { return p[a] > p[b]; });
    for (int j = 0; j < 10; ++j) cerr << " " << dict.Convert(ind[j]) << ':' << p[ind[j]];
    cerr << endl;
  }
  cerr << label << endl;
  return 0;
}

