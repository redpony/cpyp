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

double log_likelihood(const tied_parameter_resampler<crp<short>>& p,
                      const vector<crp<short>>& dt,
                      double ut,
                      const vector<crp<unsigned>>& tt,
                      double uw) {
  double llh = p.log_likelihood();
  for (auto& crp : dt) llh += crp.num_tables() * log(ut);
  for (auto& crp : tt) llh += crp.log_likelihood() + crp.num_tables() * log(uw);
  return llh;
}

void topic_summary(const unsigned vocab_size, const vector<crp<unsigned>>& topic_term, const double uniform_word) {
  vector<double> p(vocab_size);
  vector<unsigned> ind(vocab_size);
  for (auto& topic : topic_term) {
    for (unsigned j = 0; j < vocab_size; ++j) {
      p[j] = topic.prob(j, uniform_word);
      ind[j] = j;
    }
    cerr << "TOPIC (d=" << topic.discount() << ", s=" << topic.strength() << ")\n  ";
    partial_sort(ind.begin(), ind.begin() + 10, ind.end(), [&p](unsigned a, unsigned b) { return p[a] > p[b]; });
    for (int j = 0; j < 10; ++j) cerr << " " << dict.Convert(ind[j]) << ':' << p[ind[j]];
    cerr << endl;
  }
}

int main(int argc, char** argv) {
  if (argc != 4) {
    cerr << argv[0] << " <training.txt> <ntopics> <nsamples>\n\nEstimate a 'Latent Pitman-Yor Allocation' model\nInput format: each line in <training.txt> is a document\n";
    return 1;
  }
  MT19937 eng;
  string train_file = argv[1];
  const unsigned topics = atoi(argv[2]);
  const unsigned samples = atoi(argv[3]);
  
  vector<vector<unsigned> > corpus;
  set<unsigned> vocab;
  ReadFromFile(train_file, &dict, &corpus, &vocab);
  cerr << "Corpus size: " << corpus.size() << " documents\t (" << vocab.size() << " word types)\n";
  const double uniform_topic = 1.0 / topics;
  const double uniform_word = 1.0 / vocab.size();
  vector<vector<short> > z;  // topic indicators
  z.resize(corpus.size());
  vector<crp<unsigned>> topic_term(topics, crp<unsigned>(1,1,1,1));
  vector<crp<short>> doc_topic(corpus.size(), crp<short>(0.1,1));
  tied_parameter_resampler<crp<short>> doc_params(1,1,1,1,0.1,1);
  for (unsigned i = 0; i < corpus.size(); ++i) {
    doc_params.insert(&doc_topic[i]);
    z[i].resize(corpus[i].size());
  }
  vector<double> probs(topics);
  for (unsigned sample=0; sample < samples; ++sample) {
    for (unsigned i = 0; i < corpus.size(); ++i) {
      const auto& doc = corpus[i];
      for (unsigned j = 0; j < doc.size(); ++j) {
        const unsigned w = doc[j];
        short& z_ij = z[i][j];
        if (sample > 0) {
          doc_topic[i].decrement(z_ij, eng);
          topic_term[z_ij].decrement(w, eng);
        }
        for (unsigned k = 0; k < topics; ++k)
          probs[k] = doc_topic[i].prob(k, uniform_topic) * topic_term[k].prob(w, uniform_word);
        multinomial_distribution<double> mult(probs);
        z_ij = mult(eng);  // resample z_ij
        doc_topic[i].increment(z_ij, uniform_topic, eng);
        topic_term[z_ij].increment(w, uniform_word, eng);
      }
    }
    if (sample % 10 == 9) {
      cerr << " [LLH=" << log_likelihood(doc_params, doc_topic, uniform_topic, topic_term, uniform_word) << "]" << endl;
      if (sample % 30u == 29) {
        doc_params.resample_hyperparameters(eng);
        for (auto& crp : topic_term)
          crp.resample_hyperparameters(eng);
      }
      topic_summary(vocab.size(), topic_term, uniform_word);
    } else { cerr << '.' << flush; }
  }

  // print out highest probability words in each topic
  topic_summary(vocab.size(), topic_term, uniform_word);
  return 0;
}

