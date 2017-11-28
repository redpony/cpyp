cpyp
====

`cpyp` is a C++ library for nonparametric Bayesian modeling with Pitman-Yor process priors

## Features
- Memory-efficient histogram-based sampling scheme proposed by [Blunsom et al. (2009)](http://www.clg.ox.ac.uk/blunsom/pubs/blunsom-acl09-short.pdf)
- Full range of PYP hyperparameters (0 ≤ discount < 1, strength > -discount, etc.)
- Beta priors on discount hyperparameter
- (Conditional, given discount) Gamma prior on strength hyperparameter
- Tied hyperparameters
- Slice sampling for hyperparameter inference
- “Multifloor” Chinese Restaurant processes to perform inference in graphical Pitman-Yor processes
- Serialization of CRPs using [Boost.Serialization](www.boost.org/libs/serialization) (optional)
- Example implementations
    - Hierarchical Pitman-Yor process language model ([Teh, 2006](http://acl.ldc.upenn.edu/P/P06/P06-1124.pdf))
    - Domain adapting graphical Pitman-Yor process language model ([Wood & Teh, 2009](http://jmlr.csail.mit.edu/proceedings/papers/v5/wood09a/wood09a.pdf))
    - Latent Pitman-Yor allocation topic model (LDA with the D replaced)
    - Unsupervised “Naive Bayes” single-membership clustering, using fast Metropolis-Hastings sampling

## System Requirements
This library should work with any C++ compiler that implements the [C++11 standard](http://en.wikipedia.org/wiki/C%2B%2B11). No other libraries are required.

