cpyp
====

C++ library for modeling with Pitman-Yor processes

## Features
- Memory-efficient histogram-based sampling scheme proposed by [Blunsom et al. (2009)](http://www.clg.ox.ac.uk/blunsom/pubs/blunsom-acl09-short.pdf)
- Full range of PYP hyperparameters (0 > strength > -discount, discount = 0, etc.)
- Beta/gamma priors on discount/strength hyperparameters
- Example implementations
    - Hierarchical Pitman-Yor process language model ([Teh, 2006](http://acl.ldc.upenn.edu/P/P06/P06-1124.pdf))
    - Latent Pitman-Yor allocation topic model (LDA with the D replaced)

## System Requirements
This software requires a C++ compiler that implements the [C++11 standard](http://en.wikipedia.org/wiki/C%2B%2B11), for example [gcc-4.7](http://gcc.gnu.org/) or [Clang-3.1](http://clang.llvm.org/) more recent. No other libraries or tools are required.

