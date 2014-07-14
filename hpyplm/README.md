This describes how to replicate the [Wood & Teh (AISTATS 2009)](http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS09_WoodT.pdf) that use the doubly hierarchical PYP LM as a domain adapting language model.

There are two corpora provided (see below):
 * The Brown corpus (Francis & Kucera)
 * The State of the Union corpus

The test and training data are:
 * The test set is the SOTU speeches by Lyndon Johnson.
 * The training corpus consists of the Brown corpus (domain general) and the SOTU speeches by the other presidents (domain specific). In the base condition, the domain general and domain specific corpora are merged. The the domain aware LM, the Brown corpus LM is used as a possible backoff for the SOTU-specific LM.

## Get the data

Download the [Brown corpus and state-of-the-union data](http://demo.clab.cs.cmu.edu/cdyer/dhpyplm-data.tar.gz)

    tar xzf dhpyplm-data.tar.gz
    cd dhpyplm-data

Create the merged training data

    cat brown.txt sotu-train.txt > all-train.txt

## Run the HPYPLM baseline

Command to run the HPYPLM sampler for 100 iterations (generally, it does not take many iterations to converge to a mode; test perplexity is computed from the posterior predictive distribution of the final sample):

    ../hpyplm all-train.txt sotu-test.txt 100

Expected output (will vary slightly due to random variation):

    Seeding random number sequence to 2287171713
    Reading corpus...
    Reading from all-train.txt
    E-corpus size: 72722 sentences	 (45595 word types)
    Reading from sotu-test.txt
    ......... [LLH=-1.01823e+07]
    ......... [LLH=-1.01624e+07]
    ......... [LLH=-1.01556e+07]
    Resampled 495587 CRPs (d=0.742216,s=5.06505) = -1.5481e+06
    Resampled 45596 CRPs (d=0.7916,s=1.83388) = -3.36235e+06
    Resampled 1 CRPs (d=0.783559,s=6.05452) = -4.08742e+06
    ......... [LLH=-1.00581e+07]
    <snip>
    p(americans | for all) = -3.75982
    p(. | all americans) = -2.71527
    p(</s> | americans .) = -0.00326708
      Log_10 prob: -137096
            Count: 64095
             OOVs: 209
    Cross-entropy: 7.10546
       Perplexity: 137.707

## Run the DHPYPLM condition

Command to run the HPYPLM sampler for 1,000 iterations (the DHPYPLM requires far more iterations than the HPYPLM to obtain good performance; as above, test perplexity is computed from the posterior predictive distribution of the final sample):

    ../dhpyplm sotu-train.txt brown.txt sotu-test.txt 1000

Note that the order of the command line options determines which is the "in domain" training data and which is the "domain general" training data.

Results (will vary slightly due to MCMC variation):

    Seeding random number sequence to 1695250841
    *Corpus : sotu-train.txt
    Corpus : brown.txt
    Reading from sotu-test.txt
    Reading from sotu-train.txt
    Reading from brown.txt
    E-corpus size: 15382 sentences	 (45595 word types)
    ......... [LLH=-1.11234e+07]
    ......... [LLH=-1.11552e+07]
    ......... [LLH=-1.11742e+07]
    Path<3> d=0.115053,s=0.506993 p(in_domain) = 0.408102
    Resampled 111947 CRPs (d=0.766203,s=2.12444) = -362698
    <snip>

