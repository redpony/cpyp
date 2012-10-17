all: hpyplm dhpyplm

hpyplm: hpyplm.cc
	g++ -std=c++11 -O3 -Wall -I.. hpyplm.cc -o hpyplm

dhpyplm: dhpyplm.cc
	g++ -std=c++11 -O3 -g -Wall -I.. dhpyplm.cc -o dhpyplm

## stuff below here is optional

BOOST_ROOT=/cab0/tools/boost-1.49.0
BOOST_INCLUDE=$(BOOST_ROOT)/include
BOOST_SERIALIZATION=$(BOOST_ROOT)/lib/libboost_serialization.a

hpyplm_train: hpyplm_train.cc
	g++ -std=c++11 -O3 -Wall -I$(BOOST_INCLUDE) -I.. hpyplm_train.cc -o hpyplm_train $(BOOST_SERIALIZATION)

hpyplm_query: hpyplm_query.cc
	g++ -std=c++11 -O3 -Wall -I$(BOOST_INCLUDE) -I.. hpyplm_query.cc -o hpyplm_query $(BOOST_SERIALIZATION)


dhpyplm_train: dhpyplm_train.cc
	g++ -std=c++11 -O3 -Wall -I$(BOOST_INCLUDE) -I.. dhpyplm_train.cc -o dhpyplm_train $(BOOST_SERIALIZATION)

dhpyplm_query: dhpyplm_query.cc
	g++ -std=c++11 -O3 -Wall -I$(BOOST_INCLUDE) -I.. dhpyplm_query.cc -o dhpyplm_query $(BOOST_SERIALIZATION)

CDEC = ../../cdec
cdec_ff_hpyplm.o: cdec_ff_hpyplm.cc
	g++ -shared -fPIC -std=c++11 -O3 -g -Wall -I$(BOOST_INCLUDE) -I.. -I$(CDEC)/utils -I$(CDEC)/decoder cdec_ff_hpyplm.cc -c

libcdec_ff_hpyplm.so: cdec_ff_hpyplm.o
	g++ -shared -Wl,-soname,libcdec_ff_hpyplm.so -o libcdec_ff_hpyplm.so cdec_ff_hpyplm.o

