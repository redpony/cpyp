all: hpyplm dhpyplm

hpyplm: hpyplm.cc
	g++ -std=c++11 -O3 -Wall -I.. hpyplm.cc -o hpyplm

hpyplm_train: hpyplm_train.cc
	g++ -std=c++11 -O3 -Wall -I.. hpyplm_train.cc -o hpyplm_train -lboost_serialization

hpyplm_query: hpyplm_query.cc
	g++ -std=c++11 -O3 -Wall -I.. hpyplm_query.cc -o hpyplm_query -lboost_serialization

dhpyplm: dhpyplm.cc
	g++ -std=c++11 -O3 -g -Wall -I.. dhpyplm.cc -o dhpyplm

dhpyplm_train: dhpyplm_train.cc
	g++ -std=c++11 -O3 -Wall -I.. dhpyplm_train.cc -o dhpyplm_train -lboost_serialization

dhpyplm_query: dhpyplm_query.cc
	g++ -std=c++11 -O3 -Wall -I.. dhpyplm_query.cc -o dhpyplm_query -lboost_serialization

cdec_ff_hpyplm.o: cdec_ff_hpyplm.cc
	g++ -shared -fPIC -std=c++11 -O3 -g -Wall -I.. cdec_ff_hpyplm.cc -c

libcdec_ff_hpyplm.so: cdec_ff_hpyplm.o
	g++ -shared -Wl,-soname,libcdec_ff_hpyplm.so -o libcdec_ff_hpyplm.so cdec_ff_hpyplm.o
