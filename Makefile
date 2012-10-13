all: crp_test

	#g++ -std=c++11 -O3 -Wall crp_test.cc -o crp_test
crp_test: crp_test.cc
	clang++ -std=c++11 -O3 -Wall crp_test.cc -o crp_test
