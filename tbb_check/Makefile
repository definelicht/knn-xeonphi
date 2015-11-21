.PHONY: clean

all: tbb_check

tbb_check: tbb_check.cpp
	g++ -std=c++11 -o tbb_check tbb_check.cpp -ltbb

clean:
	rm -f tbb_check *~
