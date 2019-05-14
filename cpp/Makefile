CXXFLAGS = -std=c++17 -Wall -Werror

ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -O0 -p
else
	CXXFLAGS += -O2
endif

all: build

build: catboost

run: catboost
	./catboost_eval

catboost: catboost_eval.cpp parser.cpp parser.h
	$(CXX) $(CXXFLAGS) parser.cpp ripped_evaluator/evaluator.cpp \
	 catboost_eval.cpp -I ripped_evaluator/include -o catboost_eval

clean:
	rm -f catboost_eval
