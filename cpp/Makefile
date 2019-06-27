CXXFLAGS = -std=c++17 -Wall
CXXFLAGS_CEGB = -fopenmp -std=c++11 -pthread -Og -g -Wextra -Wall \
	 -Wno-unknown-pragmas -fPIC

ifeq ($(DEBUG), 1)
	CXXFLAGS += -g -O0 -p
else
	CXXFLAGS += -O2
endif


# Path to LightGBM-CEGB
LGBM_CEGB = 

INCLUDE = $(LGBM_CEGB)/include
CEGB_SRC =$(LGBM_CEGB)/build/CMakeFiles/lightgbm.dir/src

OBJ_FILES = $(CEGB_SRC)/application/*.cpp.o $(CEGB_SRC)/boosting/*.cpp.o \
	 $(CEGB_SRC)/io/*.cpp.o $(CEGB_SRC)/metric/*.cpp.o $(CEGB_SRC)/network/*.cpp.o \
	 $(CEGB_SRC)/objective/*.cpp.o $(CEGB_SRC)/treelearner/*.cpp.o

all: build

build: catboost cegb gate

catboost: catboost_eval.cpp parser.cpp parser.h
	$(CXX) $(CXXFLAGS) parser.cpp ripped_evaluator/evaluator.cpp \
	 catboost_eval.cpp -I ripped_evaluator/include -o catboost_eval

cegb: cegb_eval.cpp parser.cpp parser.h
	$(CXX) $(CXXFLAGS_CEGB) parser.cpp \
	 cegb_eval.cpp $(OBJ_FILES) -I $(INCLUDE) -o cegb_eval -rdynamic

gate: gate_eval.cpp parser.cpp parser.h
	$(CXX) $(CXXFLAGS_CEGB) parser.cpp \
	 gate_eval.cpp $(OBJ_FILES) -I $(INCLUDE) -o gate_eval -rdynamic

clean:
	rm -f catboost_eval cegb_eval gate_eval
