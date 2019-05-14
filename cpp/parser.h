// Copyright 2019, Nikita Kazeev, Higher School of Economics
#pragma once
#include <ctype.h>
#include <cstdint>
#include <cmath>

#include <iostream>
#include <vector>
#include <array>
#include <limits>

// The structure of .csv is the following:
// id, <62 float features>
const size_t N_RAW_FEATURES = 62;
const size_t N_FEATURES = N_RAW_FEATURES;
const float EMPTY_FILLER = 1000;

const char DELIMITER = ',';

void ugly_hardcoded_parse(std::istream& stream, size_t* id, std::vector<float>* result);
