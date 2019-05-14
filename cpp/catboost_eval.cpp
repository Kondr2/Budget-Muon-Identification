#include <fstream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <vector>
#include <limits>
#include <chrono>

#include "./parser.h"
#include "ripped_evaluator/evaluator.h"

struct DataObject{
    size_t id;
    std::vector<float> features = std::vector<float>(N_FEATURES);
    float prediction;
};

int main(int argc, char *argv[]) {
    std::string input_file;
    std::string output_file;
    std::string model_file;
    for (int it = 1; it < argc; ++it) {
        std::string arg(argv[it]);
        if (arg == "-input") {
            input_file = argv[it + 1];
        } else if (arg == "-output") {
            output_file = argv[it + 1];
        } else if (arg == "-model") {
            model_file = argv[it + 1];
        }
    }
    std::fstream fin(input_file, std::fstream::in);
    std::fstream fout(output_file, std::fstream::out);
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    fout << std::setprecision(std::numeric_limits<float>::max_digits10);
    fout << "id,prediction\n";
    std::vector<DataObject> data;

    while (fin.good() && fin.peek() != EOF) {
        data.push_back(DataObject());
        ugly_hardcoded_parse(fin, &data.back().id, &data.back().features);
    }
    fin.close();

    NCatboostStandalone::TOwningEvaluator evaluator(model_file);
    std::chrono::time_point start = std::chrono::high_resolution_clock::now();
    for (auto& obj : data) {
        obj.prediction = evaluator.Apply(obj.features,
                                         NCatboostStandalone::EPredictionType::RawValue);
    }
    std::chrono::time_point end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - start).count();

    for (auto& it : data) {
        fout << it.id << DELIMITER << it.prediction  << '\n';
    }
    fout.close();
    std::cout << time;
    return 0;
}
