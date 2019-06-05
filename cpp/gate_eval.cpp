#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <vector>
#include <limits>
#include <chrono>

#include "./parser.h"

#include <LightGBM/boosting.h>
#include <LightGBM/prediction_early_stop.h>

struct DataObject{
    size_t id;
    std::vector<double> features = std::vector<double>(N_FEATURES);
    float prediction;
};

int main(int argc, char *argv[]) {
    std::string input_file;
    std::string output_file;
    int first_model_file;
    int second_model_file;
    int gate_model_file;
    double threshold;
    std::string time_output;
    for (int it = 1; it < argc; ++it) {
        std::string arg(argv[it]);
        if (arg == "-input") {
            input_file = argv[it + 1];
        } else if (arg == "-output") {
            output_file = argv[it + 1];
        } else if (arg == "-gate") {
            gate_model_file = it + 1;
        } else if (arg == "-first") {
            first_model_file = it + 1;
        } else if (arg == "-second") {
            second_model_file = it + 1;
        } else if (arg == "-time") {
            time_output = argv[it + 1];
        } else if (arg == "-threshold") {
            threshold = std::atof(argv[it + 1]);
        }
    }
    std::fstream fin(input_file, std::fstream::in);
    std::vector<DataObject> data;
    fin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    std::vector<float> features(N_FEATURES);
    while (fin.good() && fin.peek() != EOF) {
        data.push_back(DataObject());
        ugly_hardcoded_parse(fin, &data.back().id, &features);
        for (size_t it = 0; it < features.size(); ++it) {
            data.back().features[it] = features[it];
        }
    }
    fin.close();
    
    std::fstream fout(output_file, std::fstream::out);
    fout << std::setprecision(std::numeric_limits<float>::max_digits10);
    fout << "id,prediction\n";

    double output[2] = {0, 0};
    auto first_model = LightGBM::Boosting::CreateBoosting(argv[first_model_file]);
    auto second_model = LightGBM::Boosting::CreateBoosting(argv[second_model_file]);
    auto gate_model = LightGBM::Boosting::CreateBoosting(argv[gate_model_file]);

    auto stop = LightGBM::CreatePredictionEarlyStopInstance("binary",
        LightGBM::PredictionEarlyStopConfig());
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t it = 0; it < data.size(); ++it) {
        gate_model->Predict(data[it].features.data(), &output[0], &stop);
        if (output[0] > threshold) {
            output[0] = 0;
            first_model->PredictRaw(data[it].features.data(), &output[0], &stop);
        } else {
            output[0] = 0;
            second_model->PredictRaw(data[it].features.data(), &output[0], &stop);
        }
        data[it].prediction = output[0];
        output[0] = 0;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - start).count();

    for (auto& it : data) {
        fout << it.id << DELIMITER << it.prediction  << '\n';
    }
    fout.close();
    std::fstream ftout(time_output, std::fstream::out);
    ftout << time;
    ftout.close();
    return 0;
}
