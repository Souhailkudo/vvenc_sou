//
// Created by souhaiel on 19/06/2023.
//

#ifndef VVENC_PARTITIONPREDICTION_H
#define VVENC_PARTITIONPREDICTION_H

#include <iostream>
#include <fstream>
#include <LightGBM/prediction_early_stop.h>
#include <LightGBM/boosting.h>

#include <algorithm>
#include <vector>
#include <nlohmann/json.hpp>
#include "ONNXModel.h"
using json = nlohmann::json;


typedef std::pair<int, int> partsize ;

class PartitionPrediction {

public:
    PartitionPrediction(std::string modelFolder, std::string mode, bool use_gpu);
    ~PartitionPrediction();

    void initializeModels();
    void predict_once(double* input, double* output, partsize size);
    void predict_once_inter(double* input, double* output, partsize size);
    bool check_size(int x, int y) ;
    ONNXModel* model ;
    json classes ;

private:
    std::map<partsize, LightGBM::Boosting*> models ;
    std::vector<partsize> partsize_list;
    std::string mode ;
    std::string model_folder ;
    bool use_gpu ;
};


#endif //VVENC_PARTITIONPREDICTION_H
