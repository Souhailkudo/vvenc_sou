//
// Created by sbelhadj on 06/03/24.
//

#ifndef VVENC_ONNXMODEL_H
#define VVENC_ONNXMODEL_H

#include <onnxruntime_cxx_api.h>

class ONNXModel{
public:
    ONNXModel(const char * model_path, bool useGPU) ;
    ~ONNXModel() ;
    std::vector<float> predict(std::vector<float> ctu_pixels, float qp) ;


private:
    const char * model_path ;
    bool useGPU ;
    Ort::Session  * session ;
    std::vector<const char*>* input_node_names = nullptr; // Input node names
    std::vector<std::vector<int64_t>> input_node_dims;    // Input node dimension.
    std::vector<const char*>* output_node_names = nullptr; // Output node names
    std::vector<std::vector<int64_t>> output_node_dims;    // Output node dimension.
    Ort::Env env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "Default");
    Ort::SessionOptions sessionOptions;
    OrtCUDAProviderOptions cuda_options;

};

#endif //VVENC_ONNXMODEL_H
