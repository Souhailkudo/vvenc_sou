//
// Created by sbelhadj on 06/03/24.
//

#include "ONNXModel.h"
#include <iostream>
ONNXModel::ONNXModel(const char * model_path, bool useGPU) {
    this->model_path = model_path ;
    this->useGPU = useGPU ;

    // model initialization
    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetIntraOpNumThreads(1);
    // Optimization will take time and memory during startup
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    // CUDA options. If used.
    if (this->useGPU)
    {
        cuda_options.device_id = 0;  //GPU_ID
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive; // Algo to search for Cudnn
        cuda_options.arena_extend_strategy = 0;
        // May cause data race in some condition
        cuda_options.do_copy_in_default_stream = 0;
        sessionOptions.AppendExecutionProvider_CUDA(cuda_options); // Add CUDA options to session options
    }
    // Model path is const wchar_t*
    this->session = new Ort::Session(env, this->model_path, sessionOptions);



    size_t num_input_nodes = session->GetInputCount();
    input_node_names = new std::vector<const char*>;
    Ort::AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < num_input_nodes; i++) {
        char* tempstring = new char[strlen(session->GetInputNameAllocated(i, allocator).get()) + 1];
        strcpy(tempstring, session->GetInputNameAllocated(i, allocator).get());
        input_node_names->push_back(tempstring);
        Ort::TypeInfo* type_info = new Ort::TypeInfo(session->GetInputTypeInfo(i));
        auto tensor_info = type_info->GetTensorTypeAndShapeInfo();
        input_node_dims.push_back(tensor_info.GetShape());
        if (input_node_dims[i].at(0) == -1) input_node_dims[i].at(0) = 1 ;
        //delete(type_info);
    }

    // get output info from model
    size_t num_output_nodes = session->GetOutputCount();
    output_node_names = new std::vector<const char*>;
    // use same allocator
    for (int i = 0; i < num_output_nodes; i++) {
        char* tempstring = new char[strlen(session->GetOutputNameAllocated(i, allocator).get()) + 1];
        strcpy(tempstring, session->GetOutputNameAllocated(i, allocator).get());
        output_node_names->push_back(tempstring);
        Ort::TypeInfo output_type_info(session->GetOutputTypeInfo(i));
        auto tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_node_dims.push_back(tensor_info.GetShape());
        if (output_node_dims[i].at(0) == -1) output_node_dims[i].at(0) = 1 ;
    }
    // show input_no_dim
//    for (int i = 0; i < input_node_dims.size(); ++i) {
//        for (int j = 0; j < input_node_dims[i].size(); ++j) {
//            std::cout << input_node_dims[i][j] << std::endl ;
//        }
//    }
}

ONNXModel::~ONNXModel() {
    // free the model
}

std::vector<float> ONNXModel::predict(std::vector<float> ctu_pixels, float qp) {
    // Used to allocate memory for input
    Ort::MemoryInfo memory_info{ nullptr };
    memory_info =Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
//    memory_info =Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, Ort)
    std::vector<float> qp_input = {qp};
    // preprocess input for onnx
    size_t input1_size = ctu_pixels.size();
    size_t input2_size = 1;
    std::vector<Ort::Value> inputTensor;
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info,(float*)ctu_pixels.data(), input1_size, input_node_dims[0].data(), input_node_dims[0].size()));
    inputTensor.emplace_back(Ort::Value::CreateTensor<float>(memory_info, (float*)qp_input.data(), input2_size, input_node_dims[1].data(), input_node_dims[1].size()));
    // preprocess output for onnx
    std::vector<Ort::Value> outputTensor;
    size_t outputTensorSize = output_node_dims[0][0]*output_node_dims[0][1];
    std::vector<float> outputTensorValues(outputTensorSize);

//    std::cout << input_node_names->data() << std::endl;
//    std::cout << inputTensor.size() << std::endl;
//    std::cout << inputTensor.data() << std::endl ;

//    std::cout << inputTensor.size() << std::endl;
//    std::cout << inputTensor.data() << std::endl ;

    outputTensor.push_back(Ort::Value::CreateTensor<float>(
            memory_info, outputTensorValues.data(), outputTensorSize,
            output_node_dims[0].data(), output_node_dims[0].size()));
    outputTensor = session->Run(Ort::RunOptions{ nullptr }, input_node_names->data(), inputTensor.data(), inputTensor.size(), output_node_names->data(), 1);
    float* arr = outputTensor.front().GetTensorMutableData<float>();
    int output_count = outputTensor.front().GetTensorTypeAndShapeInfo().GetElementCount() ;
    std::vector<float> output_vector {arr, arr + output_count};
    return output_vector;
}
