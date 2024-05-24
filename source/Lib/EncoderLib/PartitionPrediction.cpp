//
// Created by altissie on 18/10/2019.
//

#include "PartitionPrediction.h"

PartitionPrediction::PartitionPrediction(std::string model_folder, std::string mode, bool use_gpu){
    this->mode = mode ;
    this->model_folder = model_folder ;
    this->use_gpu = use_gpu ;
    if(mode=="inter")
        this->partsize_list = {
            std::make_pair(128, 128),
            std::make_pair(128, 64),
	        std::make_pair(64, 128),
//	        std::make_pair(128, 32),
//	        std::make_pair(32, 128),
//	        std::make_pair(128, 16),
//	        std::make_pair(16, 128),
	        std::make_pair(16, 64),
            std::make_pair(32, 64),
            std::make_pair(4, 64),
            std::make_pair(64, 16),
            std::make_pair(64, 32),
            std::make_pair(64, 4),
            std::make_pair(8, 64),
            std::make_pair(4, 8),
            std::make_pair(64, 64),
            std::make_pair(64, 8),
            std::make_pair(8, 32),
            std::make_pair(16, 8),
            std::make_pair(8, 8),
            std::make_pair(4, 32),
            std::make_pair(32, 32),
            std::make_pair(8, 4),
            std::make_pair(16, 16),
            std::make_pair(32, 8),
            std::make_pair(16, 4),
            std::make_pair(32, 4),
            std::make_pair(4, 16),
            std::make_pair(32, 16),
            std::make_pair(8, 16),
            std::make_pair(16, 32),
    } ;
    if(mode=="intra")
        this->partsize_list = {
                std::make_pair(4, 8),
                std::make_pair(64, 64),
                std::make_pair(8, 32),
                std::make_pair(16, 8),
                std::make_pair(8, 8),
                std::make_pair(4, 32),
                std::make_pair(32, 32),
                std::make_pair(8, 4),
                std::make_pair(16, 16),
                std::make_pair(32, 8),
                std::make_pair(16, 4),
                std::make_pair(32, 4),
                std::make_pair(4, 16),
                std::make_pair(32, 16),
                std::make_pair(8, 16),
                std::make_pair(16, 32)
        };
    std::string model_path = model_folder + "cnn_model/"+mode+"/model.onnx" ;
    const char *cstr = model_path.c_str();
    this->model = new ONNXModel(cstr, this->use_gpu) ;
}


PartitionPrediction::~PartitionPrediction(){
  //free(this->model);
}

void PartitionPrediction::initializeModels(){
    for (int i = 0; i < partsize_list.size(); ++i) {
        //load model file
        std::ifstream model_file ;
        std::string filename = model_folder+"ML_model/"+mode+"/lgbm_"+std::to_string(partsize_list[i].first)+"x"+std::to_string(partsize_list[i].second)+".txt";
        const char * charFilename = filename.c_str();
        model_file.open(filename, std::ifstream::in) ;
        std::string model_content((std::istreambuf_iterator<char>(model_file)), std::istreambuf_iterator<char>());
        unsigned long size_t = model_content.length() ;
        const char *cstr = model_content.c_str();
        models[partsize_list[i]] = LightGBM::Boosting::CreateBoosting("gbdt",charFilename);
        models[partsize_list[i]]->LoadModelFromString(cstr, size_t) ;
        model_file.close() ;
    }
    //loading classes from json
    std::ifstream ifs(model_folder+"ML_model/classes_"+mode+".json");
    classes = json::parse(ifs);
    ifs.close() ;
}

void PartitionPrediction::predict_once(double* input, double* output, partsize size){
  LightGBM::PredictionEarlyStopConfig test;
  std::string classif = "multiclass";
  if((size.first==64&&size.second==64) || (size.first==8&&size.second==4) || (size.first==4&&size.second==8)){
    classif = "binary";
  }
  const LightGBM::PredictionEarlyStopInstance early_stop = LightGBM::CreatePredictionEarlyStopInstance(classif,test);
  this->models[size]->Predict(input, output, &early_stop) ;
}

void PartitionPrediction::predict_once_inter(double* input, double* output, partsize size){
  LightGBM::PredictionEarlyStopConfig test ;
  std::string classif = "multiclass";
  if((size.first==8&&size.second==4) || (size.first==4&&size.second==8) || (size.first==128&&size.second==64)){
    classif = "binary";
  }
  const LightGBM::PredictionEarlyStopInstance early_stop = LightGBM::CreatePredictionEarlyStopInstance(classif,test);
//    const LightGBM::PredictionEarlyStopInstance * early_stop = nullptr ;
  this->models[size]->Predict(input, output, &early_stop) ;
}

bool PartitionPrediction::check_size(int x, int y){
    partsize tmp(x, y) ;
    return (std::find(this->partsize_list.begin(), this->partsize_list.end(), tmp) != this->partsize_list.end());
}
