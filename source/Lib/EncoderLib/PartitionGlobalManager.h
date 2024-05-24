//
// Created by sbelhadj on 21/06/23.
//

#ifndef VVENC_PARTITIONGLOBALMANAGER_H
#define VVENC_PARTITIONGLOBALMANAGER_H

#include "PartitionPrediction.h"

extern PartitionPrediction * predict_partitionInter;
extern PartitionPrediction * predict_partitionIntra;
extern float time_ML;
extern float time_lgbm ;
extern float time_cnn ;
extern int cnn_calls;
extern int lgbm_calls;
extern std::string model_folder ;
extern int m_predictionModes ;
// 0: raw, 1: inter, 2: intra, 3: inter+intra
extern bool m_useGpu ;

#endif //VVENC_PARTITIONGLOBALMANAGER_H
