//
// Created by sbelhadj on 20/06/23.
//

#ifndef VVENC_PARTITIONUTILS_H
#define VVENC_PARTITIONUTILS_H

#include <vector>
#include <string>
#include "EncModeCtrl.h"

std::vector<float> getLuma(vvenc::Picture * pic, int x, int y, int width, int height, float devideby) ;
std::string getLumaFromPicture(vvenc::Picture * pic) ;

void extract_from_vector_inter(std::vector<float> * full_vector, double* sub_vector, int x, int y, int w, int h) ;
void splitChoiceML_inter(int qp, int width, int height, int x, int y, std::vector<float> * proba, int * splitDecision, int splitNumber) ;
void splitChoiceML_intra(int qp, int width, int height, int x, int y, std::vector<float> * proba, int * splitDecision, int splitNumber) ;
uint8_t * splitsToTry(double *splitProba, int splitNumber) ;


#endif //VVENC_PARTITIONUTILS_H
