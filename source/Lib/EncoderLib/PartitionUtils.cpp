//
// Created by sbelhadj on 20/06/23.
//

#include "PartitionUtils.h"
#include "PartitionGlobalManager.h"

std::vector<float> getLuma(vvenc::Picture * pic, int x, int y, int width, int height, float devideby) {
    std::vector<float> currPixels;
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            float pix ;
            if( (x+j>=0) && (x+j<pic->blocks[0].width) && (y+i>=0) && (y+i<pic->blocks[0].height) )
                pix = *pic->getOrigBuf().bufs[0].bufAt(x+j, y+i)/4 ;
            else
                pix=128 ;
            currPixels.push_back(pix/devideby);
        }
    }
    return currPixels;
}

std::string getLumaFromPicture(vvenc::Picture * pic) {
    std::string currPixels = "[";
    for (int i = 0; i < pic->lheight(); i++){
        for (int j = 0; j < pic->lwidth(); j++)
        {
            int pix = *pic->getOrigBuf().bufs[0].bufAt(j, i)/4 ;
            currPixels += std::to_string(pix) + ", ";
        }
    }
    return currPixels.substr(0, currPixels.length()-2)+"]";
}

void extract_from_vector_inter(std::vector<float> * full_vector, double* sub_vector, int x, int y, int w, int h) {
    x/=2 ;
    y/=2 ;
    w = w/2 -1 ;
    h = h/2 - 1 ;
    int k = 0;
    int start = y + 63 * (x / 2);
    for (int i = 0; i < h / 2; ++i){
        for (int j = 0; j < w; ++j)
            sub_vector[k++] = full_vector->at(start++);
        start += 63 - w;
    }
    if ( x+h==63 ) {
        start = 1953 + y / 2;
        for (int i = 0; i < w / 2; ++i)
            sub_vector[k++] = full_vector->at(start++);
    }
    else{
        start++ ;
        for (int i = 0; i < w/2; ++i) {
            sub_vector[k++] = full_vector->at(start);
            start += 2;
        }
    }
}

void extract_from_vector_intra(std::vector<float> * full_vector, double* sub_vector, int x, int y, int w, int h) {
    x/=2 ;
    y/=2 ;
    w = w/2 -1 ;
    h = h/2 - 1 ;
    int k = 0;
    int start = y + 31 * (x / 2);
    for (int i = 0; i < h / 2; ++i){
        for (int j = 0; j < w; ++j)
            sub_vector[k++] = full_vector->at(start++);
        start += 31 - w;
    }
    if ( x+h==31 ) {
        start = 465 + y / 2;
        for (int i = 0; i < w / 2; ++i)
            sub_vector[k++] = full_vector->at(start++);
    }
    else{
        start++ ;
        for (int i = 0; i < w/2; ++i) {
            sub_vector[k++] = full_vector->at(start);
            start += 2;
        }
    }
}

//uint8_t * splitsToTry(double *splitProba, int splitNumber){
//
//    uint8_t *splitDecision = new uint8_t[6]; // 0 = QT, 1 = BTH, 2 = BTV, 3 = TTH, 4 = TTV, 5 = NS
//    for (int i = 0; i < 6; i++) splitDecision[i] = 0;
//    for (int i = 0; i < splitNumber; ++i) {
//        double * max = std::max_element(splitProba, splitProba + 5);
//        int argmaxVal = std::distance(splitProba, max);
//        splitDecision[argmaxVal] = 1 ;
//        splitProba[argmaxVal] = -1 ;
//    }
//    return splitDecision ;
//}

void splitChoiceML_inter(int qp, int width, int height, int x, int y, std::vector<float> * proba, int * splitDecision, int splitNumber){
    double input[1985];
    double *output = (double *) malloc(sizeof(double) * 6);
    int vector_length = (width/4-1)*(height/4)+(width/4)*(height/4-1);
//    std::cout << width << " " << height << " " << x << " " << y << std::endl ;
    double * subvector = (double * ) malloc(sizeof(double) * vector_length);
    extract_from_vector_inter(proba, subvector, y, x, width, height);

    input[0] = qp;
    for(int i=0;i<vector_length;i++){
        input[i+1] = subvector[i];
    }

    // lgbm inference
    clock_t lgbm_inference_timer = clock();
    predict_partitionInter->predict_once_inter(input, output, std::make_pair(height, width));
    float lgbm_inference_time = ((double) clock() - (double) lgbm_inference_timer) / CLOCKS_PER_SEC;
    time_ML += lgbm_inference_time ;
    time_lgbm += lgbm_inference_time ;
    lgbm_calls++ ;

    std::vector<int> class_labels = predict_partitionInter->classes[std::to_string(height)][std::to_string(width)]["labels"] ;
    //taking top
    int auto_top ;
    if (splitNumber==0) auto_top = predict_partitionInter->classes[std::to_string(height)][std::to_string(width)]["top"] ;
    else if (class_labels.size()<splitNumber) auto_top = class_labels.size() ;
    else auto_top=splitNumber ;

    //preprocess output
    double corrected_vect[6] = {0, 0, 0, 0, 0, 0} ;
    for (int i = 0; i < class_labels.size() ; ++i)
        corrected_vect[class_labels[i]] = output[i] ;


    // choosing TOP splits
//    int splitNumber = 3 ; // TOP3

    for (int i = 0; i < 6; ++i) splitDecision[i]=0 ;
    for (int i = 0; i < auto_top; ++i) {
        double * max = std::max_element(corrected_vect, corrected_vect + 5);
        int argmaxVal = std::distance(corrected_vect, max);
        splitDecision[argmaxVal] = 1 ;
        corrected_vect[argmaxVal] = -1 ;
    }

//    for (int i = 0; i < 6; ++i)
//        std::cout << splitDecision[i] << " " ;
//    std::cout << " | ";
}

void splitChoiceML_intra(int qp, int width, int height, int x, int y, std::vector<float> * proba, int * splitDecision, int splitNumber){
    double input[481] ;
    double *output = (double *) malloc(sizeof(double) * 6);
    int vector_length = (width/4-1)*(height/4)+(width/4)*(height/4-1);
    double * subvector = (double * ) malloc(sizeof(double) * vector_length);
    extract_from_vector_intra(proba, subvector, y, x, width, height);
    input[0] = qp;
    for(int i=0;i<vector_length;i++){
        input[i+1] = subvector[i];
    }
    // lgbm inference
    clock_t lgbm_inference_timer = clock();
    predict_partitionIntra->predict_once(input, output, std::make_pair(height, width));
    float lgbm_inference_time = ((double) clock() - (double) lgbm_inference_timer) / CLOCKS_PER_SEC;
    time_ML += lgbm_inference_time ;
    time_lgbm += lgbm_inference_time ;
    lgbm_calls++ ;

    std::vector<int> class_labels = predict_partitionIntra->classes[std::to_string(height)][std::to_string(width)]["labels"] ;
    //taking top
    int auto_top ;
    if (splitNumber==0) auto_top = predict_partitionIntra->classes[std::to_string(height)][std::to_string(width)]["top"] ;
    else if (class_labels.size()<splitNumber) auto_top = class_labels.size() ;
    else auto_top=splitNumber ;

    //preprocess output
    double corrected_vect[6] = {0, 0, 0, 0, 0, 0} ;
    for (int i = 0; i < class_labels.size() ; ++i)
        corrected_vect[class_labels[i]] = output[i] ;


    // choosing TOP splits
    for (int i = 0; i < 6; ++i) splitDecision[i]=0 ;
    for (int i = 0; i < auto_top; ++i) {
        double * max = std::max_element(corrected_vect, corrected_vect + 5);
        int argmaxVal = std::distance(corrected_vect, max);
        splitDecision[argmaxVal] = 1 ;
        corrected_vect[argmaxVal] = -1 ;
    }

}



