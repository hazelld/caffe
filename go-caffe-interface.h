#ifndef __GO_CAFFE_INTERFACE_H_
#define __GO_CAFFE_INTERFACE_H_

#ifdef __cplusplus
extern "C" {
#endif

#define MAX_ERROR_SIZE 4096
#define MAX_CLASSIFICATION_SIZE 256

#ifndef MAX_CLASSIFICATIONS 
#define MAX_CLASSIFICATIONS 5
#endif

typedef struct CaffePrediction {
  double confidence;
  char classification[MAX_CLASSIFICATION_SIZE];
} CaffePrediction; 

typedef struct CaffeReturn {
  int error_code;
  char error_log[MAX_ERROR_SIZE];
  CaffePrediction predictions[MAX_CLASSIFICATIONS];
  int prediction_count;
} CaffeReturn;

typedef struct CaffeConfig {
  const char* model_file;
  const char* trained_file;
  const char* mean_file;
  const char* label_file;
} CaffeConfig;


/**
 * caffe_init(CaffeConfig): This function will initialize the singleton classifier with
 * the given configuration.
 */
extern CaffeReturn caffe_init(CaffeConfig config);

/**
 * caffe_cleanup(): This function will cleanup the current singleton classifier
 */
extern CaffeReturn caffe_cleanup();

/**
 * caffe_classify(const char*): This function will classify the given image
 */
extern CaffeReturn caffe_classify(const char* image_file);


#ifdef __cplusplus
}
#endif

#endif
