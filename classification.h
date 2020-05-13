#ifndef __CLASSIFICATION_H_
#define __CLASSIFICATION_H_

/**
 * (@liam):
 * This class definition was moved from the classification.cpp that is provided by Caffe
 * in their examples. Moved to this .h so that the go-caffe-interface files can access
 * this type.
 */

#include <opencv2/core/core.hpp>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
 public:
  Classifier(const string& model_file,
             const string& trained_file,
             const string& mean_file,
             const string& label_file,
             FILE* log);

  std::vector<Prediction> Classify(const cv::Mat& img, int N = 5, FILE* log = NULL);
    
 private:
  void SetMean(const string& mean_file);

  std::vector<float> Predict(const cv::Mat& img);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
  std::vector<string> labels_;
};

#endif
