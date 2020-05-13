#include <caffe/caffe.hpp>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "classification.h"
#include "go-caffe-interface.h"

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file,
                       FILE* log) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif

  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file);

  /* Load labels. */
  std::ifstream labels(label_file.c_str());
  CHECK(labels) << "Unable to open labels file " << label_file;
  string line;
  while (std::getline(labels, line))
    labels_.push_back(string(line));

  Blob<float>* output_layer = net_->output_blobs()[0];
  CHECK_EQ(labels_.size(), output_layer->channels())
    << "Number of labels is different from the output layer dimension.";
}

static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
  return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
  std::vector<std::pair<float, int> > pairs;
  for (size_t i = 0; i < v.size(); ++i)
    pairs.push_back(std::make_pair(v[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

  std::vector<int> result;
  for (int i = 0; i < N; ++i)
    result.push_back(pairs[i].second);
  return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N, FILE* log) {
  std::vector<float> output = Predict(img);

  N = std::min<int>(labels_.size(), N);
  std::vector<int> maxN = Argmax(output, N);
  std::vector<Prediction> predictions;
  for (int i = 0; i < N; ++i) {
    int idx = maxN[i];
    predictions.push_back(std::make_pair(labels_[idx], output[idx]));
  }

  return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) {
  BlobProto blob_proto;
  ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

  /* Convert from BlobProto to Blob<float> */
  Blob<float> mean_blob;
  mean_blob.FromProto(blob_proto);
  CHECK_EQ(mean_blob.channels(), num_channels_)
    << "Number of channels of mean file doesn't match input layer.";

  /* The format of the mean file is planar 32-bit float BGR or grayscale. */
  std::vector<cv::Mat> channels;
  float* data = mean_blob.mutable_cpu_data();
  for (int i = 0; i < num_channels_; ++i) {
    /* Extract an individual channel. */
    cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
    channels.push_back(channel);
    data += mean_blob.height() * mean_blob.width();
  }

  /* Merge the separate channels into a single image. */
  cv::Mat mean;
  cv::merge(channels, mean);

  /* Compute the global mean pixel value and create a mean image
   * filled with this value. */
  cv::Scalar channel_mean = cv::mean(mean);
  mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);

  Preprocess(img, &input_channels);

  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* output_layer = net_->output_blobs()[0];
  const float* begin = output_layer->cpu_data();
  const float* end = begin + output_layer->channels();
  return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

/*
 * The following is the main() function that is provided for the example. Since we are
 * making this a library we can comment it out, but i'm leaving it incase we ever need it.
 *
int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " deploy.prototxt network.caffemodel"
              << " mean.binaryproto labels.txt img.jpg" << std::endl;
    return 1;
  }

  ::google::InitGoogleLogging(argv[0]);

  string model_file   = argv[1];
  string trained_file = argv[2];
  string mean_file    = argv[3];
  string label_file   = argv[4];
  Classifier classifier(model_file, trained_file, mean_file, label_file);

  string file = argv[5];

  std::cout << "---------- Prediction for "
            << file << " ----------" << std::endl;

  cv::Mat img = cv::imread(file, -1);
  CHECK(!img.empty()) << "Unable to decode image " << file;
  std::vector<Prediction> predictions = classifier.Classify(img);

  for (size_t i = 0; i < predictions.size(); ++i) {
    Prediction p = predictions[i];
    std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
              << p.first << "\"" << std::endl;
  }
}
*/

/* go-caffe-interface.cpp */
#define CAFFE_SUCCESS 0
#define CAFFE_ERROR -1

/** 
 * Keep a static Classifier instance that can be used between calls. Note revist this as
 * it isn't a very good pattern as it means we can only have one instance at a time with
 * one config. Ideally we could load an instance _per_ config type, meaning we could have
 * multiple models loaded at a time. When this happens, we should update the
 * caffe_classify to pass some identified (or the config) to signify which classifier
 * should be used.
 */
static Classifier *classifier = NULL;


/* Flush a log file pointer to a string to return to user */
static void flush_log(char* out, FILE* log) {
  if (log == NULL) 
    return;

  fflush(log);
  long bytes = ftell(log);
  rewind(log);
  fread(out, 1, bytes, log);
  out[bytes] = '\0';
}


/**
 * 
 */
extern CaffeReturn caffe_init(CaffeConfig config) {
  CaffeReturn ret;

  // Write all logs to this memory buffer, which is then converted to a char* and returned
  // as a part of the CaffeReturn struct. Note all the functions in this interface use
  // this pattern
  FILE* caffe_log = fmemopen(NULL, MAX_ERROR_SIZE, "w+");

  try {

    if (!caffe_log) {
      throw "Can't open log output stream";
    }

    // if there is already a loaded instance, cleanup the current and log that, then load
    // the new one  
    if (classifier != NULL) {
      fprintf(caffe_log, "Existing caffe instance, cleaning it up\n");
      CaffeReturn tmp_ret = caffe_cleanup();
      if (tmp_ret.error_code != CAFFE_SUCCESS) {
        throw "Could not teardown existing caffe instance";
      }
    }

    classifier = new Classifier(config.model_file, config.trained_file, config.mean_file, config.label_file, caffe_log);
    flush_log(ret.error_log, caffe_log);
    ret.error_code = CAFFE_SUCCESS;
  } catch (const char* error) {
    flush_log(ret.error_log, caffe_log);
    strcat(ret.error_log, error);
    ret.error_code = CAFFE_ERROR;
  }

  fclose(caffe_log);
  return ret;
}

extern CaffeReturn caffe_classify(const char* image_file) {
  CaffeReturn ret;
  ret.prediction_count = 0;

  FILE* caffe_log = fmemopen(NULL, MAX_ERROR_SIZE, "w+");
  try {
    if (!caffe_log) {
      throw "Can't open log output stream";
    }

    if (!classifier) {
      throw "Need to call caffe_init()";
    }

    cv::Mat img = cv::imread(image_file, -1);
    if (img.empty()) {
      throw "Unable to decode image";  
    }
    
    std::vector<Prediction> predictions = classifier->Classify(img, MAX_CLASSIFICATION_SIZE, caffe_log);
    for (size_t i = 0; i < predictions.size(); i++) {

      // Note because we have a fixed size of classifications due wanting a known struct
      // size, only take as many as we need. Once we have a dynamic array in that struct,
      // fix this to take all of them
      if (i >= MAX_CLASSIFICATIONS) {
        break;
      }

      fprintf(caffe_log, "Running prediction %zu\n", i);
      Prediction p = predictions[i];
      CaffePrediction predict;
      predict.confidence = p.second;
      strcpy(predict.classification, p.first.c_str());
      ret.predictions[i] = predict;
      ret.prediction_count++;
    }
    
    flush_log(ret.error_log, caffe_log);
    ret.error_code = CAFFE_SUCCESS;
  } catch(const char* error) {
    flush_log(ret.error_log, caffe_log);
    strcat(ret.error_log, error);
    ret.error_code = CAFFE_ERROR;
  }
  fclose(caffe_log);
  return ret;
}

extern CaffeReturn caffe_cleanup() {
  CaffeReturn ret;
  FILE* caffe_log = fmemopen(NULL, MAX_ERROR_SIZE, "w+");

  try {
    if (!caffe_log) {
      throw "Can't open log output stream";
    }
    delete classifier;
    flush_log(ret.error_log, caffe_log);
    ret.error_code = CAFFE_SUCCESS;
  } catch (const char* error) {
    flush_log(ret.error_log, caffe_log);
    strcat(ret.error_log, error);
    ret.error_code = CAFFE_ERROR;
  }
  fclose(caffe_log);
  return ret;
}

#else
/*
int main(int argc, char** argv) {
  LOG(FATAL) << "This example requires OpenCV; compile with USE_OPENCV.";
}
*/
#endif  // USE_OPENCV
