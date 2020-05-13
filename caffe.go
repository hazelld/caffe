package caffe

import (
	"crypto/md5"
	"os"
	"io/ioutil"
	"errors"
	"fmt"
)

type Prediction struct {
	Confidence float64
	Classification string
}

type Config struct {
	ModelFile string
	TrainedModel string
	MeanFile string
	LabelFile string

	// 
	StoragePath string
}

// Eventually this library will return a struct called Model that will correspond with an
// underlying model that has been initialized on the C side with a given config. This
// could be combined with the Factory pattern to refine the Init interface. Once this is
// used, the Classify() function will be a method on the model.
//
type Model struct {
	Config Config
}

// Init will initialize the underlying Caffe model. Currently it can only initialize a
// single instance with a single model, but in the future maybe the library can initialize
// multiple models handled at the C layer.
func Init(conf Config) (Model, error) {

	// Set some defaults
	if conf.StoragePath == "" {
		conf.StoragePath = "/tmp"
	}

	ret := caffeInit(conf)
	if ret.ErrorCode == caffeError {
		return Model{}, errors.New("Could not initialize: " + ret.ErrorLog)
	}
	return Model{conf}, nil
}

// Classify will take an image and run it through the Caffe model to get a set of
// classifications. 
func (m *Model) Classify(image []byte) ([]Prediction, error) {
	
	// Store the image into a temporary file based on the given StoragePath
	hash := fmt.Sprintf("%x", md5.Sum(image))
	file := m.Config.StoragePath + "/" + hash
	err := ioutil.WriteFile(file, image, 0644)
	if err != nil {
		return nil, err 
	}
	
	ret := caffeClassify(file)

	// Remove the stored image regardless of what happened, note if we fail to remove we
	// don't want to immidiately return since we could lose the predictions
	os.Remove(file)

	// Verify the result of the classification
	if ret.ErrorCode == caffeError {
		return nil, errors.New("Could not classify: " + ret.ErrorLog)
	}
	return ret.Predictions, nil
}


// Cleanup will shutdown the Caffe model. Since we have only one loaded model at a time,
// this will just be a function, but again in the future convert this to a method on the
// Model type
func (m *Model) Cleanup() error {
	ret := caffeCleanup()
	if ret.ErrorCode == caffeError {
		return errors.New("Could not cleanup: " + ret.ErrorLog)
	}
	return nil
}
