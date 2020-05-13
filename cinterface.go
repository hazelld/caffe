package caffe


// #cgo CXXFLAGS: -std=c++11
// #cgo CPPFLAGS: -DCPU_ONLY=1 -DUSE_OPENCV=1
// #include "go-caffe-interface.h"
import "C"

import (
	"unsafe"
)


type caffeReturn struct {
	ErrorCode int
	ErrorLog string
	Predictions []Prediction
	PredictionCount int
}

// These constants correspond with the returns codes defined in go-caffe-interface.h
const (
	caffeError = -1
	caffeSuccess = 0
)



// caffeInit is a slim wrapper for calling the C function caffe_init() and returning the
// CaffeReturn that it returns. Note when using this go library, use the Init() function
// which is a higher level abstraction.
func caffeInit(conf Config) caffeReturn {
	c := C.struct_CaffeConfig{
		model_file: C.CString(conf.ModelFile),
		trained_file: C.CString(conf.TrainedModel),
		mean_file: C.CString(conf.MeanFile),
		label_file: C.CString(conf.LabelFile),
	}

	// Call caffe_init, then map the response to the Go structure, ignoring the
	// predictions field, since it won't be set in the init function
	ret := C.caffe_init(c)
	return caffeReturn{
		ErrorCode: int(ret.error_code),
		ErrorLog: C.GoString(&(ret.error_log[0])),
	}
}

// caffeCleanup is a slim wrapper for calling the C function caffe_cleanup()
func caffeCleanup() caffeReturn {
	ret := C.caffe_cleanup()
	return caffeReturn{
		ErrorCode: int(ret.error_code),
		ErrorLog: C.GoString(&(ret.error_log[0])),
	}
}


func caffeClassify(imagePath string) caffeReturn {
	ret := C.caffe_classify(C.CString(imagePath))

	// Need to extract the prediction array from the return struct, then create a Go
	// Prediction struct for each one. Note this data won't be touched by the Go garbage
	// collector
	var cArr *C.CaffePrediction = &ret.predictions[0]
	length := int(ret.prediction_count)
	slice := (*[1 << 28]C.struct_CaffePrediction)(unsafe.Pointer(cArr))[:length:length]

	var predictions []Prediction
	for i, _ := range slice {
		p := Prediction{
			Confidence: float64(slice[i].confidence),
			Classification: C.GoString(&(slice[i].classification[0])),
		}
		predictions = append(predictions, p)
	}
	return caffeReturn{
		ErrorCode: caffeSuccess,
		ErrorLog: C.GoString(&(ret.error_log[0])),
		Predictions: predictions,
		PredictionCount: length,
	}
}

	//var cArr *C.CaffePrediction = ret.predictions
	//length := int(ret.prediction_count) 
