package main

import (
	"github.com/whazell/caffe"
	"fmt"
	"os"
	"io/ioutil"
)

func main() {

	// Use the argument given to classify a local file
	if len(os.Args) < 2 {
		fmt.Printf("Usage: ./example <image file>\n")
		os.Exit(1)
	}

	f, err := ioutil.ReadFile(os.Args[1])
	// Initialize the Caffe model. Note the model info (in this case bvlc_alexnet) needs
	// to be downloaded prior to this program being run. If you use the provided
	// Dockerfile, or the docker image, this is already done. However if you are running
	// elsewhere and want to use one of the pre-made Caffe models you would do:
	//
	// /opt/caffe/scripts/download_model_binary.py <model directory>
	// /opt/caffe/data/ilsvrc12/get_ilsvrc_aux.sh
	//
	// If you make your own model, simply provide the needed files
	model, err := caffe.Init(caffe.Config{
		ModelFile: "/opt/caffe/models/bvlc_alexnet/deploy.prototxt",
		TrainedModel: "/opt/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel",
		MeanFile: "/opt/caffe/data/ilsvrc12/imagenet_mean.binaryproto",
		LabelFile: "/opt/caffe/data/ilsvrc12/synset_words.txt",
		StoragePath: "/tmp",
	})

	if err != nil {
		fmt.Printf("%s\n", err.Error())
		os.Exit(1)
	}

	if err != nil {
		fmt.Printf("Could not open file %s: %s\n", os.Args[1], err.Error())
		os.Exit(1)
	}

	predictions, err := model.Classify(f)
	if err != nil {
		fmt.Printf("%s\n", err.Error())
		os.Exit(1)
	}

	fmt.Printf("Got %d predictions...\n", len(predictions))
	for _, p := range predictions {
		fmt.Printf("[%f] - %s\n", p.Confidence, p.Classification)
	}

	// Cleanup 
	err = model.Cleanup()
	if err != nil {
		fmt.Printf("%s\n", err.Error())
		os.Exit(1)
	}
}
