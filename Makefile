LIBS=caffe glog boost_system opencv_highgui opencv_imgproc opencv_core
INCLUDE=/opt/caffe/build/include /opt/caffe/include /opt/caffe/build/lib
LIBDIRS=/opt/caffe/build/lib $(shell pwd)
CPPFLAGS=-DCPU_ONLY=1 -DUSE_OPENCV=1 -fPIC

LDFLAGS=$(addprefix -l, $(LIBS))
LDFLAGS += $(addprefix -L, $(LIBDIRS))
CPPFLAGS += $(addprefix -I, $(INCLUDE))


all: build-library build-go-example

build-library: libclassification.so
	g++ classification.cpp $(LDFLAGS) $(CPPFLAGS) -shared -o $<


build-go-example: example/example example/main.go
	export CGO_CPPFLAGS="$(CPPFLAGS)" && export CGO_LDFLAGS="$(LDFLAGS) -Wl,-rpath=$(shell pwd)" && \
	go build -o example ./example

clean:
	rm -rf *.o example/example *.so


