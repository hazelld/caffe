FROM bvlc/caffe:cpu

RUN apt update && apt install -y vim curl && \
    curl https://dl.google.com/go/go1.14.2.linux-amd64.tar.gz -o /tmp/go1.14.2.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf /tmp/go1.14.2.linux-amd64.tar.gz && \
    rm -rf /tmp/*
ENV PATH=$PATH:/usr/local/go/bin

# Run these in 2 separate commands so the large model binary can be cached by docker
RUN /opt/caffe/scripts/download_model_binary.py /opt/caffe/models/bvlc_alexnet
RUN /opt/caffe/data/ilsvrc12/get_ilsvrc_aux.sh


