# This is probably the better container to source from but I couldn't figure out how to link CudNN
#FROM nvidia/cuda:8.0-cudnn5-runtime
FROM tensorflow/tensorflow:0.12.1-gpu

# Had to add this unofficial PPA to install ffmpeg on Ubuntu 14.04 :/
RUN add-apt-repository ppa:heyarje/libav-11

RUN apt-get update

# Install Essentia dependencies
RUN apt-get install -y build-essential libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev libtag1-dev libchromaprint-dev python-six libav-tools libavcodec-extra-56

# Install Python dependencies
RUN apt-get install -y python-dev python-numpy-dev python-numpy python-yaml
RUN apt-get install -y python-pip wget

# Download and install Essentia
RUN wget https://github.com/MTG/essentia/archive/v2.1_beta3.tar.gz
RUN tar xvfz v2.1_beta3.tar.gz
RUN cd essentia-2.1_beta3 && ./waf configure --build-static --with-python && ./waf && ./waf install

# Install Python packages
#RUN pip install tensorflow-gpu==0.12.1
RUN pip install tqdm
RUN pip install scipy

# Download and unzip Dance Dance Convolution
RUN wget https://github.com/chrisdonahue/ddc/archive/v1.0.tar.gz
RUN tar xvfz v1.0.tar.gz
RUN sed -i 's/localhost/0.0.0.0/g' ddc-1.0/infer/ddc_server.py
