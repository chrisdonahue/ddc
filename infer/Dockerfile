FROM ubuntu:16.04

RUN apt-get update

# Install dependencies
RUN apt-get install -y \
  python-pip \
  wget

# Install essentia dependencies
RUN apt-get install -y \
  build-essential \
  libyaml-dev \
  libfftw3-dev \
  libavcodec-dev \
  libavformat-dev \
  libavutil-dev \
  libavresample-dev \
  python-dev \
  libsamplerate0-dev \
  libtag1-dev

RUN pip install --no-cache-dir --upgrade pip

# Download and install Essentia
WORKDIR /essentia
RUN wget https://github.com/MTG/essentia/archive/v2.1_beta3.tar.gz
RUN tar xvfz v2.1_beta3.tar.gz
RUN pip install --no-cache-dir numpy==1.11.2
RUN cd essentia-2.1_beta3 && ./waf configure --build-static --with-python && ./waf && ./waf install

# Install Python packages
RUN pip install --no-cache-dir tensorflow==0.12.1
RUN pip install \
  scipy==0.18.1 \
  Flask==1.1.2

# Download and unzip Dance Dance Convolution
WORKDIR /ddc
COPY *.py ./
COPY server_aux server_aux/
COPY frontend frontend/

ENTRYPOINT ["python", "ddc_server.py"]
