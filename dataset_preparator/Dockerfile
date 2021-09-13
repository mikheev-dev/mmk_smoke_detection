FROM tensorflow/tensorflow:latest-gpu-jupyter
COPY . /app/
RUN pip install -r /app/requirements.txt
RUN pip install --upgrade jupyter_http_over_ws>=0.0.7
RUN jupyter serverextension enable --py jupyter_http_over_ws
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    doxygen \
    file \
    gfortran \
    gnupg \
    gstreamer1.0-plugins-good \
    imagemagick \
    libatk-adaptor \
    libatlas-base-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libboost-all-dev \
    libcanberra-gtk-module \
    libdc1394-22-dev \
    libeigen3-dev \
    libfaac-dev \
    libfreetype6-dev \
    libgflags-dev \
    libglew-dev \
    libglu1-mesa \
    libglu1-mesa-dev \
    libgoogle-glog-dev \
    libgphoto2-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-bad1.0-0 \
    libgstreamer-plugins-base1.0-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libhdf5-dev \
    libhdf5-serial-dev \
    libjpeg-dev \
    liblapack-dev \
    libmp3lame-dev \
    libopenblas-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libopenjp2-7-dev \
    libopenjp2-tools \
    libpng-dev \
    libpostproc-dev \
    libprotobuf-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libtheora-dev \
    libtiff5-dev \
    libv4l-dev \
    libvorbis-dev \
    libx264-dev \
    libxi-dev \
    libxine2-dev \
    libxmu-dev \
    libxvidcore-dev \
    libzmq3-dev \
    v4l-utils \
    x11-apps \
    x264 \
    yasm \
  && apt-get clean
RUN apt-get install -y protobuf-compiler
RUN cd /app/models/research/ && protoc object_detection/protos/*.proto --python_out=. && cp object_detection/packages/tf2/setup.py . && python -m pip install .
WORKDIR /app
RUN mkdir /dataset
CMD python ./main.py
#CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --NotebookApp.allow_origin=https://colab.research.google.com --ip 0.0.0.0 --no-browser --allow-root --NotebookApp.token= --NotebookApp.password= --NotebookApp.port_retries=0"]