# copied from https://github.com/chainer/chainer/blob/v6.7.0/docker/python3/Dockerfile
# updated for CUDA v11 and Ubuntu 20.04

# FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
# FROM dikorsch/cupy-cuda101-opencv4.1.1
FROM dikorsch/cupy-cuda110

# get the user and group. if not defined, fallback to root
ARG UID=root
ARG GID=root

# ENV SVM_TRAINING="01_svm_training"
# ENV ESTIMATOR_FOLDER="02_cs_parts_estimation"
# ENV EXTRACTOR_FOLDER="03_feature_extraction"
# ENV PIP="pip3"

# install vlfeat
WORKDIR /code
ADD http://www.vlfeat.org/download/vlfeat-0.9.21-bin.tar.gz vlfeat-0.9.21-bin.tar.gz
RUN tar xzf vlfeat-0.9.21-bin.tar.gz && \
	cp -v vlfeat-0.9.21/bin/glnxa64/*.so /usr/lib/x86_64-linux-gnu/ && \
	mkdir /usr/include/vl && \
	cp -v vlfeat-0.9.21/vl/*.h /usr/include/vl && \
	rm -r vlfeat-0.9.21 vlfeat-0.9.21-bin.tar.gz


# clone my code
# RUN git clone https://git.inf-cv.uni-jena.de/Fine-grained/feature_extraction.git ${EXTRACTOR_FOLDER}
# # RUN cd ${EXTRACTOR_FOLDER}; git checkout gcpr2019submission -b gcpr2019submission; cd ..

# RUN git clone https://git.inf-cv.uni-jena.de/Fine-grained/cs_parts_estimation.git ${ESTIMATOR_FOLDER}
# # RUN cd ${ESTIMATOR_FOLDER}; git checkout gcpr2019submission -b gcpr2019submission; cd ..

# RUN git clone https://git.inf-cv.uni-jena.de/Fine-grained/svm_training.git ${SVM_TRAINING}
# # RUN cd ${SVM_TRAINING}; git checkout gcpr2019submission -b gcpr2019submission; cd ..

RUN apt update && apt install -y gfortran libopenblas-dev liblapack-dev

COPY requirements.txt /code
RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

RUN mkdir /data
RUN chown -R ${UID}:${GID} /code

USER      ${UID}:${GID}
