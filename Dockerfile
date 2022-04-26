FROM cwaffles/openpose

WORKDIR /openpose/build/python/openpose
#RUN apt-add-repository multiverse
#RUN apt update
#RUN apt install nvidia-modprobe -y
RUN python3 -m pip install --upgrade pip setuptools wheel
RUN make install
RUN pip3 install redis numpy Pillow pandas pykalman tensorflow==1.13.1 numba scikit-learn==0.22.2
RUN cp ./pyopenpose.cpython-36m-x86_64-linux-gnu.so /usr/local/lib/python3.6/dist-packages
RUN cd /usr/local/lib/python3.6/dist-packages
RUN ln -s pyopenpose.cpython-36m-x86_64-linux-gnu.so pyopenpose
RUN export LD_LIBRARY_PATH=/openpose/build/python/openpose
WORKDIR /openpose/models
RUN bash getModels.sh
WORKDIR /openpose/examples/tutorial_api_python
COPY . ./
VOLUME [ "/data" ]
CMD ["python3", "hba_main.py" ]
