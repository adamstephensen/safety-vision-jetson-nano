# safety vision on jetson nano

To execute

``` python

cd python
python3 predict.py ../hardhat.jpg
```

## monitor memory

```
watch -n 1 nvidia-smi
sudo tegrastats
```

## useful links

- used this https://packages.ubuntu.com/bionic/python3-pil

- used this https://www.jetsonhacks.com/2019/04/14/jetson-nano-use-more-memory/

- used this https://devtalk.nvidia.com/default/topic/1048776/official-tensorflow-for-jetson-nano-/

- used this running a juptyer notebook https://medium.com/@karol_majek/10-simple-steps-to-tensorflow-object-detection-api-aa2e9b96dc94
		- skipped down to #5

```
#5 download tensorflow repository
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git clone https://github.com/tensorflow/models

#6 install object detection api - changed to use pip3. already had python-pil
sudo apt-get install protobuf-compiler python-lxml python-tk
pip3 install --user Cython
pip3 install --user contextlib2
pip3 install --user jupyter

sudo apt-get install python3-matplotlib # instead of pip3 install --user matplotlib

i also triee
apt-get install -y python3-setuptools
apt-get install -y python3-setuptools
```		



- configuring  tensorflow object detection
https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

## jetson tensorflow

https://www.dlology.com/blog/how-to-run-tensorflow-object-detection-model-on-jetson-nano/

```

%cd /content
git clone https://github.com/tensorflow/models.git

apt-get install protobuf-compiler python-pil python-lxml python-tk

pip3 install Cython
pip3 install contextlib2
pip3 install pillow
sudo pip3 install lxml # prob already installed
sudo apt-get install python3-matplotlib

pip install pycocotools

%cd /content/models/research
!protoc object_detection/protos/*.proto --python_out=.

import os
import sys
os.environ['PYTHONPATH'] += ':/content/models/research/:/content/models/research/slim/'
sys.path.append("/content/models/research/slim/")

!python object_detection/builders/model_builder_test.py


```
