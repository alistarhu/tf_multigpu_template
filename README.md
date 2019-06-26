# tf_multigpu_template
tf_multigpu_cls_template
A basic sample of tensorflow multi-gpu project for classification using mobilenet-V2. This script is test on  with 

## Contents: 
- datasets: load training data use `tf.data.Dataset`
- netdefs: define your own network
- scripts: contain train test and other utils
- configuration: set train and test params

## Test environment
- Ubuntu 16.04
- Python3.6
- tensorflow-gpu1.13.1
- cuda-10.0
- 8x RTX-2080Ti

## Get start
run training:
```
./run_train.sh
```  

run tensorboard:
```
./run_tensorboard.sh
``` 

run test:
```
./run_test.sh
``` 
