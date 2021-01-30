## Introduction
This implementation of CMSIS-NN solved the painfully slow memory copy problem in normal convolution layers and especially depthwise convolution layers. This optimization achieves 2X inference speed for DS-CNN models running on cortex m4 processors.\
Please replace the functions in the original CMSIS-NN model and please notice the changes made to the parameters.\
More is on the way. A fully automatic Tensorflow to cortex-m4 toolchain for DS-CNN models will be released if my company allows me to do so (apparently not).
## Notice
* Due to different rounding methods, the fucntions provided here may not yield the exact results as the original ones. (e.g. 0xCD and 0xCC)
