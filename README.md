## Introdcution

Adaptive multi-occlusion face recognition (including face with mask, glasses and hat) model based on CLIP([openai/CLIP: CLIP (Contrastive Language-Image Pretraining), Predict the most relevant text snippet given an image (github.com)](https://github.com/openai/CLIP)), based on FaceX-Zoo([JDAI-CV/FaceX-Zoo: A PyTorch Toolbox for Face Recognition (github.com)](https://github.com/JDAI-CV/FaceX-Zoo)) framework.

## Requirements

* python >= 3.7.1

* pytorch >= 1.1.0

* torchvision >= 0.3.0

See the detail requirements in [requirements.txt](./requirements.txt)

## Model Training

run [./train.sh](./training_mode/conventional_training/train.sh) with the train dataset CASIA-Webface ([baidu Pan Link](https://pan.baidu.com/s/1mSbJ61BWEqPqv6RZkqv7CQ?pwd=877a))

## Model Test

run [./test_lfw_adapt.sh](./test_protocol/test_lfw_adapt.sh) with the test dataset LFW-MASK ([baidu Pan Link](https://pan.baidu.com/s/1bVmH67D1SWpgv2Fb3rg66A?pwd=p50q)), or the dataset MEGLSS([baidu Pan Link](https://pan.baidu.com/s/1r_7O0GxDkEMNkb4Kvty_9A?pwd=wg1m)), CALFW-SUNGLASSES([baidu Pan Link](https://pan.baidu.com/s/1maGEKvjG1kbcusEsek1hRA?pwd=f0py)), CPLFW([baidu Pan Link](https://pan.baidu.com/s/1gJ8659xUhG-gcOZ4fMS6XA?pwd=6wmo))

## Simulate occlusion generation
You can generate a new simulated occluded dataset by running the python file [mask_data_generation.py](./data_processor/mask_data_generation.py) and changing the dataset path and occlusion image in the file.
