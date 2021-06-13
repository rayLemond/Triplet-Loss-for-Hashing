## Triplet Loss Hashing
Triplet Loss is one basic metric learning algo. Hashing is closely related to metric learning. It seems there isn't much public and solid codes for Triplet Hashing. So i've implemented a version of triplet hashing that can achieve acceptable performance.

## Environment
Pytorch=1.2.0  torchvision=0.4.0  cudatoolkit=10.0.130  NvidiaDriverVersion=455.45.01  CUDA=11.1

## How to use
1. Define the path of image data in parser (line 194).
2. Prepare the txt files for image index (line 61 - line 65). Format: filename + class index (start from 0)
3. Python TripletHashing.py to run.

## Results
mAP on CUB-200-2011 dataset

| -| - | 64 bits |
| :----:| :----: | :----: |
| - | - | 77.18 |
