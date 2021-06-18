## Triplet Loss Hashing
Triplet Loss is one basic metric learning algo. Hashing is closely related to metric learning. It seems there isn't much public and solid codes for Triplet Hashing. Here is my version of triplet hashing that can achieve acceptable performance.

## Environment
Pytorch=1.2.0  torchvision=0.4.0  cudatoolkit=10.0.130  NvidiaDriverVersion=455.45.01  CUDA=11.1

## How to use
1. Define the path of image data in parser (line 194).
2. Prepare the txt files for image index (line 61 - line 65). Format: filename + class index (start from 0)
3. Python TripletHashing.py to run.

## Details
Distance: Euclidean without sqrt  
Margin: 0.5 bits  
Backbone: CNN (Resnet50) + tanh  
Loss: Triplet Loss + Quantization Loss  
Optimizer: Adam lr = 5e-5

## Results
mAP on CUB-200-2011 dataset. 

| 16 bits | 32 bits | 48 bits | 64 bits |
| :----: | :----: | :----: | :----: |
| 74.54 | 77.03 | 76.92 | 77.18 |  

Welcome to discuss any better implementation.
