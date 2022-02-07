## Overview

The objective of this project is to produce color images given grayscale input image. This is a difficult task in general as the 
gray scale images are of single channel and color images are of 3 channels i.e RGB. Since we don't have a straightforward mathematical
formula to achieve gray2color conversion, we use Neural Network based models to achieve this.


`grayscale2color.py` is the entry point for training. The dataset consists of around 4000 landscape images. 
These images are first split into training and validation set during training. I used a 
split of 20%. Based on the `model` argument,`Trainer` objects are created and `train` method is invoked for training.

`train.py` contains 3 types of `Trainer` class. `Trainer` trains on data according to the model specified and runs training and validation.
Dataloader is initialized from one of the `colorize_data.py` classes. `colorize_data.py` creates `DataSet` objects for use in the torch dataloader.

### Network Architecture

`basic_network.py` contains 3 types of models: `basic`, `labnet` and `preincep`.

- `basic` network is a simple autoencoder. The encoder consists of first 6 children of Resnet18 which takes in a gray input 
and the decoder consists of an  upsampling network that gives a 3 channel output corresponding to the RGB output. 
- `labnet` is a modification of the `basic` network that predicts a 2 channel output in the LAB space instead of the 3 channel RGB out. Prediction of 2 channel out
is easier and ensures faster convergence during training as the model output space is reduced from estimating 3 channels to 2 channels.
- `preincep` network is a more advanced network that leverages pretrained `Inceptionv3` model. The pretrained inception model acts as a feature
extractor that can be fused with the encoder output in the traditional autoencoder. This is inspired from the work done in 
[Deep Koalarization: Image Colourization using CNNs and Inception-ResNet-v2](https://arxiv.org/abs/1712.03400), by Baldassarre et al.
- `preres` network uses pretrained `inception-resnet-v2` as the feature extractor. This model has been excluded from the results.

### Loss Function and Optimizer
All the 4 networks used here estimate the values for the rgb/lab images. Thus, the loss function used is a regression loss function.
We are trying to reduce the error between the true rgb/lab values to the predicted rgb/lab values. `MSE` and `Huber` loss are suitable 
for this case. Experiments showed that huber loss gave a lower validation error in comparison to mse loss. 
The optimizer used during training is the `Adam Optimizer`.

### Data Preprocessing

I found 3 images in this dataset which were grayscale only and 1 image with 4 channels. These were excluded during training.

`colorize_data.py` contains classes that prepare the color images for training according to the type of model to train.

`ColorizeData`: Prepares data for the `basic` network. The rgb image is transformed to grayscale, resized to 256x256 and
normalized with mean=0.5 and std=0.5. The rbg image is resized to 256x256 and normalized with mean=0.5 and std=0.5 for target image.

`LabColorizeData`: Prepares data for the `labnet` network. The rgb image is transformed to grayscale, resized to 256x256 and
normalized with mean=0.5 and std=0.5. The rgb image is also converted to the lab space which gives the target image for training. 

`PreIncColorizeData`: Prepares data for the `preincep` network. 
The rgb image is resized to 299x299 and normalized with mean=0.5 and std=0.5 for the inception network.
The rgb image is also resized to 224x224 for the encoder network. 
The rgb image is also converted to the lab space which gives the target image for training.

### Inference
`inference.py` contains different functions to run inference for `basic`, `labnet` and `preincep` networks. In case of `labnet`
and `preincep` networks, the predicted image is in the lab space. These are stacked with the input gray images to generate the 
`rgb` output. Each of the channels are rescaled appropriately here. If rgb images are given as input, they are converted to gray
and fed into the model. 

## Training Results
|Model|Epochs|Learning rate|Batch Size|Loss Fn|Validation error|
|---|---|---|---|---|---|
|basic|30|0.01|64|MSE|0.0566|
|basic|30|0.01|64|Huber|0.0268|
|labnet|30|0.01|64|MSE|0.0093|
|labnet|30|0.01|64|Huber|0.0045|
|preres|30|0.001|64|MSE|0.0034|
|preres|30|0.001|64|Huber|0.0027|
|preincep|30|0.001|32|Huber|0.0012|

## Observations 
- Even though the validation error for the `preincep` network was considerably less, these images still have a color tint to it.
This can be attributed to the MSE type loss function.This loss function is slightly problematic for colorization due to the multi-modality of the problem. 
For example, a grayish image could be red or blue, and if our model picks the wrong color, it will be harshly penalized.
As a result, our model will usually choose desaturated colors that are less likely to be "very wrong" than bright, vibrant colors.
One way to achieve this is using a classification loss function similar to the work in [Colorful Image Colorization
](https://arxiv.org/abs/1603.08511) by Zhang et.al


- GAN type networks are definitely a must-try networks for this task. Here two losses are used: L1 loss, which makes it a 
regression task, and an adversarial (GAN) loss, which helps to solve the problem in an unsupervised manner.
In a GAN we have a generator and a discriminator
model which learn to solve a problem together. In this setting,
the generator model takes a grayscale image (1-channel image) and produces a 2-channel image, a channel for *a and another for *b. 
The discriminator, takes these two produced channels and concatenates them with the input grayscale image and decides whether
this new 3-channel image is fake or real. The discriminator also needs to see some real images
(3-channel images again in Lab color space)that are not produced by the generator and should learn that they are real.


- During inference for `basic` network, the image was not resized to 256x256. This gave good resolution outputs. For the case
of `labnet` and `preincep` images were resized. I attribute the lack of quality in the generated images for these 2
networks to this.

## Usage
### General Environment Setup
Create a conda environment with pytorch, cuda. 

`$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

Install scikit-image for image processing and matplotlib for plotting the images

`$ pip install scikit-image matplotlib scikit-learn pandas tqdm`

Download the training dataset as `landscape_images/` in the project working directory.

### Training
To run training, run `grayscale2color.py` with the following arguments:

`model`: Specify the model to run. Available options are `basic`, `labnet`, `preincep`

`bs`: Specify the batch size for training

`epochs`: Number of epochs to train

`lr`: Learning rate for training

`loss`: Specify the loss funtion to use. Available options are `mse` and `huber`

Example usage: `python grayscale2color.py --model basic --bs 8 --lr 1e-2 --epochs 1 --loss huber`

### Inference
The `inference.py` script generates the color image given a grayscale or color image. If the input type is a color image, the image is first 
converted into gray and fed to the model.

The arguments for this script are as below:

`input`: The input image file location. The generated images are also stored in this location.

`model_pt`: Path to the trained model

`model`: Type of the trained model. Available choices are the same as that of the training script- `basic`, `labnet`, `preincep`

`image_type`: Type of image. Available choices are `gray` and `rgb`

Example usage: `$ python inference.py --input dir/file_loc.jpg --image_type gray --model_pt basic_colorizer.model --model basic`

You can also edit the `generate_color.sh` bash script run the above python script.
