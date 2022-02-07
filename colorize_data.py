from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np
import os
from typing import Tuple
from torchvision.transforms.functional import resize
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from PIL import Image

ENCODER_SIZE = 224
INCEPTION_SIZE = 299

def color2lab(img):
    img = np.asarray(img)
    img_lab = rgb2lab(img)
    img_lab = (img_lab + 128) / 255

    img_ab = img_lab[:, :, 1:3]
    img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()
    return img_ab

class ColorizeData(Dataset):
    def __init__(self, df):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.df = df
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256, 256)),
                                          T.Grayscale(),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256, 256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # self.clean_dataset()

    def clean_dataset(self):
        for idx, row in self.df.iterrows():
            file = row['file_name']
            if 'jpg' not in file:
                print("{} is a invalid file".format(file))
                self.df.drop(idx, inplace=True)
                continue
            img = Image.open(file)
            if T.ToTensor()(img).shape[0] != 3:
                print("{} is a GrayScale Image".format(file))
                self.df.drop(idx, inplace=True)

    def __len__(self) -> int:
        # return Length of dataset
        return len(self.df)
        pass

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        file = self.df.iloc[index]['file_name']
        img = Image.open(file)
        ip_trans = self.input_transform(img)
        targ_trans = self.target_transform(img)
        return (ip_trans, targ_trans)


class LabColorizeData(ColorizeData):
    """
    Transforms RGB to LAB for LAB network.
    Images are scaled to 256*256 and normalized
    RGB image is converted to gray scale and AB channels

    """
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        file = self.df.iloc[index]['file_name']
        img = Image.open(file)
        img_gray = self.input_transform(img)
        img_original = self.target_transform(img).permute(1, 2, 0)
        img_ab = color2lab(img_original)
        return (img_gray, img_ab)
        pass

"""
class PreResColorizeData(ColorizeData):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        file = self.df.iloc[index]['file_name']
        img = Image.open(file)
        img_inception = tf_img(self.inception_transform(load_img(file)))
        img_original = self.encoder_transform(img)
        img_ab = color2lab(img_original)
        img_gray = rgb2gray(img_original)
        img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

        return (img_gray, img_ab, img_inception)
"""


class PreIncColorizeData(ColorizeData):
    """
    RGB image is transformed to inception image of size 299x299 and normalized.
    RGB image is resized to 224*224
    RGB image is converted to gray scale and AB channels
    """
    def __init__(self, df):
        super(ColorizeData, self).__init__()
        self.df = df
        self.inception_transform = T.Compose([T.ToTensor(),
                                              T.Resize(size=(INCEPTION_SIZE, INCEPTION_SIZE)),
                                              T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.encoder_transform = T.Compose([T.ToTensor(),
                                            T.Resize(size=(ENCODER_SIZE, ENCODER_SIZE)),
                                            ])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        file = self.df.iloc[index]['file_name']
        img = Image.open(file)
        img_inception = self.inception_transform(img)
        img_original = self.encoder_transform(img)
        img_ab = color2lab(img_original.permute(1,2,0))
        img_gray = rgb2gray(img_original.permute(1,2,0))
        img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()

        return (img_gray, img_ab, img_inception)