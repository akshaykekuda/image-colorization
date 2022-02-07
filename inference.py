import torch
import torchvision.transforms as T
from torchvision.utils import save_image
import argparse
import matplotlib.pyplot as plt
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from PIL import Image
from basic_model import PreInceptionNet, PreResNet
import numpy as np

"""
Functions to generate color images are defined here.
Input images can be of gray scale or rgb. In case of rgb, the image is converted to gray and then fed to the model
The model to use can be the basic network, lab based network, or the pretrained inception based network
"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ENCODER_SIZE = 224
INCEPTION_SIZE = 299

# Transforms for inception based network
inception_transform = T.Compose([T.Resize(size=(INCEPTION_SIZE, INCEPTION_SIZE))])
encoder_transform = T.Compose([T.Resize(size=(ENCODER_SIZE, ENCODER_SIZE))])

# Transforms for input gray/color images
input_rgb2gray_transform = T.Compose([T.ToTensor(),
                                 T.Grayscale(),
                             ])

input_rgb2gray_transform_resized = T.Compose([T.ToTensor(),
                                 T.Grayscale(),
                             T.Resize(size=(256, 256)),
                             ])

input_resize = T.Compose([T.ToTensor(),
                          T.Resize(size=(256, 256)),
                          ])

def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='inference.py')

    # General system running and configuration options
    parser.add_argument('--input', type=str, help='location of the image')
    parser.add_argument('--model_pt', type=str, help='path to the trained model')
    parser.add_argument('--model', type=str, default='basic', help='model type to run')
    parser.add_argument('--image_type', type=str, default='gray', help='input image type')

    args = parser.parse_args()
    return args


def stack_lab_channels(grayscale_input, ab_input):
    """
    Stacks the L and AB channels together to create the LAB image. Then, the LAB
    image is converted to RGB.

    Parameters:
      grayscale_input: The L channel as a tensor.
      ab_input: The AB channels as a tensor.

    Returns:
      The RGB channels as a numpy array.
    """
    color_image = torch.cat((grayscale_input, ab_input), axis=0).numpy()
    color_image = color_image.transpose((1, 2, 0))

    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128

    color_image = lab2rgb(color_image.astype(np.float64))

    return color_image


def generate_gray2color(file, model_pt):
    """
    Generates color image from gray scale images using the basic network
    The predicted color image is stored as well as displayed

    Parameters:
      file: Gray scale image to convert
      model_pt: Path to the model to use

    """
    print("loading model")
    model = torch.load(model_pt, map_location=device)
    model.eval()
    img = Image.open(file)
    try:
        input = T.ToTensor()(img).unsqueeze(0).to(device)
    except:
        raise TypeError("Image not of grayscale type")
    with torch.no_grad():
        output = model(input)
    output = output.cpu()
    input = input.cpu()
    save_gen_image(output, file)
    display_images(input.squeeze(0).permute(1, 2, 0), output.squeeze(0).permute(1, 2, 0))


def generate_color2color(color_file, model_pt):
    """
    Generates color image from color images using the basic network. The input RCG image is first converted to gray scale.
    The predicted color image is stored as well as displayed

    Parameters:
      file: Gray scale image to convert
      model_pt: Path to the model to use

    """
    print("loading model")
    model = torch.load(model_pt, map_location=device)
    model.eval()
    img = Image.open(color_file)
    try:
        input = input_rgb2gray_transform(img).unsqueeze(0).to(device)
    except:
        raise TypeError("Image not of rgb type")
    print("running model inference")
    with torch.no_grad():
        output = model(input)
    output = output.cpu()
    input = input.cpu()
    save_gen_image(output, color_file)
    display_images(input.squeeze(0).permute(1, 2, 0), output.squeeze(0).permute(1, 2, 0))

def generate_gray2color_lab(file, model_pt):
    """
    Generates color image from gray scale images using the basic LAB based network.
    The predicted color image is stored as well as displayed

    Parameters:
      file: Gray scale image to convert
      model_pt: Path to the model to use

    """
    print("loading the file and model")
    img = Image.open(file)
    img_gray = input_resize(img).unsqueeze(0).to(device)
    model = torch.load(model_pt, map_location=device)
    model.eval()
    with torch.no_grad():
        output = model(img_gray)
    img_gray = img_gray.squeeze(0)
    output = output.squeeze(0)
    output = output.cpu()
    img_gray = img_gray.cpu()
    predicted_image = stack_lab_channels(img_gray, output)
    grayscale = img_gray.permute(1,2,0).numpy()
    save_gen_image(torch.from_numpy(predicted_image).permute(2,0,1), file)
    display_images(grayscale, predicted_image)

def generate_color2color_lab(file, model_pt):
    """
    Generates color image from color images using the basic LAB based network.
    The input RCG image is first converted to gray scale.
    The predicted color image is stored as well as displayed

    Parameters:
      file: Gray scale image to convert
      model_pt: Path to the model to use

    """
    print("loading the file and model")
    img = Image.open(file)
    img_gray = input_rgb2gray_transform_resized(img).unsqueeze(0).to(device)
    model = torch.load(model_pt, map_location=device)
    model.eval()
    with torch.no_grad():
        output = model(img_gray)
    img_gray = img_gray.squeeze(0)
    output = output.squeeze(0)
    output = output.cpu()
    img_gray = img_gray.cpu()
    predicted_image = stack_lab_channels(img_gray, output)
    grayscale = img_gray.permute(1,2,0).numpy()
    save_gen_image(torch.from_numpy(predicted_image).permute(2,0,1), file)
    display_images(grayscale, predicted_image)


def generate_color2color_preincep(file, model_pt):
    """
    Generates color image from gray scale images using the basic LAB based network.
    The predicted color image is stored as well as displayed

    Parameters:
      file: Gray scale image to convert
      model_pt: Path to the model to use

    """
    print("loading the file and model")
    img = Image.open(file)
    img_original = encoder_transform(img)
    img_gray = rgb2gray(img_original)
    img_gray = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).float()
    img_inception = T.ToTensor()(inception_transform(img)).unsqueeze(0)
    img_gray, img_inception = img_gray.to(device), img_inception.to(device)
    model = PreInceptionNet(pretrained=True).to(device)
    checkpoint = torch.load(model_pt, map_location=device)
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.eval()
    with torch.no_grad():
        output = model(img_gray, img_inception)
    img_gray = img_gray.squeeze(0)
    output = output.squeeze(0)
    output = output.cpu()
    img_gray = img_gray.cpu()
    predicted_image = stack_lab_channels(img_gray, output)
    grayscale = img_gray.permute(1, 2, 0)
    save_gen_image(torch.from_numpy(predicted_image).permute(2, 0, 1), file)
    display_images(grayscale, predicted_image)


def generate_gray2color_preincep(file, model_pt):
    """
    Generates color image from color images using the pretrained Inceptionv3 based network.
    The input RCG image is first converted to gray scale.
    The predicted color image is stored as well as displayed

    Parameters:
      file: Gray scale image to convert
      model_pt: Path to the model to use

    """
    print("loading the file and model")
    img = Image.open(file)
    img_original = encoder_transform(img)
    img_gray = T.ToTensor()(img_original).unsqueeze(0)
    img_inception = T.ToTensor()(inception_transform(img)).repeat(3, 1, 1).unsqueeze(0)
    img_gray, img_inception = img_gray.to(device), img_inception.to(device)
    model = PreInceptionNet(pretrained=True).to(device)
    checkpoint = torch.load(model_pt, map_location=device)
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.eval()
    with torch.no_grad():
        output = model(img_gray, img_inception)
    img_gray = img_gray.squeeze(0)
    output = output.squeeze(0)
    output = output.cpu()
    img_gray = img_gray.cpu()
    predicted_image = stack_lab_channels(img_gray, output)
    grayscale = img_gray.permute(1, 2, 0)
    save_gen_image(torch.from_numpy(predicted_image).permute(2, 0, 1), file)
    display_images(grayscale, predicted_image)

def save_gen_image(img, file):
    """
    Saves the img at location file

    Parameters:
      img: Image to save
      file: Image file name

    """
    print("saving the generated image in the dir of input image")
    f = file.split('.')[0]
    f += '_gen.jpg'
    save_image(img, f)

def display_images(gray, color):
    """
    Display the Model Input and Output

    Parameters:
      file: Gray scale image to convert
      model_pt: Path to the model to use

    """
    f, axarr = plt.subplots(1, 2, figsize=(20, 10), dpi=200)
    axarr[0].imshow(gray, cmap="gray")
    axarr[0].set_title("Grayscale Image (Model Input)")
    axarr[1].imshow(color)
    axarr[1].set_title("RGB Image (Model Output)")
    plt.show()

"""

def generate_gray2color_preres(file, model_pt):
    img = Image.open(file)
    img_inception = T.ToTensor()(inception_transform(img)).repeat(3,1,1).unsqueeze(0)
    img_original = encoder_transform(img)
    img_gray = T.ToTensor()(img_original).unsqueeze(0)
    img_gray, img_inception = img_gray.to(device), img_inception.to(device)
    model = torch.load(model_pt, map_location='cpu').to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_gray, img_inception)
    img_gray = img_gray.squeeze(0)
    output = output.squeeze(0)
    predicted_image = stack_lab_channels(img_gray, output)
    grayscale = img_gray.permute(1,2,0)
    save_gen_image(torch.from_numpy(predicted_image).permute(2,0,1), file)
    display_images(grayscale, predicted_image)

def generate_color2color_preres(file, model_pt):
    img = Image.open(file)
    img_original = encoder_transform(img)
    img_gray = rgb2gray(img_original)
    img_gray = torch.from_numpy(img_gray).unsqueeze(0).unsqueeze(0).float()
    img_inception = T.ToTensor()(inception_transform(img)).unsqueeze(0)
    img_gray, img_inception = img_gray.to(device), img_inception.to(device)
    model = torch.load(model_pt, map_location='cpu').to(device)
    model.eval()
    with torch.no_grad():
        output = model(img_gray, img_inception)
    img_gray = img_gray.squeeze(0)
    output = output.squeeze(0)
    predicted_image = stack_lab_channels(img_gray, output)
    grayscale = img_gray.permute(1,2,0).numpy()
    save_gen_image(torch.from_numpy(predicted_image).permute(2,0,1), file)
    display_images(grayscale, predicted_image)

"""

"""
def get_transformed_imgs(file):
    img = Image.open(file)
    img_inception = tf_img(inception_transform(load_img(file)))
    img_original = encoder_transform(img)
    img_original = np.asarray(img_original)

    img_lab = rgb2lab(img_original)
    img_lab = (img_lab + 128) / 255

    img_ab = img_lab[:, :, 1:3]
    img_ab = torch.from_numpy(img_ab.transpose((2, 0, 1))).float()

    img_gray = rgb2gray(img_original)
    img_gray = torch.from_numpy(img_gray).unsqueeze(0).float()
    return img_gray, img_ab, img_inception
"""


if __name__ == "__main__":

    args = _parse_args()
    if args.image_type == 'rgb':
        if args.model == 'basic':
            generate_color2color(args.input, args.model_pt)
        elif args.model == 'labnet':
            generate_color2color_lab(args.input, model_pt=args.model_pt)
        elif args.model == 'preres':
            generate_color2color_preres(args.input, args.model_pt)
        elif args.model == 'preincep':
            generate_color2color_preincep(args.input, args.model_pt)
    elif args.image_type == 'gray':
        if args.model == 'basic':
            generate_gray2color(args.input, args.model_pt)
        elif args.model == 'labnet':
            generate_gray2color_lab(args.input, model_pt=args.model_pt)
        elif args.model == 'preres':
            generate_gray2color_preres(args.input, args.model_pt)
        elif args.model == 'preincep':
            generate_gray2color_preincep(args.input, args.model_pt)
