from threading import ThreadError
from warnings import resetwarnings
import torch.nn as nn
import torchvision.models as models
import torch

class Net(nn.Module):
    """
    Basic Network
    """
    def __init__(self, input_size=256):
        super(Net, self).__init__()
        # ResNet - First layer accepts grayscale images,
        # and we take only the first few layers of ResNet for this task
        resnet = models.resnet18(num_classes=100)
        resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1))
        self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])
        RESNET_FEATURE_SIZE = 128
        ## Upsampling Network
        self.upsample = nn.Sequential(
            nn.Conv2d(RESNET_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, input):
        midlevel_features = self.midlevel_resnet(input)
        output = self.upsample(midlevel_features)
        return output


class LabNet(Net):
    """
    Output of the network is a 2 channel output corresponding to AB channels
    """
    def __init__(self, input_size=256):
        super(LabNet, self).__init__()

        ## Upsampling Network
        self.upsample[14] = nn.Conv2d(32, 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))


class Encoder(nn.Module):
    """
    The encoder for the neural network.
    The input shape is a 224x224x1 image, which is the L channel.
    """

    def __init__(self):
        super(Encoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Decoder(nn.Module):
    """
    The decoder for the neural network.
    The input shape is the fusion layer indicated in the paper.
    """

    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(1257, 256, 1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 3, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class PreResNet(nn.Module):
    """
    Combines the outputs of the encoder and InceptionResNetV2 model and feeds this
    fused output into the decoder to output a predicted 224x224x2 AB channel.
    """

    def __init__(self, pretrained):
        super(PreResNet, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

        self.feature_extractor = pretrainedmodels.__dict__["inceptionresnetv2"](
            num_classes=1001,
            pretrained="imagenet+background" if pretrained else None
            )
        self.feature_extractor.eval()

    def forward(self, encoder_input, feature_input):
        encoded_img = self.encoder(encoder_input)

        with torch.no_grad():
            embedding = self.feature_extractor(feature_input)

        embedding = embedding.view(-1, 1001, 1, 1)

        rows = torch.cat([embedding] * 28, dim=3)
        embedding_block = torch.cat([rows] * 28, dim=2)
        fusion_block = torch.cat([encoded_img, embedding_block], dim=1)

        return self.decoder(fusion_block)


class PreInceptionNet(nn.Module):
    """
    Combines the outputs of the encoder and InceptionV3 model and feeds this
    fused output into the decoder to output a predicted 224x224x2 AB channel.
    """

    def __init__(self, pretrained):
        super(PreInceptionNet, self).__init__()
        self.feature_extractor = models.inception_v3(pretrained=pretrained, aux_logits=False)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.decoder.decoder[0] = nn.Conv2d(1256, 256, 1)
        self.feature_extractor.eval()

    def forward(self, encoder_input, feature_input):
        encoded_img = self.encoder(encoder_input)

        with torch.no_grad():
            embedding = self.feature_extractor(feature_input)

        embedding = embedding.view(-1, 1000, 1, 1)

        rows = torch.cat([embedding] * 28, dim=3)
        embedding_block = torch.cat([rows] * 28, dim=2)
        fusion_block = torch.cat([encoded_img, embedding_block], dim=1)

        return self.decoder(fusion_block)
