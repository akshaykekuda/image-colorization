import torch
import os

from colorize_data import ColorizeData, LabColorizeData, PreIncColorizeData
from torch.utils.data import DataLoader
from basic_model import Net, LabNet, PreInceptionNet, PreResNet
from torch.nn import MSELoss, HuberLoss
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Trainer:
    """
    This class initializes trainer for the basic network.
    bs = Batch size for training
    lr = learning rate
    epochs = number of epochs to train
    loss_fn = the loss function to use. Possible choices: MSE or Huber Loss
    These loss functions can be ideal to our problem setting as we are dealing with a regression like problem
    """
    def __init__(self, bs, lr, epochs, loss_fn):
        self.bs = bs
        self.lr = lr
        self.epochs = epochs
        self.loss_fn = loss_fn
        pass
        # Define hparams here or load them from a config file
    def train(self, train_df, val_df):
        pass
        # dataloaders
        train_dataset = ColorizeData(train_df)
        train_dataloader = DataLoader(train_dataset, batch_size = self.bs)
        val_dataset = ColorizeData(val_df)
        val_dataloader = DataLoader(val_dataset, batch_size = self.bs)
        # Model
        model = Net()
        # Loss function to use
        try:
            if self.loss_fn=='mse':
                criterion = MSELoss()
            elif self.loss_fn=='huber':
                criterion = HuberLoss()
        except:
            raise ValueError("Loss function {} is invalid".format(self.loss_fn))
        # You may also use a combination of more than one loss function
        # or create your own.
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # train loop
        model = model.to(device)
        model.train()
        loss_arr=[]
        min_val_loss = float('inf')
        print('Start of Basic Network Training')

        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in tqdm(train_dataloader):
                input = batch[0].to(device)
                output = model(input)
                target = batch[1].to(device)
                loss = criterion(output, target)
                optimizer.zero_grad()
                epoch_loss+=loss.item()
                loss.backward()
                optimizer.step()
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print("Loss at epoch {} = {}".format(epoch, avg_epoch_loss))
            loss_arr.append(avg_epoch_loss)
            if epoch %5 == 4:
                val_loss = self.validate(model, criterion, val_dataloader)
                print("Validation error at epoch {} is {}".format(epoch, val_loss))
                if val_loss < min_val_loss:
                    # checkpointing only when the val loss is decreasing
                    print("Val loss less at epoch {}. Saving model".format(epoch))
                    torch.save(model, 'colorizer_basic.model')
                    min_val_loss = val_loss
                model.train()
        print('End of Training')

    def validate(self, model, criterion, val_dataloader):
        # Validation loop begin
        # ------
        model.eval()
        error = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                input = batch[0].to(device)
                output = model(input)
                target = batch[1].to(device)
                error += criterion(output, target)
        return error/len(val_dataloader)

        # Validation loop end
        # ------
        # Determine your evaluation metrics on the validation dataset.

class LabTrainer(Trainer):
    """
    This class initializes trainer for the basic LAB based network.
    The overall objective of this model is to predict A and B channels of a image given its L channel. This is
    a easier task now as the objective becomes predicting a 2 channel image rather than a 3 channel image. This
    ensures faster conversion.

    bs = Batch size for training
    lr = learning rate
    epochs = number of epochs to train
    loss_fn = the loss function to use. Possible choices: MSE or Huber Loss
    """
    def train(self, train_df, val_df):
        pass
        # dataloaders
        train_dataset = LabColorizeData(train_df)
        train_dataloader = DataLoader(train_dataset, batch_size = self.bs)
        val_dataset = LabColorizeData(val_df)
        val_dataloader = DataLoader(val_dataset, batch_size = self.bs)
        # Model
        model = LabNet()
        # Loss function to use
        if self.loss_fn=='mse':
            criterion = MSELoss()
        elif self.loss_fn=='huber':
            criterion = HuberLoss()        # You may also use a combination of more than one loss function
        # or create your own.
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # train loop
        model = model.to(device)
        model.train()
        loss_arr=[]
        print("Running Lab Network training")
        min_val_loss = float('inf')
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in tqdm(train_dataloader):
                input = batch[0].to(device)
                output = model(input)
                target = batch[1].to(device)
                loss = criterion(output, target)
                optimizer.zero_grad()
                epoch_loss+=loss.item()
                loss.backward()
                optimizer.step()
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print("Loss at epoch {} = {}".format(epoch, avg_epoch_loss))
            loss_arr.append(avg_epoch_loss)
            if epoch %5 == 4:
                val_loss = self.validate(model, criterion, val_dataloader)
                print("Validation loss at epoch {} is {}".format(epoch, val_loss))
                if val_loss < min_val_loss:
                    print("Val loss less at epoch {}. Saving model".format(epoch))
                    torch.save(model, 'colorizer_labnet.model')
                    min_val_loss = val_loss
                model.train()
        print('End of Training')


    def validate(self, model, criterion, val_dataloader):
        pass
        # Validation loop begin
        # ------
        model.eval()
        loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                input = batch[0].to(device)
                output = model(input)
                target = batch[1].to(device)
                loss += criterion(output, target)
        return loss / len(val_dataloader)

class PreInceptionTrainer(Trainer):
    """
    This class initializes trainer for the pretrained InceptionV3 based network.
    The pretrained inception block acts as a feature extractor, which extracts features from the image.
    These features are fused with the encoder representations to form a encoder block.
    The decoder block remains similar to the previous experiments. The weights of the pretrained inception
    network are not updated during training.


    bs = Batch size for training
    lr = learning rate
    epochs = number of epochs to train
    loss_fn = the loss function to use. Possible choices: MSE or Huber Loss
    """

    def train(self, train_df, val_df):
        pass
        # dataloaders
        train_dataset = PreIncColorizeData(train_df)
        train_dataloader = DataLoader(train_dataset, batch_size = self.bs)
        val_dataset = PreIncColorizeData(val_df)
        val_dataloader = DataLoader(val_dataset, batch_size = self.bs)
        # Model
        model = PreInceptionNet(pretrained=True)
        # Loss function to use
        if self.loss_fn=='mse':
            criterion = MSELoss()
        elif self.loss_fn=='huber':
            criterion = HuberLoss()        # You may also use a combination of more than one loss function
        # or create your own.
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # train loop
        model = model.to(device)
        model.train()
        min_val_loss = float('inf')
        loss_arr=[]
        print("Start of Inception based Network training")
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in tqdm(train_dataloader):
                img_gray, img_ab, img_inception = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                optimizer.zero_grad()
                output = model(img_gray, img_inception)
                loss = criterion(output, img_ab)
                epoch_loss+=loss.item()
                loss.backward()
                optimizer.step()
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print("Loss at epoch {} = {}".format(epoch, avg_epoch_loss))
            loss_arr.append(avg_epoch_loss)
            if epoch %5 == 4 or epoch == self.epochs-1:
                val_loss = self.validate(model, criterion, val_dataloader)
                print("Validation loss at epoch {} is {}".format(epoch, val_loss))
                if val_loss < min_val_loss:
                    print("Val loss less at epoch {}. Saving model".format(epoch))
                    torch.save({'encoder_state_dict':model.encoder.state_dict(),
                                'decoder_state_dict':model.decoder.state_dict()}, 'colorizer_pre_incep.model')
                    min_val_loss = val_loss
                    model.train()
        print('End of Training')


    def validate(self, model, criterion, val_dataloader):
        pass
        # Validation loop begin
        # ------
        model.eval()
        loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                img_gray, img_ab, img_inception = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                output = model(img_gray, img_inception)
                loss += criterion(output, img_ab)
        return loss/len(val_dataloader)

class PreResTrainer(PreInceptionTrainer):
    """
    This class initializes trainer for the pretrained InceptionResnetV2 based network.
    The pretrained inception block acts as a feature extractor, which extracts features from the image.
    These features are fused with the encoder representations to form a encoder block.
    The decoder block remains similar to the previous experiments. The weights of the pretrained inception
    network are not updated during training.

    bs = Batch size for training
    lr = learning rate
    epochs = number of epochs to train
    loss_fn = the loss function to use. Possible choices: MSE or Huber Loss
    """

    def train(self, train_df, val_df):
        pass
        # dataloaders
        train_dataset = PreIncColorizeData(train_df)
        train_dataloader = DataLoader(train_dataset, batch_size = self.bs)
        val_dataset = PreIncColorizeData(val_df)
        val_dataloader = DataLoader(val_dataset, batch_size = self.bs)
        # Model
        model = PreResNet(pretrained=True)
        # Loss function to use
        if self.loss_fn=='mse':
            criterion = MSELoss()
        elif self.loss_fn=='huber':
            criterion = HuberLoss()
        # You may also use a combination of more than one loss function
        # or create your own.
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # train loop
        model = model.to(device)
        model.train()
        min_val_loss = float('inf')
        loss_arr=[]
        print("Start of InceptionResnet based Network training")

        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch in tqdm(train_dataloader):
                img_gray, img_ab, img_inception = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                optimizer.zero_grad()
                output = model(img_gray, img_inception)
                loss = criterion(output, img_ab)
                epoch_loss+=loss.item()
                loss.backward()
                optimizer.step()
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print("Loss at epoch {} = {}".format(epoch, avg_epoch_loss))
            loss_arr.append(avg_epoch_loss)
            if epoch %5 == 4 or epoch == self.epochs-1:
                val_loss = self.validate(model, criterion, val_dataloader)
                print("Validation loss at epoch {} is {}".format(epoch, val_loss))
                if val_loss < min_val_loss:
                    print("Val loss less at epoch {}. Saving model".format(epoch))
                    torch.save(model, 'colorizer_pre_res.model')
                    min_val_loss = val_loss
                    model.train()
        print('End of Training')
