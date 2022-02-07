import pandas as pd
from sklearn.model_selection import train_test_split
from train import Trainer, LabTrainer, PreInceptionTrainer, PreResTrainer
import torch
import numpy as np
import os
import argparse

np.random.seed(0)
torch.manual_seed(0)

def _parse_args():
    """
    Command-line arguments to the system.
    :return: the parsed args bundle
    """
    parser = argparse.ArgumentParser(description='grayscale2color.py')

    # General system running and configuration options
    parser.add_argument('--model', type=str, default='basic', help='model to run')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
    parser.add_argument('--loss', type=str, default='mse', help='loss function to use')
    args = parser.parse_args()
    return args

image_pt = 'landscape_images/'

if __name__ == "__main__":
    args = _parse_args()
    print(args)
    image_df = pd.DataFrame(columns = ['file_name'])
    image_df.file_name = os.listdir(image_pt)
    image_df.file_name = image_pt + image_df.file_name
    print("Number of images =", len(image_df))
    train_df, val_df = train_test_split(image_df, test_size=0.20)
    try:
        if args.model == 'basic':
            trainer = Trainer(bs=args.bs , lr =args.lr, epochs=args.epochs, loss_fn = args.loss)
        elif args.model == 'labnet':
            trainer = LabTrainer(bs=args.bs , lr =args.lr, epochs=args.epochs, loss_fn = args.loss)
        elif args.model == 'preres':
            trainer = PreResTrainer(bs=args.bs , lr =args.lr, epochs=args.epochs, loss_fn = args.loss)
        elif args.model == 'preincep':
            trainer = PreInceptionTrainer(bs=args.bs , lr =args.lr, epochs=args.epochs, loss_fn = args.loss)
    except:
        raise ValueError("{} not a vaild model".format(args.model))

    trainer.train(train_df, val_df)



