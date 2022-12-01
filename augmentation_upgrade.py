import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics.aggregation import MeanMetric
from torchmetrics.functional.classification import accuracy
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import scikitplot as skplt
import seaborn as sn
import pandas as pd

import tensorflow as tf
import pathlib

from src.datasets import Labeled_data
from src.models import ConvNet, Net
from src.engines import train, evaluate
from src.utils import load_checkpoint, save_checkpoint, save_pretrained

parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default="augmentation_upgrade")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--root", type=str, default="data")
parser.add_argument("--batch_size", type=int, default=13)
parser.add_argument("--num_workers", type=int, default=3)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--label_smoothing", type=float, default=0.05)
parser.add_argument("--logs", type=str, default='logs')
parser.add_argument("--checkpoints", type=str, default='checkpoints')
parser.add_argument("--resume", type=bool, default=False)
args = parser.parse_args()


def visulaize_result(classes, y_true, y_pred, title, epochs, train_acc, val_acc, train_loss, val_loss):
    print(classification_report(y_true, y_pred, target_names=classes))
    
    ## Build confusion matrix
    y_true = y_true[44:]
    y_pred = y_pred[44:]
    print(y_true)
    print(y_pred)
    print(len(y_true))
    print(len(y_pred))
    
    
    print("\nConfusion Matrix : ")
    print(confusion_matrix(y_true, y_pred))
    
    skplt.metrics.plot_confusion_matrix([classes[i] for i in y_true], [classes[i] for i in y_pred],
                                    normalize=True,
                                    title="Confusion Matrix",
                                    cmap="Purples",
                                    hide_zeros=True,
                                    figsize=(12,8))
    plt.xticks(rotation=90)
    plt.savefig(f'savefig/confusion_matrix_{title}.png')
    
    ## train_val_accuracy_loss
    epochs_range = range(args.epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig(f'savefig/train_val_acc_loss_{title}.png')
    plt.show()
    
def main(args):
    # Build dataset
    label_root = 'data/crop_data_labeled_class6'
    
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(90),
        T.TrivialAugmentWide(),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])   
    train_data = Labeled_data(label_root, train=True, transform=train_transform)
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])    
    val_data = Labeled_data(label_root, train=False, transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    
    # Build model
    model = ConvNet()
    checkpoint_path = f'{args.checkpoints}/augmentation_save_model.pth'
    model.load_state_dict(torch.load(checkpoint_path))
    model = model.to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    metric_fn = accuracy

    # Build logger
    train_logger = SummaryWriter(f'{args.logs}/train/{args.title}')
    val_logger = SummaryWriter(f'{args.logs}/val/{args.title}')

    # Load model
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.checkpoints, args.title, model, optimizer)
    
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
    
    y_pred = []
    y_true = []
    # Main loop
    for epoch in range(start_epoch, args.epochs):

        train_summary = train(train_loader, model, optimizer, scheduler, loss_fn, metric_fn, args.device)
        val_summary, y_pred, y_true = evaluate(val_loader, model, loss_fn, metric_fn, args.device)
        
        y_pred.extend(y_pred)
        y_true.extend(y_true)

        # print log
        print((f'Epoch {epoch+1}: '
                + f'Train Loss {train_summary["loss"]:.04f}, '
                + f'Train Accuracy {train_summary["metric"]:.04f}, '
                + f'Val Loss {val_summary["loss"]:.04f}, '
                + f'Val Accuracy {val_summary["metric"]:.04f}'))

        train_loss.append(train_summary["loss"])
        train_acc.append(train_summary["metric"])
        val_loss.append(val_summary["loss"])
        val_acc.append(val_summary["metric"])

        # write log
        train_logger.add_scalar('Loss', train_summary['loss'], epoch + 1)
        train_logger.add_scalar('Accuracy', train_summary['metric'], epoch + 1)
        val_logger.add_scalar('Loss', val_summary['loss'], epoch + 1)
        val_logger.add_scalar('Accuracy', val_summary['metric'], epoch + 1)

        # save model
        save_checkpoint(args.checkpoints, args.title, model, optimizer, epoch + 1)
        
    save_pretrained(args.checkpoints, f'{args.title}_save_model', model)
    

    
    # visualize result
    classes = ('cataracts', 'conjunctivitis', 'exophthalmos', 'glaucoma', 'normal', 'uveitis')
    
    visulaize_result(classes, y_true, y_pred, args.title, args.epochs, train_acc, val_acc, train_loss, val_loss)
    
    

if __name__=="__main__":
    main(args)