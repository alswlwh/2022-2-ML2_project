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

from src.datasets import Labeled_data, Unlabeled_data, NS_pseudo_data
from src.models import ConvNet
from src.engines import train, evaluate, pl_train, pl_evaluate
from src.utils import load_checkpoint, save_checkpoint, save_pretrained

parser = argparse.ArgumentParser()
parser.add_argument("--title", type=str, default="noisy-student")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--root", type=str, default="data")
parser.add_argument("--labeled_batch_size", type=int, default=13)
parser.add_argument("--unlabeled_batch_size", type=int, default=600)  # [600,1000]
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--label_smoothing", type=float, default=0.05)
parser.add_argument("--logs", type=str, default='logs')
parser.add_argument("--checkpoints", type=str, default='checkpoints')
parser.add_argument("--resume", type=bool, default=False)
args = parser.parse_args()
        

def visulaize_result(classes, y_true, y_pred, title, i, epochs, train_acc, val_acc, train_loss, val_loss):
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
    plt.savefig(f'savefig/confusion_matrix_{title}_{i}_{args.unlabeled_batch_size}.png')
    
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
    plt.savefig(f'savefig/train_val_acc_loss_{title}_{i}_{args.unlabeled_batch_size}.png')
    plt.show()   
    
def main(args):
    # Build dataset
    label_root = 'data/crop_data_labeled_class6'
    unlabel_root = 'data/crop_data_unlabeled'
    
    train_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_data = Labeled_data(label_root, train=True, transform=train_transform)
    train_loader = DataLoader(train_data, args.labeled_batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)

    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])    
    val_data = Labeled_data(label_root, train=False, transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=args.labeled_batch_size, shuffle=False, num_workers=args.num_workers)
        
    # Build model
    model = ConvNet()
    model = model.to(args.device)


    print(f'[model 0] size of total data : ', len(train_data))
 
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    metric_fn = accuracy
            
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
            
    y_pred = []
    y_true = []
    # Main loop
    for epoch in range(args.epochs):

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
                
    save_pretrained(args.checkpoints, f'{args.title}_0_save_model', model)
            
    classes = ('cataracts', 'conjunctivitis', 'exophthalmos', 'glaucoma', 'normal', 'uveitis')
            
    visulaize_result(classes, y_true, y_pred, args.title, 0, args.epochs, train_acc, val_acc, train_loss, val_loss)
    
    
    # append 600 unlabeled data to labeled data each batch (+ 1000)
    unlabeled_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    unlabeled_data = Unlabeled_data(unlabel_root, transform=unlabeled_transform)
    unlabeled_loader = DataLoader(unlabeled_data, args.unlabeled_batch_size, shuffle=True, num_workers=0, drop_last=True)
            
    cnt = 0
    batch_image = torch.ones([args.unlabeled_batch_size, 3, 64, 64])  # [600,1000]
    for i, batch in enumerate(unlabeled_loader):
        
            
        # batch_x = batch
        batch_image = torch.cat((batch_image, batch))

        if cnt == 0:
            batch_image = batch_image[args.unlabeled_batch_size:, :, :, :]  # [600,1000]

        cnt += 1
        

        checkpoint_path = f'{args.checkpoints}/{args.title}_{i}_save_model.pth'
        model.load_state_dict(torch.load(checkpoint_path))
        model = model.to(args.device)
            
        ns_train_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.TrivialAugmentWide(),
            T.ToTensor(),
            T.RandomErasing(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        NS_pseudo_train_data = NS_pseudo_data(label_root, batch_image, model, args.device, transform=ns_train_transform)
        ns_train_loader = DataLoader(NS_pseudo_train_data, args.labeled_batch_size, shuffle=True, num_workers=0, drop_last=True)
            
        print(f'[model {i+1}] size of total data : ', len(NS_pseudo_train_data))

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(ns_train_loader))
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        metric_fn = accuracy

        # Build logger
        train_logger = SummaryWriter(f'{args.logs}/train/{args.title}_{i+1}')
        val_logger = SummaryWriter(f'{args.logs}/val/{args.title}_{i+1}')

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

            ns_train_summary = train(ns_train_loader, model, optimizer, scheduler, loss_fn, metric_fn, args.device)
            val_summary, y_pred, y_true = evaluate(val_loader, model, loss_fn, metric_fn, args.device)

            y_pred.extend(y_pred)
            y_true.extend(y_true)
        
            # print log
            print((f'Epoch {epoch+1}: '
                    + f'Train Loss {ns_train_summary["loss"]:.04f}, '
                    + f'Train Accuracy {ns_train_summary["metric"]:.04f}, '
                    + f'Val Loss {val_summary["loss"]:.04f}, '
                    + f'Val Accuracy {val_summary["metric"]:.04f}'))

            train_loss.append(ns_train_summary["loss"])
            train_acc.append(ns_train_summary["metric"])
            val_loss.append(val_summary["loss"])
            val_acc.append(val_summary["metric"])

            # write log
            train_logger.add_scalar('Loss', ns_train_summary['loss'], epoch + 1)
            train_logger.add_scalar('Accuracy', ns_train_summary['metric'], epoch + 1)
            val_logger.add_scalar('Loss', val_summary['loss'], epoch + 1)
            val_logger.add_scalar('Accuracy', val_summary['metric'], epoch + 1)

            # save model
            save_checkpoint(args.checkpoints, f'{args.title}_{i+1}', model, optimizer, epoch + 1)

        save_pretrained(args.checkpoints, f'{args.title}_{i+1}_save_model', model)

        # visualize result
        classes = ('cataracts', 'conjunctivitis', 'exophthalmos', 'glaucoma', 'normal', 'uveitis')

        visulaize_result(classes, y_true, y_pred, args.title, i+1, args.epochs, train_acc, val_acc, train_loss, val_loss)
    
    '''cnt = 0
    batch_image = torch.ones([args.unlabeled_batch_size, 3, 64, 64])  # [600,800,1000]
    for i, batch in enumerate(unlabeled_loader):
        
        if i > 0:
            
            # batch_x = batch
            batch_image = torch.cat((batch_image, batch))

            if cnt == 0:
                batch_image = batch_image[args.unlabeled_batch_size:, :, :, :]  # [600,800,1000]

            cnt += 1
        
        if i == 0:
            
            print(f'[model {i}] size of total data : ', len(train_data))
 
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(train_loader))
            loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            metric_fn = accuracy
            
            train_acc = []
            val_acc = []
            train_loss = []
            val_loss = []
            
            y_pred = []
            y_true = []
            # Main loop
            for epoch in range(args.epochs):

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
                
            save_pretrained(args.checkpoints, f'{args.title}_{i}_pretrain', model)
            
            # constant for classes
            classes = ('cataracts', 'conjunctivitis', 'exophthalmos', 'glaucoma', 'normal', 'uveitis')
            
            visulaize_result(classes, y_true, y_pred, args.title, i, args.epochs, train_acc, val_acc, train_loss, val_loss)
        
        else:

            checkpoint_path = f'{args.checkpoints}/{args.title}_{i-1}_pretrain.pth'
            model.load_state_dict(torch.load(checkpoint_path))
            model = model.to(args.device)
            
            ns_train_transform = T.Compose([
                T.RandomHorizontalFlip(),
                T.TrivialAugmentWide(),
                T.ToTensor(),
                T.RandomErasing(),
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            NS_pseudo_train_data = NS_pseudo_data(label_root, batch_image, model, args.device, transform=ns_train_transform)
            ns_train_loader = DataLoader(NS_pseudo_train_data, args.labeled_batch_size, shuffle=True, num_workers=0, drop_last=True)
            
            print(f'[model {i}] size of total data : ', len(NS_pseudo_train_data))

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs * len(ns_train_loader))
            loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
            metric_fn = accuracy

            # Build logger
            train_logger = SummaryWriter(f'{args.logs}/train/{args.title}_{i}')
            val_logger = SummaryWriter(f'{args.logs}/val/{args.title}_{i}')

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

                ns_train_summary = train(ns_train_loader, model, optimizer, scheduler, loss_fn, metric_fn, args.device)
                val_summary, y_pred, y_true = evaluate(val_loader, model, loss_fn, metric_fn, args.device)

                y_pred.extend(y_pred)
                y_true.extend(y_true)
        
                # print log
                print((f'Epoch {epoch+1}: '
                        + f'Train Loss {ns_train_summary["loss"]:.04f}, '
                        + f'Train Accuracy {ns_train_summary["metric"]:.04f}, '
                        + f'Val Loss {val_summary["loss"]:.04f}, '
                        + f'Val Accuracy {val_summary["metric"]:.04f}'))

                train_loss.append(ns_train_summary["loss"])
                train_acc.append(ns_train_summary["metric"])
                val_loss.append(val_summary["loss"])
                val_acc.append(val_summary["metric"])

                # write log
                train_logger.add_scalar('Loss', ns_train_summary['loss'], epoch + 1)
                train_logger.add_scalar('Accuracy', ns_train_summary['metric'], epoch + 1)
                val_logger.add_scalar('Loss', val_summary['loss'], epoch + 1)
                val_logger.add_scalar('Accuracy', val_summary['metric'], epoch + 1)

                # save model
                save_checkpoint(args.checkpoints, f'{args.title}_{i}', model, optimizer, epoch + 1)

            save_pretrained(args.checkpoints, f'{args.title}_{i}_pretrain', model)

            # visualize result
            classes = ('cataracts', 'conjunctivitis', 'exophthalmos', 'glaucoma', 'normal', 'uveitis')

            visulaize_result(classes, y_true, y_pred, args.title, i, args.epochs, train_acc, val_acc, train_loss, val_loss)'''
            
    

if __name__=="__main__":
    main(args)