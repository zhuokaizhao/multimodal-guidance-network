# the script finetunes the pre-trained MobileOne models
import os
import re
import sys
import math
import time
import subprocess
from tqdm import tqdm
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from collections import defaultdict
import matplotlib.pyplot as plt
from mobileone import mobileone, reparameterize_model
from torchvision import transforms

import zoetrope_dataset


# helper function that runs linux command to download model if needed
def runcmd(cmd, verbose=False, *args, **kwargs):
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


# save model
def save_model(
    output_dir,
    model,
    model_name,
    optim_name,
    epoch,
    optimizer,
    history,
    lr_scheduler=None,
):
    """
    Saves model checkpoint on given epoch with given data name.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    checkpoint_path = os.path.join(
        output_dir,
        model_name,
    )
    if optim_name == "adamw" or optim_name == "adam":
        torch.save(
            {
                "num_epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                f"{optim_name}_optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            checkpoint_path,
        )
    else:  # sgd
        torch.save(
            {
                "num_epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                f"{optim_name}_optimizer_state_dict": optimizer.state_dict(),
                f"{optim_name}_lr_scheduler_state_dict": lr_scheduler.state_dict(),
                "history": history,
            },
            checkpoint_path,
        )
    print(f"Checkpoint model saved to {checkpoint_path}")


def load_model(model, model_type, load_pretrain, checkpoint_path):
    """
    Load a saved checkpoint of your model to all participating processes
    """
    # load the model checkpoint
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        # change the last FC layer
        model.linear = torch.nn.Linear(2048, 2)
        model.load_state_dict(checkpoint["model_state_dict"])
        trained_epoch = int(checkpoint["num_epoch"])
        history = checkpoint["history"]
        print(f"\nCheckpoint weights loaded")
    else:
        if load_pretrain:
            # load the pretrained model
            cur_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
            pretrained_model_path = os.path.join(
                cur_dir, "mobileone_unfused", f"mobileone_{model_type}_unfused.pth.tar"
            )
            if not os.path.exists(pretrained_model_path):
                runcmd(
                    f"wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_{model_type}_unfused.pth.tar -P {cur_dir}/mobileone_unfused",
                    verbose=True,
                )
            checkpoint = torch.load(pretrained_model_path)
            # load weights
            model.load_state_dict(checkpoint)
            print(f"\nPretrained weights loaded")

        # change the last FC layer
        if model_type == "s0":
            model.linear = torch.nn.Linear(1024, 2)
        elif model_type == "s1":
            model.linear = torch.nn.Linear(1280, 2)
        elif model_type == "s2":
            model.linear = torch.nn.Linear(2048, 2)
        elif model_type == "s3":
            model.linear = torch.nn.Linear(2048, 2)
        elif model_type == "s4":
            model.linear = torch.nn.Linear(2048, 2)
        # starting training from "scratch"
        trained_epoch = 0
        history = defaultdict(list)

    return model, trained_epoch, history


# data loader for train data using DDP
def load_data(train_dir, val_dir, batch_size):
    # data transforms used in both train and val
    data_transforms = {
        # data augmentation schemes follow https://github.com/open-mmlab/mmpretrain/blob/1.x/configs/mobileone/mobileone-s2_8xb32_in1k.py
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # get the train dataset
    train_data = zoetrope_dataset.ZoetropeDataset(
        train_dir,
        data_transforms["train"],
    )
    print(f"Train dataset loaded successfully - contains {len(train_data)} images")

    # generate dataloader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    # get the val dataset
    val_data = zoetrope_dataset.ZoetropeDataset(
        val_dir,
        data_transforms["val"],
    )
    print(f"Val dataset loaded successfully - contains {len(val_data)} images")
    # generate dataloader
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    # combine into dictionary
    image_datasets = {
        "train": train_data,
        "val": val_data,
    }
    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val"]}

    return dataloaders, dataset_sizes


# multiprocess function that does the training
def run_training(
    model_type,
    load_pretrain,
    train_dir,
    val_dir,
    checkpoint_freq,
    checkpoint_path,
    optim_name,
    num_epochs,
    batch_size,
    output_dir,
    device,
    verbose,
):
    # initialize model
    model = mobileone(variant=f"{model_type}")
    if verbose:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nMobileOne-{model_type} model initialized successfully on GPU")
        print(f"Model contains {total_params} total parameters")

    # load pretrained model or previous checkpoint weights
    model, trained_epoch, history = load_model(
        model, model_type, load_pretrain, checkpoint_path
    )
    # get model and initialized loss function into memory for each GPU
    model.to(device)

    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters())
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            optimizer.load_state_dict(checkpoint[f"{optim_name}_optimizer_state_dict"])
    elif optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters())
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            optimizer.load_state_dict(checkpoint[f"{optim_name}_optimizer_state_dict"])
    elif optim_name == "sgd":
        # define training attributes as in paper - problem: weight decay scheduler not available in pytorch
        # SGD with momentum optimizer, initial learning rate is 0.1, initial weight decay is 10^-4
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=10e-4)
        # learning rate annealed using a cosine schedule
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path)
            optimizer.load_state_dict(checkpoint[f"{optim_name}_optimizer_state_dict"])
            lr_scheduler.load_state_dict(
                checkpoint[f"{optim_name}_lr_scheduler_state_dict"]
            )

    # label smoothing regularization with cross entropy loss with smoothing factor set to 0.1
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion.to(device)

    # load data
    dataloaders, dataset_sizes = load_data(train_dir, val_dir, batch_size)

    # start training
    best_epoch = 0
    best_precision = 0
    best_recall = 0
    for epoch in range(trained_epoch, trained_epoch + num_epochs):
        print(f"\nEpoch {epoch+1}/{trained_epoch+num_epochs}")
        TP = 0  # True Positive: Data: violative, Pred: violative
        TN = 0  # True Negative: Data: game, Pred: game
        FP = 0  # False Positive: Data: game, Pred: violative
        FN = 0  # False Negative: Data: violative, Pred: game

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # stats in current epoch and current phase
            batch_losses = 0.0
            batch_num_correct = 0
            all_batch_losses = []
            all_batch_accuracy = []
            # iterate over data
            for images, labels in tqdm(dataloaders[phase], desc=f"{phase} Batch"):
                # put data to device
                images = images.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward pass + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                if optim_name == "sgd":
                    lr_scheduler.step()

                # stats
                batch_losses += loss.item() * images.size(0)
                all_batch_losses.append(loss.item())
                batch_num_correct += torch.sum(preds == labels.data)
                all_batch_accuracy.append(
                    torch.sum(preds == labels.data).cpu().item() / len(images)
                )

                # compute stats on precision, recall during validation phase
                if phase == "val":
                    for i in range(len(images)):
                        cur_label = labels.data[i]
                        cur_pred = preds.data[i]
                        if cur_label == 0:  # game data
                            if cur_label == cur_pred:  # True Negative
                                TN += 1
                            else:  # False Positive
                                FP += 1
                        elif cur_label == 1:  # violative data
                            if cur_label == cur_pred:  # True Positive
                                TP += 1
                            else:  # False Negative
                                FN += 1

            epoch_loss = batch_losses / dataset_sizes[phase]
            epoch_acc = batch_num_correct.double() / dataset_sizes[phase]
            print(f"Loss: {epoch_loss}; Acc: {epoch_acc}")
            history[f"{phase}_batch_loss"] += all_batch_losses
            history[f"{phase}_batch_accuracy"] += all_batch_accuracy
            history[f"{phase}_epoch_loss"].append(epoch_loss)
            history[f"{phase}_epoch_accuracy"].append(epoch_acc)
            if phase == "val":
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                history[f"precision"].append(precision)
                history[f"recall"].append(recall)
                print(
                    f"Current Precision: {precision}; Best Precision: {best_precision}"
                )
                print(f"Test Recall: {recall}; Best Recall: {best_recall}")

                # save the best model based on precision and recall
                if (precision + recall) >= (best_precision + best_recall):
                    best_precision = precision
                    best_recall = recall
                    model_name = f"best_mobileone_{model_type}_{optim_name}_batch_size_{batch_size}.pt"
                    history["num_epoch"] = epoch + 1
                    best_dir = os.path.join(output_dir, "best_models")
                    save_model(
                        best_dir,
                        model,
                        model_name,
                        optim_name,
                        epoch,
                        optimizer,
                        history,
                    )

        # save checkpoint every 5 epochs
        if (epoch + 1) % checkpoint_freq == 0:
            model_name = f"mobileone_{model_type}_{optim_name}_batch_size_{batch_size}_epoch_{epoch+1}.pt"
            history["num_epoch"] = epoch + 1
            save_model(
                output_dir, model, model_name, optim_name, epoch, optimizer, history
            )


def main():
    # input arguments
    parser = argparse.ArgumentParser()
    # model type from MobileOne-based models
    parser.add_argument("--model_type", action="store", dest="model_type", default="s4")
    # whether or not load pre-trained weights for model type s0 - s4
    parser.add_argument(
        "--load_pretrain", action="store_true", dest="load_pretrain", default=False
    )
    # training dataset ditectory
    parser.add_argument(
        "--train_dir",
        action="store",
        dest="train_dir",
        default="vision_language_dataset/train/",
    )
    # validation dataset ditectory
    parser.add_argument(
        "--val_dir",
        action="store",
        dest="val_dir",
        default="vision_language_dataset/test/",
    )
    # optimizer options
    parser.add_argument(
        "--optimizer", action="store", dest="optim_name", default="adamw"
    )
    # number of training epochs
    parser.add_argument("--num_epochs", action="store", dest="num_epochs", default=100)
    # test batch size
    parser.add_argument("--batch_size", action="store", dest="batch_size", default=128)
    # frequency for saving checkpoint model
    parser.add_argument(
        "--checkpoint_freq", action="store", dest="checkpoint_freq", default=10
    )
    # input model path for continue training
    parser.add_argument("--checkpoint_path", action="store", dest="checkpoint_path")
    # output ditectory for model checkpoint, etc
    parser.add_argument(
        "--output_dir",
        action="store",
        dest="output_dir",
        default="MobileOne/finetuned_models/",
    )
    # manually define device
    parser.add_argument("--device", action="store", dest="device", default="cuda:0")
    # verbosity
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=False
    )

    # load basic arguments
    args = parser.parse_args()
    # we have to get pre-trained model type
    model_type = args.model_type
    if model_type not in ["s0", "s1", "s2", "s3", "s4"]:
        raise ValueError(
            f"Invalid model type: {model_type}, available models: 's0', 's1', 's2', 's3', 's4'"
        )
    load_pretrain = args.load_pretrain
    train_dir = args.train_dir
    val_dir = args.val_dir
    optim_name = args.optim_name
    num_epochs = int(args.num_epochs)
    batch_size = int(args.batch_size)
    checkpoint_freq = int(args.checkpoint_freq)
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    device = args.device
    verbose = args.verbose

    if verbose:
        print(f"model_type: {model_type}")
        print(f"load_pretrain: {load_pretrain}")
        print(f"train_dir: {train_dir}")
        print(f"val_dir: {val_dir}")
        print(f"optimizer name: {optim_name}")
        print(f"num_epoch: {num_epochs}")
        print(f"batch_size: {batch_size}")
        print(f"checkpoint_freq: {checkpoint_freq}")
        print(f"checkpoint_path: {checkpoint_path}")
        print(f"output_dir: {output_dir}")
        print(f"device: {device}")

    # start training
    run_training(
        model_type,
        load_pretrain,
        train_dir,
        val_dir,
        checkpoint_freq,
        checkpoint_path,
        optim_name,
        num_epochs,
        batch_size,
        output_dir,
        device,
        verbose,
    )


if __name__ == "__main__":
    main()
