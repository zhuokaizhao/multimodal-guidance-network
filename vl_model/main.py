# main file for running the vision-language safety model
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
from torchvision import transforms

from model import VisionLanguageSafetyModel, VisionOnlySafetyModel
from vl_dateset import load_data_for_training, load_data_for_testing
from mobileone import mobileone_encoder


# helper function that runs linux command to download model if needed
def runcmd(cmd, verbosity=False, *args, **kwargs):
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    std_out, std_err = process.communicate()
    if verbosity:
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


def load_model_checkpoint(
    model,
    checkpoint_path,
    strict=True,
):
    """
    Load a saved checkpoint of your model to all participating processes
    """
    # load the model checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    trained_epoch = int(checkpoint["num_epoch"])
    history = checkpoint["history"]
    print(f"\nCheckpoint weights loaded")

    return model, trained_epoch, history


def load_optim_checkpoint(
    optim_name,
    optimizer,
    checkpoint_path,
    lr_scheduler=None,
):
    """
    Load a saved checkpoint of your model optimizer to all participating processes
    """
    # load the model checkpoint
    checkpoint = torch.load(checkpoint_path)
    optimizer.load_state_dict(checkpoint[f"{optim_name}_optimizer_state_dict"])
    if lr_scheduler != None:
        lr_scheduler.load_state_dict(
            checkpoint[f"{optim_name}_lr_scheduler_state_dict"]
        )
        print(f"\nOptimizer and lr_scheduler weights loaded")
        return optimizer, lr_scheduler
    else:
        print(f"\nOptimizer weights loaded")
        return optimizer


# multiprocess function that does the training
def run_training(
    vision_encoder_name,
    text_encoder_name,
    freeze_language_encoder,
    train_dir,
    val_dir,
    checkpoint_freq,
    checkpoint_path,
    optim_name,
    num_epochs,
    batch_size,
    output_dir,
    device,
    verbosity,
):
    # initialize model - load pretrained model or previous checkpoint weights
    if checkpoint_path:
        model = VisionLanguageSafetyModel(
            vision_encoder_name=vision_encoder_name,
            text_encoder_name=text_encoder_name,
            num_classes=2,
            load_pretrain=False,
            num_heads=8,
        )
        model, trained_epoch, history = load_model_checkpoint(model, checkpoint_path)
        best_epoch = trained_epoch
        best_precision = history["precision"][-1]
        best_recall = history["recall"][-1]
    else:
        model = VisionLanguageSafetyModel(
            vision_encoder_name=vision_encoder_name,
            text_encoder_name=text_encoder_name,
            num_classes=2,
            load_pretrain=True,
            num_heads=8,
        )
        # initialize training stats
        trained_epoch = 0
        history = defaultdict(list)
        best_epoch = 0
        best_precision = 0
        best_recall = 0

    if verbosity:
        for name, param in model.named_parameters():
            if freeze_language_encoder and "text_encoder" in name:
                param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"\nModel contains {total_params} total parameters and {trainable_params} trainable parameters"
        )

    # get model and initialized loss function into memory for each GPU
    model.to(device)
    print(
        f"VisionLanguageSafetyModel (VLSM) has been loaded on GPU {device} successfully"
    )

    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters())
        if checkpoint_path:
            optimizer = load_optim_checkpoint(optim_name, optimizer, checkpoint_path)
    elif optim_name == "adam":
        optimizer = torch.optim.adam(model.parameters())
        if checkpoint_path:
            optimizer = load_optim_checkpoint(optim_name, optimizer, checkpoint_path)
    elif optim_name == "sgd":
        # define training attributes as in paper - problem: weight decay scheduler not available in pytorch
        # SGD with momentum optimizer, initial learning rate is 0.1, initial weight decay is 10^-4
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=10e-4)
        # learning rate annealed using a cosine schedule
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
        if checkpoint_path:
            optimizer = load_optim_checkpoint(
                optim_name, optimizer, checkpoint_path, lr_scheduler
            )

    # label smoothing regularization with cross entropy loss with smoothing factor set to 0.1
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion.to(device)

    # load data
    dataloaders, dataset_sizes = load_data_for_training(
        train_dir,
        val_dir,
        batch_size,
        text_encoder_name,
    )
    if verbosity:
        print(f"\nDataloader has been generated successfully.")

    # start training
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
            for batch_data_dict in tqdm(dataloaders[phase], desc=f"{phase} Batch"):
                # unpack image inputs
                batch_images = batch_data_dict["images"].to(device)
                # unpack text inputs
                if text_encoder_name == "distilbert":
                    batch_ids = (
                        batch_data_dict["ids"].squeeze(1).to(device)
                    )  # input_ids shape should be (batch_size, seq_length)
                    batch_masks = batch_data_dict["masks"].to(device)
                # unpack labels
                batch_labels = batch_data_dict["labels"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == "train"):
                    if text_encoder_name == "distilbert":
                        outputs = model(
                            images=batch_images,
                            text_input_ids=batch_ids,
                            text_attention_mask=batch_masks,
                        )

                    # get the results
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, batch_labels)

                    # backward pass + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                if optim_name == "sgd":
                    lr_scheduler.step()

                # stats
                batch_losses += loss.item() * batch_images.size(0)
                all_batch_losses.append(loss.item())
                batch_num_correct += torch.sum(preds == batch_labels.data)
                all_batch_accuracy.append(
                    torch.sum(preds == batch_labels.data).cpu().item()
                    / len(batch_images)
                )

                # compute stats on precision, recall during validation phase
                if phase == "val":
                    for i in range(len(batch_images)):
                        cur_label = batch_labels.data[i]
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
            epoch_acc = batch_num_correct.cpu().item() / dataset_sizes[phase]
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
                # save the best model based on precision and recall
                if (precision + recall) >= (best_precision + best_recall):
                    best_precision = max(best_precision, precision)
                    best_recall = max(best_recall, recall)
                    if freeze_language_encoder:
                        model_name = f"best_frozen_vlsm_{vision_encoder_name}_{text_encoder_name}_{optim_name}_batch_size_{batch_size}.pt"
                    else:
                        model_name = f"best_unfrozen_vlsm_{vision_encoder_name}_{text_encoder_name}_{optim_name}_batch_size_{batch_size}.pt"
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
                print(
                    f"Current Precision: {precision}; Best Precision: {best_precision}"
                )
                print(f"Current Recall: {recall}; Best Recall: {best_recall}")

        # save checkpoint every 5 epochs
        if freeze_language_encoder:
            model_name = f"frozen_vlsm_{vision_encoder_name}_{text_encoder_name}_{optim_name}_batch_size_{batch_size}_epoch_{epoch+1}.pt"
        else:
            model_name = f"unfrozen_vlsm_{vision_encoder_name}_{text_encoder_name}_{optim_name}_batch_size_{batch_size}_epoch_{epoch+1}.pt"

        if (epoch + 1) % checkpoint_freq == 0:
            history["num_epoch"] = epoch + 1
            save_model(
                output_dir,
                model,
                model_name,
                optim_name,
                epoch,
                optimizer,
                history,
            )


# adapt multimodal trained vision encoder for single modal inference
def run_adaptation(
    vision_encoder_name,
    text_encoder_name,
    train_dir,
    val_dir,
    checkpoint_freq,
    multimodal_checkpoint_path,  # multimodal model checkpoint path
    vision_checkpoint_path,  # checkpoint path for the small model to resume training
    optim_name,
    num_epochs,
    batch_size,
    output_dir,
    device,
    verbosity,
):
    # initiate adaptation training from multimodal model checkpoint
    if multimodal_checkpoint_path and not vision_checkpoint_path:
        model = VisionOnlySafetyModel(vision_encoder_name=vision_encoder_name)
        # load the weights from large model that correspond to vision encoder
        checkpoint = torch.load(multimodal_checkpoint_path)
        selected_model_state_dict = {}
        for name, param in checkpoint["model_state_dict"].items():
            if "vision_encoder" in name:
                selected_model_state_dict[name] = param
        # load weights to the model
        model.load_state_dict(
            selected_model_state_dict, strict=False
        )  # strict=False allows partial load
        model.to(device)  # move model before constructing optimizer
        if verbosity:
            print(f"Vision encoder weights loaded successfully.")
        if optim_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters())
        elif optim_name == "adam":
            optimizer = torch.optim.AdamW(model.parameters())
        elif optim_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=10e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=1000
            )

        # initialize training stats
        trained_epoch = 0
        history = defaultdict(list)
    # resume adaptation training from a vision checkpoint
    elif not multimodal_checkpoint_path and vision_checkpoint_path:
        model = VisionOnlySafetyModel(vision_encoder_name=vision_encoder_name)
        model, trained_epoch, history = load_model_checkpoint(
            model, vision_checkpoint_path
        )
        model.to(device)  # move model before constructing optimizer
        if optim_name == "adamw":
            optimizer = torch.optim.AdamW(model.parameters())
            optimizer = load_optim_checkpoint(
                optim_name, optimizer, vision_checkpoint_path
            )
        elif optim_name == "adam":
            optimizer = torch.optim.AdamW(model.parameters())
            optimizer = load_optim_checkpoint(
                optim_name, optimizer, vision_checkpoint_path
            )
        elif optim_name == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=10e-4)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=1000
            )
            optimizer, lr_scheduler = load_optim_checkpoint(
                optim_name, optimizer, vision_checkpoint_path, lr_scheduler
            )
        if verbosity:
            print(
                f"Vision-only adapted model weights and {optim_name} optimizers loaded successfully."
            )
    else:
        raise Exception(
            f"multimodal_checkpoint_path ({multimodal_checkpoint_path}) and vision_checkpoint_path ({vision_checkpoint_path}) cannot be both True"
        )

    # training
    if verbosity:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"\nVision-only adapted model contains {total_params} total parameters and {trainable_params} trainable parameters"
        )

    # get model and initialized loss function into memory for each GPU
    print(f"Vision-only model has been loaded on GPU {device} successfully")

    # label smoothing regularization with cross entropy loss with smoothing factor set to 0.1
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion.to(device)

    # load data
    dataloaders, dataset_sizes = load_data_for_training(
        train_dir,
        val_dir,
        batch_size,
    )
    if verbosity:
        print(f"\nDataloader has been generated successfully.")

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
            for batch_data_dict in tqdm(dataloaders[phase], desc=f"{phase} Batch"):
                # unpack image inputs
                batch_images = batch_data_dict["images"].to(device)
                # unpack labels
                batch_labels = batch_data_dict["labels"].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward pass
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(batch_images)

                    # get the results
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, batch_labels)

                    # backward pass + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                if optim_name == "sgd":
                    lr_scheduler.step()

                # stats
                batch_losses += loss.item() * batch_images.size(0)
                all_batch_losses.append(loss.item())
                batch_num_correct += torch.sum(preds == batch_labels.data)
                all_batch_accuracy.append(
                    torch.sum(preds == batch_labels.data).cpu().item()
                    / len(batch_images)
                )

                # compute stats on precision, recall during validation phase
                if phase == "val":
                    for i in range(len(batch_images)):
                        cur_label = batch_labels.data[i]
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
            epoch_acc = batch_num_correct.cpu().item() / dataset_sizes[phase]
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
                    if (
                        "frozen" in multimodal_checkpoint_path.split("/")[-1]
                        and "unfrozen" not in multimodal_checkpoint_path.split("/")[-1]
                    ):
                        model_name = f"best_adapted_frozen_{vision_encoder_name}_{text_encoder_name}_{optim_name}_batch_size_{batch_size}.pt"
                    elif "unfrozen" in multimodal_checkpoint_path.split("/")[-1]:
                        model_name = f"best_adapted_unfrozen_{vision_encoder_name}_{text_encoder_name}_{optim_name}_batch_size_{batch_size}.pt"
                    else:
                        raise Exception(
                            f"Unknown base model {multimodal_checkpoint_path}"
                        )
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
        if multimodal_checkpoint_path and not vision_checkpoint_path:
            if "frozen" in multimodal_checkpoint_path.split("/")[-1]:
                model_name = f"frozen_adapted_{vision_encoder_name}_{text_encoder_name}_{optim_name}_batch_size_{batch_size}_epoch_{epoch+1}.pt"
            elif "unfrozen" in multimodal_checkpoint_path.split("/")[-1]:
                model_name = f"unfrozen_adapted_{vision_encoder_name}_{text_encoder_name}_{optim_name}_batch_size_{batch_size}_epoch_{epoch+1}.pt"
            else:
                raise Exception(f"Unknown base model {multimodal_checkpoint_path}")
        elif not multimodal_checkpoint_path and vision_checkpoint_path:
            if "frozen" in vision_checkpoint_path.split("/")[-1]:
                model_name = f"frozen_adapted_{vision_encoder_name}_{text_encoder_name}_{optim_name}_batch_size_{batch_size}_epoch_{epoch+1}.pt"
            elif "unfrozen" in vision_checkpoint_path.split("/")[-1]:
                model_name = f"unfrozen_adapted_{vision_encoder_name}_{text_encoder_name}_{optim_name}_batch_size_{batch_size}_epoch_{epoch+1}.pt"
            else:
                raise Exception(f"Unknown base model {vision_checkpoint_path}")
        if (epoch + 1) % checkpoint_freq == 0:
            history["num_epoch"] = epoch + 1
            save_model(
                output_dir,
                model,
                model_name,
                optim_name,
                epoch,
                optimizer,
                history,
            )


# function that performs model evaluation
def run_testing(
    inference_option,
    vision_encoder_name,
    text_encoder_name,
    test_dir,
    checkpoint_path,
    batch_size,
    output_dir,
    device,
    verbosity,
):
    # load the model that corresponds to inference option
    if inference_option == "1":  # vision only inference from adapted model
        model = VisionOnlySafetyModel(vision_encoder_name=vision_encoder_name)
        model, trained_epoch, history = load_model_checkpoint(model, checkpoint_path)
    elif inference_option == "2" or inference_option == "3":  # multimodal model
        model = VisionLanguageSafetyModel(
            vision_encoder_name=vision_encoder_name,
            text_encoder_name=text_encoder_name,
            num_classes=2,
            load_pretrain=False,
            num_heads=8,
        )
        model, _, _ = load_model_checkpoint(model, checkpoint_path)

    if verbosity:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model contains {total_params} total parameters")

    model.to(device)

    # test loop
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion.to(device)

    # load data
    dataloader, dataset_size = load_data_for_testing(
        test_dir,
        batch_size,
        text_encoder_name=text_encoder_name,
    )
    if verbosity:
        print(f"\nTest dataloader has been generated successfully.")

    # start testing
    all_time_costs = []
    batch_losses = 0.0
    batch_num_correct = 0
    all_batch_losses = []
    all_batch_accuracy = []
    TP = 0  # True Positive: Data: violative, Pred: violative
    TN = 0  # True Negative: Data: game, Pred: game
    FP = 0  # False Positive: Data: game, Pred: violative
    FN = 0  # False Negative: Data: violative, Pred: game

    # iterate over data
    for batch_data_dict in tqdm(dataloader, desc=f"Test Batch"):
        # unpack image inputs
        batch_images = batch_data_dict["images"].to(device)
        # unpack text inputs
        if text_encoder_name == "distilbert":
            batch_ids = (
                batch_data_dict["ids"].squeeze(1).to(device)
            )  # input_ids shape should be (batch_size, seq_length)
            batch_masks = batch_data_dict["masks"].to(device)
        # unpack labels
        batch_labels = batch_data_dict["labels"].to(device)

        # forward pass
        with torch.set_grad_enabled(False):
            start_time = time.time()
            if inference_option == "1":
                outputs = model(
                    images=batch_images,
                )
            elif inference_option == "2" or inference_option == "3":
                outputs = model(
                    images=batch_images,
                    text_input_ids=batch_ids,
                    text_attention_mask=batch_masks,
                )

            # get the results
            _, preds = torch.max(outputs, 1)
            all_time_costs.append(time.time() - start_time)
            loss = criterion(outputs, batch_labels)

        # stats on accruacy
        batch_losses += loss.item() * batch_images.size(0)
        all_batch_losses.append(loss.item())
        batch_num_correct += torch.sum(preds == batch_labels.data)
        all_batch_accuracy.append(
            torch.sum(preds == batch_labels.data).cpu().item() / len(batch_images)
        )

        # stats on precision, recall
        for i in range(len(batch_images)):
            cur_label = batch_labels.data[i]
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

    # save testing results
    test_loss = batch_losses / dataset_size
    test_acc = batch_num_correct.cpu().item() / dataset_size
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print(f"Test Loss: {test_loss}; Test Acc: {test_acc}")
    print(f"Test Precision: {precision}; Test Recall: {recall}")
    # compute average inference time
    avg_inference_time = np.mean(all_time_costs) / batch_size
    # compute normalized time
    normalize_factor = 0.0011083800106858595  # the ViT-B/16 CLIP baseline
    if not normalize_factor:
        normalize_factor = avg_inference_time
        normalized_inference_time = 1.0
    else:
        normalized_inference_time = avg_inference_time / normalize_factor
    print(f"Average inference time: {avg_inference_time} seconds")
    print(f"Normalized inference time: {normalized_inference_time}")


def main():
    # input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        action="store",
        dest="mode",
        help="Select 'train', 'adapt', or 'test' for mode selection.",
    )
    parser.add_argument(
        "--inference_option",
        action="store",
        dest="inference_option",
        help="Select inference option '1', '2', or '3' that correspond to \
        1. finetuned vision encoder only \
        2. pretrained language encoder + finetuned vision encoder, or \
        3. finetuned both vision and language encoder.",
    )
    parser.add_argument(
        "--vision_encoder",
        action="store",
        dest="vision_encoder",
        default="mobileone-s4",
        help="Name of the vision encoder. Default is 'mobileone-s4'.",
    )
    parser.add_argument(
        "--text_encoder",
        action="store",
        dest="text_encoder",
        default="distilbert",
        help="Name of the text encoder. Default is 'distilbert'.",
    )
    parser.add_argument(
        "--train_dir",
        action="store",
        dest="train_dir",
        default="./vision_language_dataset/train/",
        help="Training data director. Default is './vision_language_dataset/train/'.",
    )
    parser.add_argument(
        "--val_dir",
        action="store",
        dest="val_dir",
        default="./vision_language_dataset/test/",
        help="Validation data director. Default is './vision_language_dataset/test/'.",
    )
    parser.add_argument(
        "--test_dir",
        action="store",
        dest="test_dir",
        default="./vision_language_dataset/test/",
        help="Test data director. Default is './vision_language_dataset/test/'.",
    )
    parser.add_argument(
        "--optimizer",
        action="store",
        dest="optim_name",
        default="adamw",
        help="Optimizer name. Default is 'adamw'.",
    )
    parser.add_argument(
        "--num_epochs",
        action="store",
        dest="num_epochs",
        default=100,
        help="Number of training epochs. Default is 100.",
    )
    parser.add_argument(
        "--batch_size",
        action="store",
        dest="batch_size",
        help="Train/test batch size. Default for train is 128, for test is 1024.",
    )
    parser.add_argument(
        "--checkpoint_freq",
        action="store",
        dest="checkpoint_freq",
        default=10,
        help="Number of epochs when a model checkpoint is saved. Default is 10.",
    )
    parser.add_argument(
        "--vision_language_checkpoint_path",
        action="store",
        dest="vision_language_checkpoint_path",
        help="Input model path for resuming vision-language model training or testing.",
    )
    parser.add_argument(
        "--vision_checkpoint_path",
        action="store",
        dest="vision_checkpoint_path",
        help="Input model path for resuming vision-only model (adapted) training or testing.",
    )
    parser.add_argument(
        "--output_dir",
        action="store",
        dest="output_dir",
        default="./vl_model/models/",
        help="Output ditectory for saving model checkpoints, etc",
    )
    parser.add_argument(
        "--freeze_language_encoder",
        action="store_true",
        dest="freeze_language_encoder",
        default=False,
        help="Freeze language encoder back-propagation when flagged.",
    )
    parser.add_argument(
        "--device",
        action="store",
        dest="device",
        default="cuda:1",
        help="Device that the model is trained on. Default: cuda:1.",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        action="store_true",
        dest="verbosity",
        default=False,
        help="Verbosity.",
    )

    # load basic arguments
    args = parser.parse_args()
    mode = args.mode
    if mode == "test":
        inference_option = args.inference_option
    vision_encoder_name = args.vision_encoder
    text_encoder_name = args.text_encoder
    freeze_language_encoder = args.freeze_language_encoder
    if mode == "train" or mode == "adapt":
        train_dir = args.train_dir
        val_dir = args.val_dir
        optim_name = args.optim_name
        checkpoint_freq = int(args.checkpoint_freq)
    elif mode == "test":
        test_dir = args.test_dir
    num_epochs = int(args.num_epochs)
    if args.batch_size:
        batch_size = int(args.batch_size)
    else:
        if mode == "train" or mode == "adapt":
            batch_size = 128
        elif mode == "test":
            batch_size = 256
    vision_language_checkpoint_path = args.vision_language_checkpoint_path
    vision_checkpoint_path = args.vision_checkpoint_path
    output_dir = args.output_dir
    device = args.device
    verbosity = args.verbosity

    if verbosity:
        print(f"\nmode: {mode}")
        if mode == "test":
            if inference_option == "1":
                print(
                    f"inference_option: {inference_option} - finetuned vision encoder only"
                )
            elif inference_option == "2":
                print(
                    f"inference_option: {inference_option} - pretrained language encoder + finetuned vision encoder"
                )
            elif inference_option == "3":
                print(
                    f"inference_option: {inference_option} - finetuned both vision and language encoder"
                )
            else:
                raise Exception(f"Unknown inference option {inference_option}")
        print(f"vision_encoder: {vision_encoder_name}")
        print(f"text_encoder: {text_encoder_name}")
        print(f"freeze_language_encoder: {freeze_language_encoder}")
        if mode == "train":
            print(f"train_dir: {train_dir}")
            print(f"val_dir: {val_dir}")
            print(f"optimizer name: {optim_name}")
            print(f"checkpoint_freq: {checkpoint_freq}")
        elif mode == "test":
            print(f"test_dir: {test_dir}")
        print(f"num_epoch: {num_epochs}")
        print(f"batch_size: {batch_size}")
        print(f"vision_language_checkpoint_path: {vision_language_checkpoint_path}")
        print(f"vision_checkpoint_path: {vision_checkpoint_path}")
        print(f"output_dir: {output_dir}")
        print(f"device: {device}")

    if mode == "train":
        # start training
        run_training(
            vision_encoder_name=vision_encoder_name,
            text_encoder_name=text_encoder_name,
            freeze_language_encoder=freeze_language_encoder,
            train_dir=train_dir,
            val_dir=val_dir,
            checkpoint_freq=checkpoint_freq,
            checkpoint_path=vision_language_checkpoint_path,
            optim_name=optim_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            output_dir=output_dir,
            device=device,
            verbosity=verbosity,
        )
    elif mode == "adapt":
        run_adaptation(
            vision_encoder_name=vision_encoder_name,
            text_encoder_name=text_encoder_name,
            train_dir=train_dir,
            val_dir=val_dir,
            checkpoint_freq=checkpoint_freq,
            multimodal_checkpoint_path=vision_language_checkpoint_path,
            vision_checkpoint_path=vision_checkpoint_path,
            optim_name=optim_name,
            num_epochs=num_epochs,
            batch_size=batch_size,
            output_dir=output_dir,
            device=device,
            verbosity=verbosity,
        )
    elif mode == "test":
        if inference_option == "1":
            run_testing(
                inference_option=inference_option,
                vision_encoder_name=vision_encoder_name,
                text_encoder_name=text_encoder_name,
                test_dir=test_dir,
                checkpoint_path=vision_checkpoint_path,
                batch_size=batch_size,
                output_dir=output_dir,
                device=device,
                verbosity=verbosity,
            )
        elif (
            inference_option == "2" or inference_option == "3"
        ):  # option 2 and 3 are both vision language model
            run_testing(
                inference_option=inference_option,
                vision_encoder_name=vision_encoder_name,
                text_encoder_name=text_encoder_name,
                test_dir=test_dir,
                checkpoint_path=vision_language_checkpoint_path,
                batch_size=batch_size,
                output_dir=output_dir,
                device=device,
                verbosity=verbosity,
            )
        else:
            raise Exception(f"Unknown inference option {inference_option}.")
    else:
        raise Exception(f"Unknown mode {mode}.")


if __name__ == "__main__":
    main()
