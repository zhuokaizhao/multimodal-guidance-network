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

from vl_dataset import load_data_for_testing


def load_model(model, model_type, checkpoint_path):
    """
    Load a saved checkpoint of your model to all participating processes
    """
    # load the model checkpoint
    checkpoint = torch.load(checkpoint_path)
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
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nCheckpoint weights loaded")

    return model


def main():
    # input arguments
    parser = argparse.ArgumentParser()
    # model type from different MobileOne models
    parser.add_argument("--model_type", action="store", dest="model_type", default="s4")
    # input model path
    parser.add_argument("--checkpoint_path", action="store", dest="checkpoint_path")
    # input dataset ditectory
    parser.add_argument(
        "--test_dir",
        action="store",
        dest="test_dir",
        default="vision_language_dataset/test",
    )
    # test batch size
    parser.add_argument("--batch_size", action="store", dest="batch_size", default=256)
    # output ditectory
    parser.add_argument(
        "--output_dir", action="store", dest="output_dir", default="MobileOne/results/"
    )
    # manually define device
    parser.add_argument("--device", action="store", dest="device", default="cuda:0")
    # verbosity
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=False
    )

    # load basic arguments
    args = parser.parse_args()
    model_type = args.model_type
    checkpoint_path = args.checkpoint_path
    test_dir = args.test_dir
    batch_size = int(args.batch_size)
    output_dir = args.output_dir
    device = args.device
    verbose = args.verbose

    # initialize model
    model = mobileone(variant=f"{model_type}")
    # load finetuned model
    model = load_model(model, model_type, checkpoint_path)
    model.eval()
    # model_eval = reparameterize_model(model)
    if verbose:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nMobileOne-{model_type} model initialized successfully on GPU")
        print(f"Model contains {total_params} total parameters")

    # get model and initialized loss function into memory for each GPU
    model.to(device)

    # load data
    dataloader, dataset_size = load_data_for_testing(test_dir, batch_size)

    # all the inference times for this encoder
    all_time_costs = []
    batch_num_correct = 0
    TP = 0  # True Positive: Data: violative, Pred: violative
    TN = 0  # True Negative: Data: game, Pred: game
    FP = 0  # False Positive: Data: game, Pred: violative
    FN = 0  # False Negative: Data: violative, Pred: game

    with torch.no_grad():
        # iterate over data
        for batch_data_dict in tqdm(dataloader, desc=f"Test Batch"):
            # unpack image inputs and labels
            batch_images = batch_data_dict["images"].to(device)
            batch_labels = batch_data_dict["labels"].to(device)

            # forward pass
            start_time = time.time()
            outputs = model(batch_images)
            all_time_costs.append(time.time() - start_time)
            _, preds = torch.max(outputs, 1)

            # add number of correct
            batch_num_correct += torch.sum(preds == batch_labels.data)

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

    # accuracy
    test_acc = batch_num_correct.cpu().item() / dataset_size
    # compute precision or recall
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    print(f"Test Accuracy: {test_acc}%")
    print(f"Test Precision: {precision}; Test Recall: {recall}")
    # compute average inference time
    avg_inference_time = np.mean(all_time_costs) / batch_size
    # compute normalized time
    # normalized factor takes CLIP-RN50 inference time
    normalize_factor = 0.0007913371767191326
    if not normalize_factor:
        normalize_factor = avg_inference_time
        normalized_inference_time = 1
    else:
        normalized_inference_time = avg_inference_time / normalize_factor
    print(f"Average inference time: {avg_inference_time} seconds")
    print(f"Normalized inference time: {normalized_inference_time}")


if __name__ == "__main__":
    main()
