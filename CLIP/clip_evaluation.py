import os
import re
import math
import clip
import time
import subprocess
from tqdm import tqdm
import torch
import pickle
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from vl_dataset import load_data_for_testing


def main():
    # input arguments
    parser = argparse.ArgumentParser()
    # model type from CLIP models
    parser.add_argument(
        "--model_type", action="store", dest="model_type", default="ViT-B/16"
    )
    # input dataset ditectory
    parser.add_argument(
        "--data_dir",
        action="store",
        dest="data_dir",
        default="vision_language_dataset/test",
    )
    # test batch size
    parser.add_argument("--batch_size", action="store", dest="batch_size", default=256)
    # output ditectory
    parser.add_argument(
        "--output_dir", action="store", dest="output_dir", default="CLIP/results/"
    )
    # manually define device
    parser.add_argument("--device", action="store", dest="device", default="cuda:0")
    # verbosity
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=True
    )

    # load basic arguments
    args = parser.parse_args()
    data_dir = args.data_dir
    batch_size = int(args.batch_size)
    output_dir = args.output_dir
    device = args.device
    verbosity = args.verbose

    # we have to get pre-trained model type
    model_type = args.model_type
    if model_type not in clip.available_models() and model_type != "all":
        raise ValueError(
            f"Invalid model type: {model_type}, available models: {clip.available_models()} or 'all'"
        )

    if model_type == "all":
        all_model_types = []
        for model_name in clip.available_models():
            if "x" in model_name:  # remove the redundant-arch model
                continue
            all_model_types.append(model_name)
        if verbosity:
            print(f"Evaluating CLIP variants: {all_model_types}")
    else:
        all_model_types = [model_type]
        if verbosity:
            print(f"Evaluating CLIP variant: {all_model_types}")

    # used for normalized inference time
    normalize_factor = None
    all_results = []
    for cur_model_type in all_model_types:
        # basic result
        total_correct = 0
        total_wrong = 0
        # load model and preprocess (A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input)
        model, preprocess = clip.load(cur_model_type, device=device, jit=False)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        visual_params = sum(
            p.numel() for p in model.visual.parameters() if p.requires_grad
        )
        text_params = sum(
            p.numel() for p in model.token_embedding.parameters() if p.requires_grad
        ) + sum(p.numel() for p in model.transformer.parameters() if p.requires_grad)
        if verbosity:
            print(f"\n{cur_model_type} model loaded successfully. Start evaluation")
            print(
                f"Model contains {total_params} total parameters, with {visual_params} for visual encoder and {text_params} for text encoder"
            )

        # load test data
        dataloader, dataset_size = load_data_for_testing(data_dir, batch_size)
        # text label embeddings
        text_inputs = torch.cat(
            [clip.tokenize(f"a {c} photo") for c in ["in-game", "violative"]]
        )
        if verbosity:
            print(f"\nTest dataloader has been generated successfully.")

        # all the inference times for this encoder
        all_time_costs = []
        batch_num_correct = 0
        TP = 0  # True Positive: Data: violative, Pred: violative
        TN = 0  # True Negative: Data: game, Pred: game
        FP = 0  # False Positive: Data: game, Pred: violative
        FN = 0  # False Negative: Data: violative, Pred: game

        with torch.no_grad():
            # generate image embeddings
            for batch_data_dict in tqdm(dataloader, desc=f"Test Batch"):
                # unpack image inputs and labels
                batch_images = batch_data_dict["images"].to(device)
                batch_labels = batch_data_dict["labels"].to(device)
                start_time = time.time()
                image_features = model.encode_image(batch_images.to(device))
                text_features = model.encode_text(text_inputs.to(device))

                # pick the most similar labels
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                all_time_costs.append(time.time() - start_time)
                _, preds = torch.max(similarity, 1)

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
        if not normalize_factor:
            normalize_factor = avg_inference_time
            normalized_inference_time = 1.0
        else:
            normalized_inference_time = avg_inference_time / normalize_factor

        print(f"Average inference time: {avg_inference_time} seconds")
        print(f"Normalized inference time: {normalized_inference_time}")

        # construct list of lists for panda dataframe
        cur_result = [
            cur_model_type,
            total_params,
            visual_params,
            text_params,
            avg_inference_time,
            normalized_inference_time,
            precision,
            recall,
            test_acc,
        ]
        all_results.append(cur_result)

    # create dataframe and save
    results_df = pd.DataFrame(
        all_results,
        columns=[
            "Model Type",
            "Total Number of Parameters",
            "Number of Visual Encoder Parameters",
            "Number of Text Encoder Parameters",
            "Average Inference Time",
            "Normalized Inference Time",
            "Precision",
            "Recall",
            "Accuracy",
        ],
    )
    print(results_df)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    result_path = os.path.join(output_dir, f"{model_type}_evaluation_results.csv")
    results_df.to_csv(result_path)
    print(f"Results have been saved to {result_path}")


if __name__ == "__main__":
    main()
