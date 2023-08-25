import os
import re
import clip
import math
from tqdm import tqdm
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import zoetrope_dataset


def main():
    # input arguments
    parser = argparse.ArgumentParser()
    # if model inference results are provided, no need for all other inputs
    parser.add_argument("--embed_dir", action="store", dest="embed_dir")
    # model type from CLIP models
    parser.add_argument(
        "--clip_model_type", action="store", dest="model_type", default="ViT-B/16"
    )
    # input dataset ditectory
    parser.add_argument("--data_dir", action="store", dest="data_dir", default="data/")
    # test batch size
    parser.add_argument("--batch_size", action="store", dest="batch_size", default=512)
    # output ditectory
    parser.add_argument(
        "--output_dir", action="store", dest="output_dir", default="embeddings/"
    )
    # visualized plto ditectory
    parser.add_argument("--plot_dir", action="store", dest="plot_dir", default="plots/")
    # verbosity
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=True
    )

    # cuda information
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # load arguments
    args = parser.parse_args()
    if args.embed_dir:
        embed_dir = args.embed_dir
        all_game_embeddings = np.load(os.path.join(embed_dir, "game_embeddings.npy"))
        print(f"\nLoaded image embeddings with shape {all_game_embeddings.shape}")
        all_violative_embeddings = np.load(
            os.path.join(embed_dir, "violative_embeddings.npy")
        )
        print(
            f"Loaded violative embeddings with shape {all_violative_embeddings.shape}"
        )

        #
    # load model and data for inference
    else:
        model_type = args.model_type
        if model_type not in clip.available_models():
            raise ValueError(
                f"Invalid model type: {model_type}, available models: {clip.available_models()}"
            )

        data_dir = args.data_dir
        batch_size = int(args.batch_size)
        output_dir = args.output_dir
        verbose = args.verbose

        # load model and process (A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input)
        model, preprocess = clip.load(model_type, device=device, jit=False)
        if verbose:
            print(f"\n{model_type} model loaded successfully")

        # load test data
        test_data_dir = os.path.join(data_dir, "test")
        # assumed data file system has test/game and test/violative
        test_game_dir = os.path.join(test_data_dir, "game")
        test_violative_dir = os.path.join(test_data_dir, "violative")
        if not os.path.exists(test_game_dir) or not os.path.exists(test_violative_dir):
            raise Exception(
                f"Data dir {test_data_dir} is invalid: must contain subfolders 'game' and 'violative'"
            )

        # load the game clip dataset
        game_data, game_loader = zoetrope_dataset.load_test_data(
            test_game_dir, preprocess=preprocess, batch_size=batch_size
        )
        # get all the game clip embeddings
        all_game_embeddings = np.zeros((len(os.listdir(test_game_dir)), 512))
        with torch.no_grad():
            for i, (batch_img, batch_text) in enumerate(tqdm(game_loader)):
                batch_image_features = model.encode_image(batch_img.to(device))
                all_game_embeddings[i * batch_size : (i + 1) * batch_size] = (
                    batch_image_features[i].cpu().detach().numpy()
                )

        # load the violative clip dataset
        violative_data, violative_loader = zoetrope_dataset.load_test_data(
            test_violative_dir, preprocess=preprocess, batch_size=batch_size
        )
        all_violative_embeddings = np.zeros((len(os.listdir(test_violative_dir)), 512))
        with torch.no_grad():
            for i, (batch_img, batch_text) in enumerate(tqdm(violative_loader)):
                batch_image_features = model.encode_image(batch_img.to(device))
                all_violative_embeddings[i * batch_size : (i + 1) * batch_size] = (
                    batch_image_features[i].cpu().detach().numpy()
                )

        # save resulting embeddigns
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        np.save(os.path.join(output_dir, "game_embeddings.npy"), all_game_embeddings)
        np.save(
            os.path.join(output_dir, "violative_embeddings.npy"),
            all_violative_embeddings,
        )
        print(f"Image and text embeddings have been saved to {output_dir}")

    plot_dir = args.plot_dir
    # dimension reduction to 2
    print("\nComputing T-SNE on game embeddings...")
    reduced_game_embeddings = TSNE(
        n_components=2, perplexity=30, n_iter=1000, verbose=False
    ).fit_transform(all_game_embeddings)
    print("\nComputing T-SNE on violative embeddings...")
    reduced_violative_embeddings = TSNE(
        n_components=2, perplexity=30, n_iter=1000, verbose=False
    ).fit_transform(all_violative_embeddings)
    print(reduced_game_embeddings.shape)
    # visualize results
    plt.scatter(
        reduced_game_embeddings[:, 0],
        reduced_game_embeddings[:, 1],
        alpha=0.5,
        c="orange",
        label="Game",
    )
    # plt.scatter(
    #     reduced_violative_embeddings[0, 0],
    #     reduced_violative_embeddings[0, 1],
    #     c="green",
    #     label="violative",
    # )
    # plt.legend(["Game", "Violative"])
    # save the plot
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)
    plot_path = os.path.join(plot_dir, "low_embed_plot.png")
    plt.savefig(plot_path)
    print(
        f"\nLow dimensional plot of game and violative embeddings has been saved to {plot_path}"
    )


if __name__ == "__main__":
    main()
