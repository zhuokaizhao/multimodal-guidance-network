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
def save_model(output_dir, model, model_type, epoch, optimizer, history):
    """
    Saves model checkpoint on given epoch with given data name.
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    checkpoint_path = os.path.join(
        output_dir, f"mobileone_{model_type}_finetune_epoch_{epoch}.pt"
    )
    torch.save(
        {
            "num_epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
        },
        checkpoint_path,
    )
    print(f"Checkpoint model saved to {checkpoint_path}")


def load_model(rank, model, model_type, checkpoint_path):
    """
    Load a saved checkpoint of your model to all participating processes
    """
    # # ensures the processes to be syncronized
    # torch.distributed.barrier()
    # distribute over all processes
    map_location = {"cuda:0": f"cuda:{rank}"}
    # load the model checkpoint
    if checkpoint_path:
        model.load_state_dict(checkpoint["model_state_dict"])
        trained_epoch = int(checkpoint["num_epoch"])
        history = checkpoint["history"]
        print(f"\nCheckpoint weights loaded on GPU: {rank}")
    else:
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
        checkpoint = torch.load(pretrained_model_path, map_location=map_location)
        # load weights
        model.load_state_dict(checkpoint)
        # starting training from "scratch"
        trained_epoch = 0
        history = defaultdict(list)
        print(f"\nPretrained weights loaded on GPU: {rank}")

    return model, optimizer, trained_epoch, history


# data loader for train data using DDP
def load_data(train_dir, val_dir, batch_size, num_gpus):
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

    # initialize the DistributedSampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    # generate dataloader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,  # local batch size of each GPU
        shuffle=False,  # sampler is already taking care of shuffling the data
        num_workers=4 * num_gpus,
        pin_memory=True,  # data transport is happening in a specifically defined part of the GPU memory
        sampler=train_sampler,
    )

    # get the val dataset
    val_data = zoetrope_dataset.ZoetropeDataset(
        val_dir,
        data_transforms["val"],
    )
    print(f"Val dataset loaded successfully - contains {len(val_data)} images")
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_data)
    # generate dataloader
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,  # local batch size of each GPU
        shuffle=False,  # sampler is already taking care of shuffling the data
        num_workers=4 * num_gpus,
        pin_memory=True,  # data transport is happening in a specifically defined part of the GPU memory
        sampler=val_sampler,
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
def run_training_process_on_given_gpu(
    rank,  # rank has to be the first argument
    num_gpus,
    model_type,
    train_dir,
    val_dir,
    checkpoint_freq,
    checkpoint_path,
    num_epochs,
    batch_size,
    output_dir,
    verbose,
):
    torch.cuda.set_device(rank)
    # initialize model
    model = mobileone(variant=f"{model_type}")
    if verbose:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nMobileOne-{model_type} model initialized successfully on GPU {rank}")
        print(f"Model contains {total_params} total parameters")

    # a process group is initialized in each process
    torch.distributed.init_process_group(
        backend="nccl",  # choose between gloo and nccl, where nccl is better for multi-GPU training
        # more here: https://pytorch.org/docs/stable/distributed.html
        rank=rank,
        world_size=num_gpus,
        init_method="env://",  # pulls all informations it needs from the environment
    )

    # get model and initialized loss function into memory for each GPU
    # load pretrained model or previous checkpoint weights
    model, trained_epoch, history = load_model(rank, model, model_type, checkpoint_path)
    # change the last FC layer
    model.linear = torch.nn.Linear(2048, 2)
    model.cuda(rank)
    # prepare model for distributed training
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # label smoothing regularization with cross entropy loss with smoothing factor set to 0.1
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    criterion.cuda(rank)

    # define training attributes as in paper - problem: weight decay scheduler not available in pytorch
    # SGD with momentum optimizer, initial learning rate is 0.1, initial weight decay is 10e-4
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=10e-4)
    # learning rate annealed using a cosine schedule
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer
    # )

    # quick experiment with adam
    optimizer = torch.optim.AdamW(model.parameters())
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # load data
    dataloaders, dataset_sizes = load_data(train_dir, val_dir, batch_size, num_gpus)

    # start training
    for epoch in range(trained_epoch, trained_epoch + num_epochs):
        # only print message on the first device
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{trained_epoch+num_epochs}")

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            # stats in current epoch and current phase
            batch_losses = 0.0
            batch_num_correct = 0

            # iterate over data
            for images, labels in tqdm(dataloaders[phase], desc=f"{phase} Batch"):
                # put data to device
                images = images.cude(
                    rank,
                    non_blocking=True,  # dataloader doesn't wait with the next command until the data is shifted
                )
                labels = labels.cuda(
                    rank,
                    non_blocking=True,  # dataloader doesn't wait with the next command until the data is shifted
                )

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

                # stats
                batch_losses += loss.item() * inputs.size(0)
                batch_num_correct += torch.sum(preds == labels.data)

            epoch_loss = batch_losses / dataset_sizes[phase]
            epoch_acc = batch_num_correct.double() / dataset_sizes[phase]

            if rank == 0:
                history[f"{phase}_loss"].append(epoch_loss)
                history[f"{phase}_accuracy"].append(epoch_acc)

        # save checkpoint every 5 epochs
        if epoch % checkpoint_freq == 0 and rank == 0:
            save_model(output_dir, model, model_type, epoch, optimizer, history)


def main():
    # input arguments
    parser = argparse.ArgumentParser()
    # model type from CLIP models
    parser.add_argument("--model_type", action="store", dest="model_type", default="s2")
    # training dataset ditectory
    parser.add_argument(
        "--train_dir", action="store", dest="train_dir", default="data/train/"
    )
    # validation dataset ditectory
    parser.add_argument(
        "--val_dir", action="store", dest="val_dir", default="data/test/"
    )
    # number of training epochs
    parser.add_argument("--num_epochs", action="store", dest="num_epochs", default=300)
    # test batch size
    parser.add_argument("--batch_size", action="store", dest="batch_size", default=256)
    # frequency for saving checkpoint model
    parser.add_argument(
        "--checkpoint_freq", action="store", dest="checkpoint_freq", default=1
    )
    # input model path for continue training
    parser.add_argument("--checkpoint_path", action="store", dest="checkpoint_path")
    # output ditectory for model checkpoint, etc
    parser.add_argument(
        "--output_dir", action="store", dest="output_dir", default="results/"
    )
    # verbosity
    parser.add_argument(
        "-v", "--verbose", action="store_true", dest="verbose", default=True
    )

    # load basic arguments
    args = parser.parse_args()
    # we have to get pre-trained model type
    model_type = args.model_type
    if model_type not in ["s0", "s1", "s2", "s3", "s4"]:
        raise ValueError(
            f"Invalid model type: {model_type}, available models: 's0', 's1', 's2', 's3', 's4'"
        )
    train_dir = args.train_dir
    val_dir = args.val_dir
    num_epochs = int(args.num_epochs)
    batch_size = int(args.batch_size)
    checkpoint_freq = int(args.checkpoint_freq)
    checkpoint_path = args.checkpoint_path
    output_dir = args.output_dir
    verbose = args.verbose

    # cuda information
    num_gpus = torch.cuda.device_count()
    print(f"\nTraining using {num_gpus} GPUs")

    # set the environment variables master address and master port
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # spawn multiple processes, one for each GPU
    torch.multiprocessing.spawn(
        run_training_process_on_given_gpu,  # contain training for each GPU
        args=(
            num_gpus,
            model_type,
            train_dir,
            val_dir,
            checkpoint_freq,
            checkpoint_path,
            num_epochs,
            batch_size,
            output_dir,
            verbose,
        ),  # no rank argument here
        nprocs=num_gpus,  # number of process to be spawned
        join=True,
    )


if __name__ == "__main__":
    main()
