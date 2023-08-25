# Custom dataset of game and violative images to be used in PyTorch
import os
import cv2
import glob
import torch
from PIL import Image
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from transformers import DistilBertTokenizer


# based on model parameters, choose different tokenizers, etc
def get_tokenizer_settings(model_name):
    if model_name == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased", truncation=True, do_lower_case=True
        )
        max_len = 121

    return tokenizer, max_len


# data loader for train data using DDP
def load_data_for_training(train_dir, val_dir, batch_size, text_encoder_name=None):
    # data transforms for images used in both train and val
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
    # tokenizer is when using distilbert
    if text_encoder_name:
        tokenizer, max_len = get_tokenizer_settings(text_encoder_name)
        train_data = VisionLanguageDataset(
            data_dir=train_dir,
            transform=data_transforms["train"],
            tokenizer=tokenizer,
            text_encoder_name=text_encoder_name,
            max_len=max_len,
        )
        print(
            f"\nTrain dataset loaded successfully - contains {len(train_data)} images/texts pairs"
        )
    else:
        train_data = VisionLanguageDataset(
            data_dir=train_dir,
            transform=data_transforms["train"],
        )
        print(
            f"\nTrain dataset loaded successfully - contains {len(train_data)} images"
        )

    # generate dataloader
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    # get the val dataset
    if text_encoder_name:
        val_data = VisionLanguageDataset(
            data_dir=val_dir,
            transform=data_transforms["val"],
            tokenizer=tokenizer,
            text_encoder_name=text_encoder_name,
            max_len=max_len,
        )
    else:
        val_data = VisionLanguageDataset(
            data_dir=val_dir,
            transform=data_transforms["val"],
        )
    print(
        f"Validation dataset loaded successfully - contains {len(val_data)} images/texts pairs"
    )
    # generate dataloader
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    # combine into dictionary
    train_val_datasets = {
        "train": train_data,
        "val": val_data,
    }
    dataloaders = {
        "train": train_loader,
        "val": val_loader,
    }
    dataset_sizes = {x: len(train_val_datasets[x]) for x in ["train", "val"]}

    return dataloaders, dataset_sizes


# data loader for train data using DDP
def load_data_for_testing(test_dir, batch_size, text_encoder_name=None):
    # data transforms for images used in both train and val
    # data augmentation schemes follow https://github.com/open-mmlab/mmpretrain/blob/1.x/configs/mobileone/mobileone-s2_8xb32_in1k.py
    data_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # get the train dataset
    # tokenizer is when using distilbert
    if text_encoder_name:
        tokenizer, max_len = get_tokenizer_settings(text_encoder_name)
        test_data = VisionLanguageDataset(
            data_dir=test_dir,
            transform=data_transform,
            tokenizer=tokenizer,
            text_encoder_name=text_encoder_name,
            max_len=max_len,
        )
        print(
            f"\nTest dataset loaded successfully - contains {len(test_data)} images/texts pairs"
        )
    else:
        test_data = VisionLanguageDataset(
            data_dir=test_dir,
            transform=data_transform,
        )
        print(f"\nTest dataset loaded successfully - contains {len(test_data)} images")

    # generate dataloader
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    test_dataset_size = len(test_data)

    return test_loader, test_dataset_size


class VisionLanguageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        text_encoder_name=None,
        tokenizer=None,
        max_len=121,
    ):
        self.all_images_paths = glob.glob(
            os.path.join(data_dir, "**/*.png"), recursive=True
        )
        self.transform = transform
        self.text_encoder_name = text_encoder_name
        if self.text_encoder_name:
            self.all_texts_paths = [
                cur_image_path.replace(".png", ".txt")
                for cur_image_path in self.all_images_paths
            ]
            self.tokenizer = tokenizer
            self.max_len = max_len

    def __len__(self):
        if self.text_encoder_name:
            assert len(self.all_images_paths) == len(self.all_texts_paths)
        return len(self.all_images_paths)

    def __getitem__(self, index):
        cur_img_path = self.all_images_paths[index]
        # load image
        image = cv2.imread(cur_img_path)
        image = Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)

        # identify label based on the input path
        if "game" in cur_img_path:
            label = torch.tensor(0)  # 0 for game
        elif "violative" in cur_img_path:
            label = torch.tensor(1)  # 1 for violative
        else:
            raise Exception(f"Unknown label for image {cur_img_path}")

        # load text
        if self.text_encoder_name:
            text_file = open(self.all_texts_paths[index], mode="r")
            text = text_file.read()
            text_file.close()
            # tokenize text
            if self.text_encoder_name == "distilbert":
                text_inputs = self.tokenizer.encode_plus(
                    text=text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_len,
                    return_token_type_ids=True,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                ids = text_inputs["input_ids"]
                mask = text_inputs["attention_mask"]
                token_type_ids = text_inputs["token_type_ids"]

                return {
                    "images": image,
                    "ids": ids,
                    "masks": mask,
                    "token_type_ids": token_type_ids,
                    "labels": label,
                }
        # when we are generating datasets for vision only
        elif self.text_encoder_name == None:
            return {
                "images": image,
                "labels": label,
            }
