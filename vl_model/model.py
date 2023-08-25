# construct our vision-language model
import os
import sys
import torch
import itertools

from mobileone import mobileone_encoder, reparameterize_model
from distilbert import DistilBERTEncoder


# helper function that runs linux command to download model if needed
def runcmd(cmd, verbose=False, *args, **kwargs):
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass


# to use view in Sequential
class Flatten(torch.nn.Module):
    def forward(self, input):
        """
        Note that input.size(0) is usually the batch size.
        So what it does is that given any input with input.size(0) # of batches,
        will flatten to be 1 * nb_elements.
        """
        batch_size = input.size(0)
        out = input.view(batch_size, -1)
        return out  # (batch_size, *size)


class SelfAttentionLayer(torch.torch.nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        # add position embedding
        self.positional_embedding = torch.torch.nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        # Query, Key, and Value are each passed through separate Linear layers
        self.q_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.k_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.v_proj = torch.nn.Linear(embed_dim, embed_dim)
        self.c_proj = torch.nn.Linear(embed_dim, output_dim or embed_dim)
        # multi-head attention
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        attn_output, attn_weight = torch.nn.functional.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True,
        )
        return x.squeeze(0), attn_weight[:, 0, :-1]


class VisionLanguageSafetyModel(torch.nn.Module):
    def __init__(
        self,
        vision_encoder_name,
        text_encoder_name,
        num_classes=2,
        load_pretrain=False,
        num_heads=8,
    ):
        super(VisionLanguageSafetyModel, self).__init__()
        self.vision_encoder_name = vision_encoder_name
        self.text_encoder_name = text_encoder_name
        self.load_pretrain = load_pretrain

        # get vision and text encoder
        self.vision_encoder = self.get_vision_encoder()
        self.text_encoder = self.get_text_encoder()

        # text encoder dim
        if text_encoder_name == "distilbert":
            text_dim = 768

        # image embeddings conv
        if "mobileone" in vision_encoder_name:
            if self.mobileone_type == "s4":
                self.vision_dim = 2048
                # (2048, 7, 7) -> (2048, 11, 11)
                self.vision_conv = torch.torch.nn.Conv2d(
                    in_channels=self.vision_dim,
                    out_channels=self.vision_dim,
                    kernel_size=3,
                    padding=3,
                )

        # conv layer on fusing multimodal features
        self.multimodal_fusion = torch.nn.Sequential(
            torch.torch.nn.Conv2d(
                in_channels=text_dim + self.vision_dim,
                out_channels=self.vision_dim,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.torch.nn.Conv2d(
                in_channels=self.vision_dim,
                out_channels=self.vision_dim // 2,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
        )

        # multihead self attention layer
        embed_dim = 1024
        self.spacial_dim = 11
        self.self_attention = SelfAttentionLayer(self.spacial_dim, embed_dim, num_heads)

        # classification head
        self.classification_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            Flatten(),
            torch.nn.Linear(self.vision_dim, self.vision_dim // 2),
            torch.nn.Linear(self.vision_dim // 2, self.vision_dim // 4),
            torch.nn.Linear(self.vision_dim // 4, num_classes),
        )

    # load different vision encoders
    def get_vision_encoder(self):
        # vision encoder
        if "mobileone" in self.vision_encoder_name:
            print(f"\nUsing MobileOne backbone as vision encoder.")
            self.mobileone_type = self.vision_encoder_name.split("-")[-1]
            vision_encoder = mobileone_encoder(variant=self.mobileone_type)
            # load the pretrained model
            if self.load_pretrain:
                cur_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
                pretrained_vision_encoder_path = os.path.join(
                    cur_dir,
                    "mobileone_unfused",
                    f"mobileone_{self.mobileone_type}_unfused.pth.tar",
                )
                if not os.path.exists(pretrained_vision_encoder_path):
                    runcmd(
                        f"wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_{self.mobileone_type}_unfused.pth.tar -P {cur_dir}/mobileone_unfused",
                        verbose=True,
                    )
                pretrain_checkpoint = torch.load(pretrained_vision_encoder_path)
                # load weights except the classifier head
                encoder_checkpoint = {}
                for i, cur_key in enumerate(pretrain_checkpoint.keys()):
                    if i < len(pretrain_checkpoint.keys()) - 2:
                        encoder_checkpoint[cur_key] = pretrain_checkpoint[cur_key]
                vision_encoder.load_state_dict(pretrain_checkpoint)
                print(f"Pretrained weights loaded")

        return vision_encoder

    # prepare text encoder
    def get_text_encoder(self):
        if self.text_encoder_name == "distilbert":
            print(f"\nUsing DistilBERT backbone as text encoder.")
            text_encoder = DistilBERTEncoder(load_pretrain=self.load_pretrain)
            if self.load_pretrain:
                print(f"Pretrained weights loaded")
        elif self.text_encoder_name == "roberta":
            print(f"\nUsing RoBERTa backbone as text encoder.")

        return text_encoder

    def forward(
        self,
        images,
        text_input_ids,
        text_attention_mask,
        text_token_type_ids=None,
    ):
        # get the vision encoder output
        vision_features = self.vision_encoder(images)
        vision_features = self.vision_conv(vision_features)
        # print(f"Vision feature shape: {vision_features.shape}")
        # vision features dim might be (batch_size, 2048)
        # get the text encoder output
        if self.text_encoder_name == "distilbert":
            text_features = self.text_encoder(
                input_ids=text_input_ids,
                attention_mask=text_attention_mask,
            )  # text features dim is (batch_size, seq_len, 768)
            # reshape text features to (batch_size, 768, 11, 11)
            text_features = text_features.view(
                text_features.shape[0], self.spacial_dim, self.spacial_dim, -1
            )
            text_features = text_features.permute((0, 3, 1, 2))
            # print(f"Text feature shape: {text_features.shape}")

        # aggregate and fuse
        in_features = torch.cat((vision_features, text_features), dim=1)
        in_features = self.multimodal_fusion(in_features)

        # cross attention layer
        _, attention_map = self.self_attention(in_features)
        attention_map = attention_map.reshape(
            (attention_map.shape[0], self.spacial_dim, -1)
        )
        attention_map = attention_map.unsqueeze(1).repeat(1, self.vision_dim, 1, 1)

        # element-wise multiplication with vision input
        vision_features = vision_features * attention_map

        # classification head
        out = self.classification_head(vision_features)

        return out


class VisionOnlySafetyModel(torch.nn.Module):
    def __init__(
        self,
        vision_encoder_name,
        num_classes=2,
    ):
        super(VisionOnlySafetyModel, self).__init__()
        self.vision_encoder_name = vision_encoder_name
        if "mobileone" in self.vision_encoder_name:
            print(f"\nUsing MobileOne backbone as vision encoder.")
            self.mobileone_type = self.vision_encoder_name.split("-")[-1]
            if self.mobileone_type == "s4":
                self.vision_dim = 2048
            # get vision encoder
            self.vision_encoder = mobileone_encoder(variant=self.mobileone_type)

        # classification head
        self.classification_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=1),
            Flatten(),
            torch.nn.Linear(self.vision_dim, self.vision_dim // 2),
            torch.nn.Linear(self.vision_dim // 2, self.vision_dim // 4),
            torch.nn.Linear(self.vision_dim // 4, num_classes),
        )

    def forward(self, images):
        x = self.vision_encoder(images)
        x = self.classification_head(x)

        return x
