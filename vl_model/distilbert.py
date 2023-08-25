import torch
import transformers
from transformers import DistilBertModel, DistilBertConfig


class DistilBERTEncoder(torch.nn.Module):
    def __init__(self, load_pretrain):
        super(DistilBERTEncoder, self).__init__()
        if load_pretrain:
            self.text_encoder = DistilBertModel.from_pretrained(
                "distilbert-base-uncased"
            )
        else:
            configuration = DistilBertConfig()
            self.text_encoder = DistilBertModel(configuration)

    def forward(self, input_ids, attention_mask):
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        ).last_hidden_state

        return text_features
