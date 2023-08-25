import torch
from fairseq.models.roberta import RobertaModel


class RoBERTa(torch.nn.Module):
    def __init__(self, load_pretrain):
        super(RoBERTa, self).__init__()
        self.text_encoder = RobertaModel

    def forward(self, input_ids, attention_mask):
        text_features = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        return text_features
