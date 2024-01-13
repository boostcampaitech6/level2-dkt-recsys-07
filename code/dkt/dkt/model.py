import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel


class ModelBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.n_layers = args.n_layers

        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = self.hidden_dim, self.hidden_dim // 3
        self.embedding_dict = nn.ModuleDict(
            {
                col: nn.Embedding(args.n_cate[col] + 1, intd)
                for col in self.args.cate_cols
            }
        )
        self.embedding_dict["Interaction"] = nn.Embedding(3, intd)

        # Concatentaed Embedding Projection
        self.comb_proj = nn.Linear(intd * (len(self.embedding_dict)), hd)

        # Fully connected layer
        self.fc = nn.Linear(hd, 1)

    def forward(self, **input):
        batch_size = input["Interaction"].size(0)
        # Embedding
        embeddings = []
        for col in self.args.cate_cols:
            embeddings.append(self.embedding_dict[col](input[col].int()))
        embeddings.append(
            self.embedding_dict["Interaction"](input["Interaction"].int())
        )
        embed = torch.cat(embeddings, dim=2)
        X = self.comb_proj(embed)
        return X, batch_size


class LSTM(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.lstm = nn.LSTM(
            args.hidden_dim, args.hidden_dim, args.n_layers, batch_first=True
        )

    def forward(self, **input):
        X, batch_size = super().forward(**input)
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.n_heads = args.n_heads
        self.drop_out = args.drop_out
        self.lstm = nn.LSTM(
            args.hidden_dim, args.hidden_dim, args.n_layers, batch_first=True
        )
        self.config = BertConfig(
            3,  # not used
            hidden_size=args.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=args.hidden_dim,
            hidden_dropout_prob=args.drop_out,
            attention_probs_dropout_prob=args.drop_out,
        )
        self.attn = BertEncoder(self.config)

    def forward(self, **input):
        X, batch_size = super().forward(**input)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = input["mask"].unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class BERT(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.n_heads = args.n_heads
        self.drop_out = args.drop_out
        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=args.hidden_dim,
            num_hidden_layers=args.n_layers,
            num_attention_heads=args.n_heads,
            max_position_embeddings=args.max_seq_len,
        )
        self.encoder = BertModel(self.config)

    def forward(self, **input):
        X, batch_size = super().forward(**input)

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=input["mask"])
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out
