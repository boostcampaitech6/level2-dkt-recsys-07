import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel


class ModelBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.cate_cols = args.cate_cols
        self.cont_cols = args.cont_cols
        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = args.model.hidden_dim, args.model.hidden_dim // 3
        self.embedding_dict = nn.ModuleDict(
            {
                col: nn.Embedding(args.n_cate[col] + 1, intd)
                for col in self.args.cate_cols
            }
        )
        self.embedding_dict["Interaction"] = nn.Embedding(3, intd)

        # Concatentaed Embedding Projection
        if args.cont_cols == []:
            hd *= 2
        self.comb_proj = nn.Linear(intd * (len(self.embedding_dict)), hd)
        self.cont_proj = nn.Linear(len(self.args.cont_cols), hd)
        self.category_layer_normalization = nn.LayerNorm(hd)
        self.continuous_layer_normalization = nn.LayerNorm(hd)

        # Fully connected layer
        self.fc = nn.Linear(2 * args.model.hidden_dim, 1)

    def forward(self, **input):
        batch_size = input["Interaction"].size(0)
        # Categorical cols
        embeddings = []
        for col in self.cate_cols:
            embeddings.append(self.embedding_dict[col](input[col].int()))
        embeddings.append(
            self.embedding_dict["Interaction"](input["Interaction"].int())
        )
        embed = torch.cat(embeddings, dim=2)
        X = self.comb_proj(embed)
        X = self.category_layer_normalization(X)
        # Continuos cols
        if self.cont_cols:
            conts = []
            for col in self.cont_cols:
                conts.append(input[col].float())
            if len(self.cont_cols) > 1:
                conts = torch.cat(conts, dim=2)
            else:
                conts = conts[-1]
            Y = self.cont_proj(conts)
            Y = self.continuous_layer_normalization(Y)

            X = torch.cat((X, Y), dim=2)
        return X, batch_size


class LSTM(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.hidden_dim = args.model.hidden_dim
        self.lstm = nn.LSTM(
            2 * args.model.hidden_dim,
            2 * args.model.hidden_dim,
            args.model.n_layers,
            batch_first=True,
        )

    def forward(self, **input):
        X, batch_size = super().forward(**input)
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, 2 * self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.hidden_dim = args.model.hidden_dim
        self.n_layers = args.model.n_layers

        self.lstm = nn.LSTM(
            2 * args.model.hidden_dim,
            2 * args.model.hidden_dim,
            args.model.n_layers,
            batch_first=True,
        )
        self.config = BertConfig(
            3,  # not used
            hidden_size=2 * args.model.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=args.model.n_heads,
            intermediate_size=2 * args.model.hidden_dim,
            hidden_dropout_prob=args.model.drop_out,
            attention_probs_dropout_prob=args.model.drop_out,
        )
        self.attn = BertEncoder(self.config)

    def forward(self, **input):
        X, batch_size = super().forward(**input)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, 2 * self.hidden_dim)

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
        self.hidden_dim = args.model.hidden_dim
        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=2 * args.model.hidden_dim,
            num_hidden_layers=args.model.n_layers,
            num_attention_heads=args.model.n_heads,
            max_position_embeddings=args.model.max_seq_len,
        )
        self.encoder = BertModel(self.config)

    def forward(self, **input):
        X, batch_size = super().forward(**input)

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=input["mask"])
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, 2 * self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LastQueryTransformerLSTM(ModelBase):
    def __init__(self, args):
        super().__init__(args)
        self.hidden_dim = args.model.hidden_dim
        self.position_embedding = nn.Embedding(
            1 + args.model.max_seq_len, 2 * args.model.hidden_dim
        )
        self.mha = nn.MultiheadAttention(
            embed_dim=2 * args.model.hidden_dim,
            num_heads=args.model.n_heads,
            dropout=args.model.drop_out,
            batch_first=True,
        )
        self.lstm = nn.LSTM(
            2 * args.model.hidden_dim,
            2 * args.model.hidden_dim,
            args.model.n_layers,
            batch_first=True,
        )

    def forward(self, **input):
        X, batch_size = super().forward(**input)
        P = self.position_embedding(input["Position"])
        X = X + P
        Y, _ = self.mha(X[:, -1, :].view(batch_size, -1, 2 * self.hidden_dim), X, X)
        X = X + Y.view(batch_size, 1, 2 * self.hidden_dim)
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, 2 * self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out
