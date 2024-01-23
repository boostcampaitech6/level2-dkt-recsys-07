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
        if args.cont_cols == []:
            hd *= 2
        self.comb_proj = nn.Linear(intd * (len(self.embedding_dict)), hd)
        self.cont_proj = nn.Linear(len(self.args.cont_cols), hd)
        self.category_layer_normalization = nn.LayerNorm(hd)
        self.continuous_layer_normalization = nn.LayerNorm(hd)

        # Fully connected layer
        self.fc = nn.Linear(2 * args.hidden_dim, 1)

    def forward(self, **input):
        batch_size = input["Interaction"].size(0)
        # Categorical cols
        embeddings = []
        for col in self.args.cate_cols:
            embeddings.append(self.embedding_dict[col](input[col].int()))
        embeddings.append(
            self.embedding_dict["Interaction"](input["Interaction"].int())
        )
        embed = torch.cat(embeddings, dim=2)
        X = self.comb_proj(embed)
        X = self.category_layer_normalization(X)
        # Continuos cols
        if self.args.cont_cols:
            conts = []
            for col in self.args.cont_cols:
                conts.append(input[col].float())
            if len(self.args.cont_cols) > 1:
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
        self.lstm = nn.LSTM(
            2 * args.hidden_dim, 2 * args.hidden_dim, args.n_layers, batch_first=True
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
        self.n_heads = args.n_heads
        self.drop_out = args.drop_out
        self.lstm = nn.LSTM(
            2 * args.hidden_dim, 2 * args.hidden_dim, args.n_layers, batch_first=True
        )
        self.config = BertConfig(
            3,  # not used
            hidden_size=2 * args.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=2 * args.hidden_dim,
            hidden_dropout_prob=args.drop_out,
            attention_probs_dropout_prob=args.drop_out,
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
        self.n_heads = args.n_heads
        self.drop_out = args.drop_out
        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=2 * args.hidden_dim,
            num_hidden_layers=args.n_layers,
            num_attention_heads=args.n_heads,
            max_position_embeddings=args.max_seq_len,
        )
        self.encoder = BertModel(self.config)

    def forward(self, **input):
        X, batch_size = super().forward(**input)

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=input["mask"])
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, 2 * self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out

class MF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.user_dims = args.n_cate['userID']
        self.item_dims = args.n_cate['assessmentItemID']
        # hidden_dim을 embeddding dim으로 사용
        self.user_embedding = nn.Embedding(self.user_dims, args.hidden_dim)
        self.item_embedding = nn.Embedding(self.item_dims, args.hidden_dim)
        torch.nn.init.xavier_normal_(self.user_embedding.weight.data)
        torch.nn.init.xavier_normal_(self.item_embedding.weight.data)
        # 전체 정답률 평균
        self.mu = 0.6546385
        self.b_u = nn.Parameter(torch.zeros(self.user_dims))
        self.b_i = nn.Parameter(torch.zeros(self.item_dims))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        uid = x[:, 0]
        iid = x[:, 1]

        user_x = self.user_embedding(uid)
        item_x = self.item_embedding(iid)
        dot = (user_x * item_x).sum(dim = 1)
        return self.mu + dot + self.b_u[uid] + self.b_i[iid] + self.b
        
class LMF(MF):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, x):
        uid = x[:, 0]
        iid = x[:, 1]

        user_x = self.user_embedding(uid)
        item_x = self.item_embedding(iid)
        dot = (user_x * item_x).sum(dim = 1)
        logit = dot + self.b_u[uid] + self.b_i[iid]
        return torch.exp(logit) / (1 + torch.exp(logit))