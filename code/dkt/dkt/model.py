import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertConfig, BertEncoder, BertModel


class ModelBase(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.n_layers = args.n_layers
        # Embeddings
        # hd: Hidden dimension, intd: Intermediate hidden dimension
        hd, intd = args.hidden_dim, args.hidden_dim // 3
        self.embedding_interaction = nn.Embedding(3, intd) # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        # 범주형 데이터 임베딩 레이어 초기화
        self.embeddings = nn.ModuleDict({
            col: nn.Embedding(args.cate_sizes[col] + 1, args.hidden_dim // 3)  # +1 for padding token
            for col in args.cate_cols
        })
        
        self.cate_norm = nn.LayerNorm(args.hidden_dim//2).to(args.device)
        self.cont_norm = nn.LayerNorm(args.hidden_dim//2).to(args.device)
        
        self.only_cate_norm = nn.LayerNorm(args.hidden_dim).to(args.device)
        
        # Concatentaed Embedding Projection
        self.cate_proj = nn.Linear(intd * (len(args.cate_cols)), hd//2)
        self.cont_proj = nn.Linear(len(args.cont_cols), hd//2)

        self.only_cate_proj = nn.Linear(intd * (len(args.cate_cols)), hd)

        # Fully connected layer
        self.fc = nn.Linear(hd, 1)
    
    def forward(self, data):
        batch_size = data["interaction"].size(0)
        
        # self.cate_cols의 복사본 생성
        x_cate_cols = self.args.cate_cols.copy()

        # 복사본에서 "userID" 제거
        if "userID" in x_cate_cols:
            x_cate_cols.remove("userID")

        if x_cate_cols == []:
            x_cont = torch.cat([data[col].unsqueeze(2) for col in self.args.cont_cols], dim=2)
            embed_interaction = self.embedding_interaction(data["interaction"].int())
            cate = self.cate_norm(self.cate_proj(embed_interaction))
            cont = self.cont_norm(self.cont_proj(x_cont))
            x = torch.cat([cate, cont], dim=2)

        elif self.args.cont_cols == []:
            # 범주형 데이터 임베딩 처리
            x_cate_emb = [self.embeddings[col](data[col].int()) for col in x_cate_cols]
            embed_interaction = self.embedding_interaction(data["interaction"].int())
            x_cate_emb = torch.cat(x_cate_emb + [embed_interaction], dim=-1)
            x = self.only_cate_norm(self.only_cate_proj(x_cate_emb))
        else:
            # 범주형 데이터 임베딩 처리
            x_cate_emb = [self.embeddings[col](data[col].int()) for col in x_cate_cols]
            embed_interaction = self.embedding_interaction(data["interaction"].int())
            x_cate_emb = torch.cat(x_cate_emb + [embed_interaction], dim=-1)
            # 연속형 데이터 처리
            x_cont = torch.cat([data[col].unsqueeze(2) for col in self.args.cont_cols], dim=2)

            cate = self.cate_norm(self.cate_proj(x_cate_emb))
            cont = self.cont_norm(self.cont_proj(x_cont))
            # 데이터 결합 및 예측
            x = torch.cat([cate, cont], dim=2)
        return x, batch_size


class LSTM(ModelBase):
    def __init__(
        self,
        args
    ):
        super().__init__(
            args
        )
        self.lstm = nn.LSTM(
            self.args.hidden_dim, self.args.hidden_dim, self.args.n_layers, batch_first=True
        )

    def forward(self, data):
        X, batch_size = super().forward(data = data)
        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out


class LSTMATTN(ModelBase):
    def __init__(
        self,
        args
    ):
        super().__init__(
            args
        )
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        self.lstm = nn.LSTM(
            self.args.hidden_dim, self.args.hidden_dim, self.args.n_layers, batch_first=True
        )
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.args.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.args.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

    def forward(self, data):
        X, batch_size = super().forward(data=data)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        extended_attention_mask = data["mask"].unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask)
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class BERT(ModelBase):
    def __init__(
        self,
        args
    ):
        super().__init__(
            args
        )
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out
        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.args.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )
        self.encoder = BertModel(self.config)

    def forward(self, data):
        X, batch_size = super().forward(data=data)

        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=data["mask"])
        out = encoded_layers[0]
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)
        return out

