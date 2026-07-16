import torch.nn as nn
import torch
import math
import torch.nn.functional as F


class trans_reg_model(nn.Module):

    def __init__(self, seq_dim, inp_embed_dim, embed_dim, num_layers, hidden_dim, n_heads, num_lay_trans, regression_head, dropout=0.0):

        super().__init__()
        self.seq_dim = seq_dim
        self.hidden_layer = hidden_dim
        self.num_layers = num_layers
        self.embed_dim = embed_dim 
        layers = []
        layers = [nn.Linear(inp_embed_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]

        for _ in range(self.num_layers):
            layers.extend([
            nn.Linear(self.hidden_layer, self.hidden_layer),
            nn.GELU(),
            nn.Dropout(dropout)])
        
        layers.append(nn.Linear(self.hidden_layer, embed_dim))
        self.embed = nn.Sequential(*layers)

        self.positional_encoding = nn.Parameter(
            torch.zeros(1, seq_dim, embed_dim) 
        )
        nn.init.normal_(self.positional_encoding, mean=0, std=0.1)
        self.scale_pre_pos = math.sqrt(self.embed_dim)

        causal = torch.triu(torch.full((seq_dim, seq_dim), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True, dropout=dropout, dim_feedforward=4*embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_lay_trans)

        # attention pooling
        self.q_pooling = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal(self.q_pooling, mean=0, std=0.1)
        self.pool_scale_term = torch.tensor(self.embed_dim, dtype=torch.float32)

        self.pre_head_norm = nn.LayerNorm(embed_dim) # embedding level normalization by centering and reducing each embedding dimension wrt to all embedding dimension.
        self.fc = nn.Sequential(nn.Linear(embed_dim, regression_head), nn.GELU(), nn.Linear(regression_head, 1))

    def pooling(self, out_trans):
        s = (out_trans@self.q_pooling)/torch.sqrt(self.pool_scale_term)
        alpha = torch.exp(s - torch.logsumexp(s, dim=1, keepdim=True))
        return (out_trans*alpha.unsqueeze(-1)).sum(dim=1)

    def forward(self, x):    
        embed = x * self.scale_pre_pos + self.positional_encoding # scale_pre_pos is to remove if self.embed() is not used!
        
        out_trans = self.transformer_encoder(embed, mask=self.causal_mask)
        out_trans_summary = out_trans[:, -1, :] 

        h = self.pre_head_norm(out_trans_summary)
        out = self.fc(h).squeeze(-1)
        return out 


    def loss_fun(self, pred, target):
        
        return F.mse_loss(pred, target) #F.smooth_l1_loss(pred, target) 


class lstm_reg_model(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        regression_head: int,
        dropout: float = 0.0,
        proj_dim: int | None = None,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.proj_dim = proj_dim if proj_dim is not None else input_dim

        if proj_dim is None:
            self.input_proj = nn.Identity()
        else:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.lstm = nn.LSTM(
            input_size=self.proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.pre_head_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, regression_head),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regression_head, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1.0
                n = param.shape[0]
                param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        returns: (B,)
        """
        x = self.input_proj(x)
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]                       # last layer hidden state, shape (B, hidden_dim)
        h = self.pre_head_norm(h)
        out = self.fc(h).squeeze(-1)
        return out

    def loss_fun(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target)
